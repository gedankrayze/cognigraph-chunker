pub mod providers;

use std::sync::LazyLock;

use pyo3::prelude::*;

use cognigraph_chunker::embeddings::EmbeddingProvider;
use cognigraph_chunker::semantic::{SemanticConfig as RustSemanticConfig, semantic_chunk, semantic_chunk_plain};

use crate::error::to_py_err;
use crate::signal::PyFilteredIndices;
use providers::{PyOllamaProvider, PyOnnxProvider, PyOpenAiProvider};

/// Shared Tokio runtime for all Python → async Rust calls.
static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime")
});

#[pyclass(name = "SemanticConfig")]
#[derive(Clone)]
pub struct PySemanticConfig {
    #[pyo3(get, set)]
    pub sim_window: usize,
    #[pyo3(get, set)]
    pub sg_window: usize,
    #[pyo3(get, set)]
    pub poly_order: usize,
    #[pyo3(get, set)]
    pub threshold: f64,
    #[pyo3(get, set)]
    pub min_distance: usize,
    #[pyo3(get, set)]
    pub max_blocks: usize,
}

#[pymethods]
impl PySemanticConfig {
    #[new]
    #[pyo3(signature = (*, sim_window=3, sg_window=11, poly_order=3, threshold=0.5, min_distance=2, max_blocks=10_000))]
    fn new(
        sim_window: usize,
        sg_window: usize,
        poly_order: usize,
        threshold: f64,
        min_distance: usize,
        max_blocks: usize,
    ) -> Self {
        Self {
            sim_window,
            sg_window,
            poly_order,
            threshold,
            min_distance,
            max_blocks,
        }
    }
}

impl From<&PySemanticConfig> for RustSemanticConfig {
    fn from(py: &PySemanticConfig) -> Self {
        Self {
            sim_window: py.sim_window,
            sg_window: py.sg_window,
            poly_order: py.poly_order,
            threshold: py.threshold,
            min_distance: py.min_distance,
            max_blocks: py.max_blocks,
        }
    }
}

#[pyclass(name = "SemanticResult")]
pub struct PySemanticResult {
    #[pyo3(get)]
    pub chunks: Vec<(String, usize)>,
    #[pyo3(get)]
    pub similarities: Vec<f64>,
    #[pyo3(get)]
    pub smoothed: Vec<f64>,
    #[pyo3(get)]
    pub split_indices: PyFilteredIndices,
}

/// Run semantic chunking with a provider.
///
/// Dispatches to the concrete provider type since EmbeddingProvider uses AFIT.
#[pyfunction]
#[pyo3(signature = (text, provider, config=None, *, markdown=true))]
pub fn py_semantic_chunk(
    py: Python<'_>,
    text: &str,
    provider: &Bound<'_, PyAny>,
    config: Option<&PySemanticConfig>,
    markdown: bool,
) -> PyResult<PySemanticResult> {
    let rust_config = config
        .map(RustSemanticConfig::from)
        .unwrap_or_default();

    let text_owned = text.to_string();

    // Dispatch by concrete provider type.
    // GIL is held during the blocking call. This is acceptable since the work is
    // CPU/IO bound (network requests for embeddings) and Python threads can't run
    // Rust code concurrently anyway.
    let _ = py;
    if let Ok(p) = provider.downcast::<PyOllamaProvider>() {
        let provider_ref = p.borrow();
        return run_semantic(&text_owned, &provider_ref.inner, &rust_config, markdown);
    }
    if let Ok(p) = provider.downcast::<PyOpenAiProvider>() {
        let provider_ref = p.borrow();
        return run_semantic(&text_owned, &provider_ref.inner, &rust_config, markdown);
    }
    if let Ok(p) = provider.downcast::<PyOnnxProvider>() {
        let provider_ref = p.borrow();
        return run_semantic(&text_owned, &provider_ref.inner, &rust_config, markdown);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "provider must be OllamaProvider, OpenAiProvider, or OnnxProvider",
    ))
}

fn run_semantic<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    config: &RustSemanticConfig,
    markdown: bool,
) -> PyResult<PySemanticResult> {
    let result = RUNTIME.block_on(async {
        if markdown {
            semantic_chunk(text, provider, config).await
        } else {
            semantic_chunk_plain(text, provider, config).await
        }
    }).map_err(to_py_err)?;

    Ok(PySemanticResult {
        chunks: result.chunks,
        similarities: result.similarities,
        smoothed: result.smoothed,
        split_indices: PyFilteredIndices {
            indices: result.split_indices.indices,
            values: result.split_indices.values,
        },
    })
}
