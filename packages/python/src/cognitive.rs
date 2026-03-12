//! Python bindings for cognitive chunking.

use pyo3::prelude::*;

use cognigraph_chunker::embeddings::EmbeddingProvider;
use cognigraph_chunker::semantic::cognitive_types::{
    CognitiveConfig as RustCognitiveConfig, CognitiveWeights,
};
use cognigraph_chunker::semantic::{cognitive_chunk, cognitive_chunk_plain};

use crate::error::to_py_err;
use crate::semantic::RUNTIME;
use crate::semantic::providers::{PyOllamaProvider, PyOnnxProvider, PyOpenAiProvider};

#[pyclass(name = "CognitiveConfig", from_py_object)]
#[derive(Clone)]
pub struct PyCognitiveConfig {
    #[pyo3(get, set)]
    pub soft_budget: usize,
    #[pyo3(get, set)]
    pub hard_budget: usize,
    #[pyo3(get, set)]
    pub sim_window: usize,
    #[pyo3(get, set)]
    pub sg_window: usize,
    #[pyo3(get, set)]
    pub poly_order: usize,
    #[pyo3(get, set)]
    pub max_blocks: usize,
    #[pyo3(get, set)]
    pub emit_signals: bool,
    #[pyo3(get, set)]
    pub language: Option<String>,
}

#[pymethods]
impl PyCognitiveConfig {
    #[new]
    #[pyo3(signature = (*, soft_budget=512, hard_budget=768, sim_window=3, sg_window=11, poly_order=3, max_blocks=10_000, emit_signals=false, language=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        soft_budget: usize,
        hard_budget: usize,
        sim_window: usize,
        sg_window: usize,
        poly_order: usize,
        max_blocks: usize,
        emit_signals: bool,
        language: Option<String>,
    ) -> Self {
        Self {
            soft_budget,
            hard_budget,
            sim_window,
            sg_window,
            poly_order,
            max_blocks,
            emit_signals,
            language,
        }
    }
}

impl From<&PyCognitiveConfig> for RustCognitiveConfig {
    fn from(py: &PyCognitiveConfig) -> Self {
        let language = py.language.as_deref().and_then(|s| s.parse().ok());
        Self {
            weights: CognitiveWeights::default(),
            soft_budget: py.soft_budget,
            hard_budget: py.hard_budget,
            sim_window: py.sim_window,
            sg_window: py.sg_window,
            poly_order: py.poly_order,
            max_blocks: py.max_blocks,
            emit_signals: py.emit_signals,
            language,
        }
    }
}

/// A cognitive chunk returned to Python.
#[pyclass(name = "CognitiveChunk", from_py_object)]
#[derive(Clone)]
pub struct PyCognitiveChunk {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub chunk_index: usize,
    #[pyo3(get)]
    pub offset_start: usize,
    #[pyo3(get)]
    pub offset_end: usize,
    #[pyo3(get)]
    pub heading_path: Vec<String>,
    #[pyo3(get)]
    pub dominant_entities: Vec<String>,
    #[pyo3(get)]
    pub dominant_relations: Vec<PyRelationTriple>,
    #[pyo3(get)]
    pub token_estimate: usize,
    #[pyo3(get)]
    pub continuity_confidence: f64,
    #[pyo3(get)]
    pub synopsis: Option<String>,
    #[pyo3(get)]
    pub prev_chunk: Option<usize>,
    #[pyo3(get)]
    pub next_chunk: Option<usize>,
}

/// A relation triple.
#[pyclass(name = "RelationTriple", from_py_object)]
#[derive(Clone)]
pub struct PyRelationTriple {
    #[pyo3(get)]
    pub subject: String,
    #[pyo3(get)]
    pub predicate: String,
    #[pyo3(get)]
    pub object: String,
    #[pyo3(get)]
    pub confidence: f64,
}

/// Result of cognitive chunking.
#[pyclass(name = "CognitiveResult")]
pub struct PyCognitiveResult {
    #[pyo3(get)]
    pub chunks: Vec<PyCognitiveChunk>,
    #[pyo3(get)]
    pub block_count: usize,
    #[pyo3(get)]
    pub shared_entities: std::collections::HashMap<String, Vec<usize>>,
}

/// Run cognitive chunking with a provider.
#[pyfunction]
#[pyo3(signature = (text, provider, config=None, *, markdown=true))]
pub fn py_cognitive_chunk(
    py: Python<'_>,
    text: &str,
    provider: &Bound<'_, PyAny>,
    config: Option<&PyCognitiveConfig>,
    markdown: bool,
) -> PyResult<PyCognitiveResult> {
    let rust_config = config.map(RustCognitiveConfig::from).unwrap_or_default();
    let text_owned = text.to_string();

    let _ = py;
    if let Ok(p) = provider.cast::<PyOllamaProvider>() {
        let provider_ref = p.borrow();
        return run_cognitive(&text_owned, &provider_ref.inner, &rust_config, markdown);
    }
    if let Ok(p) = provider.cast::<PyOpenAiProvider>() {
        let provider_ref = p.borrow();
        return run_cognitive(&text_owned, &provider_ref.inner, &rust_config, markdown);
    }
    if let Ok(p) = provider.cast::<PyOnnxProvider>() {
        let provider_ref = p.borrow();
        return run_cognitive(&text_owned, &provider_ref.inner, &rust_config, markdown);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "provider must be OllamaProvider, OpenAiProvider, or OnnxProvider",
    ))
}

fn run_cognitive<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    config: &RustCognitiveConfig,
    markdown: bool,
) -> PyResult<PyCognitiveResult> {
    let result = RUNTIME
        .block_on(async {
            if markdown {
                cognitive_chunk(text, provider, config).await
            } else {
                cognitive_chunk_plain(text, provider, config).await
            }
        })
        .map_err(to_py_err)?;

    let chunks = result
        .chunks
        .into_iter()
        .map(|c| PyCognitiveChunk {
            text: c.text,
            chunk_index: c.chunk_index,
            offset_start: c.offset_start,
            offset_end: c.offset_end,
            heading_path: c.heading_path,
            dominant_entities: c.dominant_entities,
            dominant_relations: c
                .dominant_relations
                .into_iter()
                .map(|r| PyRelationTriple {
                    subject: r.subject,
                    predicate: r.predicate,
                    object: r.object,
                    confidence: 1.0,
                })
                .collect(),
            token_estimate: c.token_estimate,
            continuity_confidence: c.continuity_confidence,
            synopsis: c.synopsis,
            prev_chunk: c.prev_chunk,
            next_chunk: c.next_chunk,
        })
        .collect();

    Ok(PyCognitiveResult {
        chunks,
        block_count: result.block_count,
        shared_entities: result.shared_entities,
    })
}
