//! Python bindings for quality metrics evaluation.

use pyo3::prelude::*;

use cognigraph_chunker::embeddings::EmbeddingProvider;
use cognigraph_chunker::semantic::quality_metrics::{
    ChunkForEval, MetricConfig, MetricWeights, QualityMetrics, evaluate_chunks,
};

use crate::error::to_py_err;
use crate::semantic::RUNTIME;
use crate::semantic::providers::{PyOllamaProvider, PyOnnxProvider, PyOpenAiProvider};

// ── PyMetricWeights ───────────────────────────────────────────────────────────

#[pyclass(name = "MetricWeights", from_py_object)]
#[derive(Clone)]
pub struct PyMetricWeights {
    #[pyo3(get, set)]
    pub sc: f64,
    #[pyo3(get, set)]
    pub icc: f64,
    #[pyo3(get, set)]
    pub dcc: f64,
    #[pyo3(get, set)]
    pub bi: f64,
    #[pyo3(get, set)]
    pub rc: f64,
}

#[pymethods]
impl PyMetricWeights {
    #[new]
    #[pyo3(signature = (*, sc=0.20, icc=0.20, dcc=0.20, bi=0.20, rc=0.20))]
    fn new(sc: f64, icc: f64, dcc: f64, bi: f64, rc: f64) -> Self {
        Self {
            sc,
            icc,
            dcc,
            bi,
            rc,
        }
    }
}

impl From<&PyMetricWeights> for MetricWeights {
    fn from(py: &PyMetricWeights) -> Self {
        Self {
            sc: py.sc,
            icc: py.icc,
            dcc: py.dcc,
            bi: py.bi,
            rc: py.rc,
        }
    }
}

impl From<MetricWeights> for PyMetricWeights {
    fn from(m: MetricWeights) -> Self {
        Self {
            sc: m.sc,
            icc: m.icc,
            dcc: m.dcc,
            bi: m.bi,
            rc: m.rc,
        }
    }
}

// ── PyQualityMetrics ──────────────────────────────────────────────────────────

#[pyclass(name = "QualityMetrics", skip_from_py_object)]
#[derive(Clone)]
pub struct PyQualityMetrics {
    #[pyo3(get)]
    pub size_compliance: f64,
    #[pyo3(get)]
    pub intrachunk_cohesion: f64,
    #[pyo3(get)]
    pub contextual_coherence: f64,
    #[pyo3(get)]
    pub block_integrity: f64,
    #[pyo3(get)]
    pub reference_completeness: f64,
    #[pyo3(get)]
    pub composite: f64,
}

impl From<QualityMetrics> for PyQualityMetrics {
    fn from(m: QualityMetrics) -> Self {
        Self {
            size_compliance: m.size_compliance,
            intrachunk_cohesion: m.intrachunk_cohesion,
            contextual_coherence: m.contextual_coherence,
            block_integrity: m.block_integrity,
            reference_completeness: m.reference_completeness,
            composite: m.composite,
        }
    }
}

// ── PyChunkForEval ────────────────────────────────────────────────────────────

#[pyclass(name = "ChunkForEval", from_py_object)]
#[derive(Clone)]
pub struct PyChunkForEval {
    #[pyo3(get, set)]
    pub text: String,
    #[pyo3(get, set)]
    pub offset_start: usize,
    #[pyo3(get, set)]
    pub offset_end: usize,
}

#[pymethods]
impl PyChunkForEval {
    #[new]
    fn new(text: String, offset_start: usize, offset_end: usize) -> Self {
        Self {
            text,
            offset_start,
            offset_end,
        }
    }
}

impl From<&PyChunkForEval> for ChunkForEval {
    fn from(py: &PyChunkForEval) -> Self {
        Self {
            text: py.text.clone(),
            offset_start: py.offset_start,
            offset_end: py.offset_end,
        }
    }
}

// ── PyMetricConfig ────────────────────────────────────────────────────────────

#[pyclass(name = "MetricConfig", from_py_object)]
#[derive(Clone)]
pub struct PyMetricConfig {
    #[pyo3(get, set)]
    pub soft_budget: usize,
    #[pyo3(get, set)]
    pub hard_budget: usize,
    #[pyo3(get, set)]
    pub weights: Option<PyMetricWeights>,
}

#[pymethods]
impl PyMetricConfig {
    #[new]
    #[pyo3(signature = (*, soft_budget=512, hard_budget=768, weights=None))]
    fn new(soft_budget: usize, hard_budget: usize, weights: Option<PyMetricWeights>) -> Self {
        Self {
            soft_budget,
            hard_budget,
            weights,
        }
    }
}

impl From<&PyMetricConfig> for MetricConfig {
    fn from(py: &PyMetricConfig) -> Self {
        Self {
            soft_budget: py.soft_budget,
            hard_budget: py.hard_budget,
            weights: py
                .weights
                .as_ref()
                .map(MetricWeights::from)
                .unwrap_or_default(),
        }
    }
}

// ── py_evaluate_chunks ────────────────────────────────────────────────────────

/// Evaluate quality metrics for a set of chunks against the original text.
#[pyfunction]
#[pyo3(signature = (original_text, chunks, provider, config=None))]
pub fn py_evaluate_chunks(
    py: Python<'_>,
    original_text: &str,
    chunks: Vec<PyChunkForEval>,
    provider: &Bound<'_, PyAny>,
    config: Option<&PyMetricConfig>,
) -> PyResult<PyQualityMetrics> {
    let rust_config = config.map(MetricConfig::from).unwrap_or_default();
    let rust_chunks: Vec<ChunkForEval> = chunks.iter().map(ChunkForEval::from).collect();
    let text_owned = original_text.to_string();

    let _ = py;
    if let Ok(p) = provider.cast::<PyOllamaProvider>() {
        let provider_ref = p.borrow();
        return run_evaluate(&text_owned, &rust_chunks, &provider_ref.inner, &rust_config);
    }
    if let Ok(p) = provider.cast::<PyOpenAiProvider>() {
        let provider_ref = p.borrow();
        return run_evaluate(&text_owned, &rust_chunks, &provider_ref.inner, &rust_config);
    }
    if let Ok(p) = provider.cast::<PyOnnxProvider>() {
        let provider_ref = p.borrow();
        return run_evaluate(&text_owned, &rust_chunks, &provider_ref.inner, &rust_config);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "provider must be OllamaProvider, OpenAiProvider, or OnnxProvider",
    ))
}

fn run_evaluate<P: EmbeddingProvider>(
    original_text: &str,
    chunks: &[ChunkForEval],
    provider: &P,
    config: &MetricConfig,
) -> PyResult<PyQualityMetrics> {
    let metrics = RUNTIME
        .block_on(async { evaluate_chunks(original_text, chunks, provider, config).await })
        .map_err(to_py_err)?;

    Ok(PyQualityMetrics::from(metrics))
}
