//! Python bindings for adaptive chunking.

use pyo3::prelude::*;

use cognigraph_chunker::embeddings::EmbeddingProvider;
use cognigraph_chunker::llm::{CompletionClient, LlmConfig};
use cognigraph_chunker::semantic::adaptive_chunk::{
    AdaptiveConfig as RustAdaptiveConfig, adaptive_chunk,
};
use cognigraph_chunker::semantic::adaptive_types::{
    AdaptiveReport, AdaptiveResult, CandidateScore, ScreeningDecision,
};
use cognigraph_chunker::semantic::quality_metrics::MetricWeights;

use crate::error::to_py_err;
use crate::quality_metrics::{PyMetricWeights, PyQualityMetrics};
use crate::semantic::RUNTIME;
use crate::semantic::providers::{PyOllamaProvider, PyOnnxProvider, PyOpenAiProvider};

// ── AdaptiveConfig ────────────────────────────────────────────────────────────

#[pyclass(name = "AdaptiveConfig", skip_from_py_object)]
#[derive(Clone)]
pub struct PyAdaptiveConfig {
    #[pyo3(get, set)]
    pub candidates: Vec<String>,
    #[pyo3(get, set)]
    pub force_candidates: bool,
    #[pyo3(get, set)]
    pub soft_budget: usize,
    #[pyo3(get, set)]
    pub hard_budget: usize,
    #[pyo3(get, set)]
    pub metric_weights: Option<PyMetricWeights>,
    #[pyo3(get, set)]
    pub sim_window: usize,
    #[pyo3(get, set)]
    pub sg_window: usize,
    #[pyo3(get, set)]
    pub poly_order: usize,
}

#[pymethods]
impl PyAdaptiveConfig {
    #[new]
    #[pyo3(signature = (
        *,
        candidates=None,
        force_candidates=false,
        soft_budget=512,
        hard_budget=768,
        metric_weights=None,
        sim_window=3,
        sg_window=11,
        poly_order=3,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        candidates: Option<Vec<String>>,
        force_candidates: bool,
        soft_budget: usize,
        hard_budget: usize,
        metric_weights: Option<PyMetricWeights>,
        sim_window: usize,
        sg_window: usize,
        poly_order: usize,
    ) -> Self {
        Self {
            candidates: candidates
                .unwrap_or_else(|| vec!["semantic".to_string(), "cognitive".to_string()]),
            force_candidates,
            soft_budget,
            hard_budget,
            metric_weights,
            sim_window,
            sg_window,
            poly_order,
        }
    }
}

impl From<&PyAdaptiveConfig> for RustAdaptiveConfig {
    fn from(py: &PyAdaptiveConfig) -> Self {
        let metric_weights = py
            .metric_weights
            .as_ref()
            .map(MetricWeights::from)
            .unwrap_or_default();
        Self {
            candidates: py.candidates.clone(),
            force_candidates: py.force_candidates,
            soft_budget: py.soft_budget,
            hard_budget: py.hard_budget,
            metric_weights,
            sim_window: py.sim_window,
            sg_window: py.sg_window,
            poly_order: py.poly_order,
        }
    }
}

// ── CandidateScore ────────────────────────────────────────────────────────────

#[pyclass(name = "CandidateScore", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCandidateScore {
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub metrics: PyQualityMetrics,
    #[pyo3(get)]
    pub chunk_count: usize,
    #[pyo3(get)]
    pub total_tokens: usize,
}

impl From<CandidateScore> for PyCandidateScore {
    fn from(r: CandidateScore) -> Self {
        Self {
            method: r.method,
            metrics: PyQualityMetrics::from(r.metrics),
            chunk_count: r.chunk_count,
            total_tokens: r.total_tokens,
        }
    }
}

// ── ScreeningDecision ─────────────────────────────────────────────────────────

#[pyclass(name = "ScreeningDecision", skip_from_py_object)]
#[derive(Clone)]
pub struct PyScreeningDecision {
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub included: bool,
    #[pyo3(get)]
    pub reason: String,
}

impl From<ScreeningDecision> for PyScreeningDecision {
    fn from(r: ScreeningDecision) -> Self {
        Self {
            method: r.method,
            included: r.included,
            reason: r.reason,
        }
    }
}

// ── AdaptiveReport ────────────────────────────────────────────────────────────

#[pyclass(name = "AdaptiveReport", skip_from_py_object)]
#[derive(Clone)]
pub struct PyAdaptiveReport {
    #[pyo3(get)]
    pub candidates: Vec<PyCandidateScore>,
    #[pyo3(get)]
    pub pre_screening: Vec<PyScreeningDecision>,
    #[pyo3(get)]
    pub metric_weights: PyMetricWeights,
}

impl From<AdaptiveReport> for PyAdaptiveReport {
    fn from(r: AdaptiveReport) -> Self {
        Self {
            candidates: r
                .candidates
                .into_iter()
                .map(PyCandidateScore::from)
                .collect(),
            pre_screening: r
                .pre_screening
                .into_iter()
                .map(PyScreeningDecision::from)
                .collect(),
            metric_weights: PyMetricWeights::from(r.metric_weights),
        }
    }
}

// ── AdaptiveResult ────────────────────────────────────────────────────────────

#[pyclass(name = "AdaptiveResult")]
pub struct PyAdaptiveResult {
    #[pyo3(get)]
    pub winner: String,
    /// The winner's chunks serialized as a JSON string.
    #[pyo3(get)]
    pub chunks_json: String,
    #[pyo3(get)]
    pub report: PyAdaptiveReport,
    #[pyo3(get)]
    pub count: usize,
}

impl From<AdaptiveResult> for PyAdaptiveResult {
    fn from(r: AdaptiveResult) -> Self {
        let chunks_json = serde_json::to_string(&r.chunks).unwrap_or_else(|_| "[]".to_string());
        Self {
            winner: r.winner,
            chunks_json,
            report: PyAdaptiveReport::from(r.report),
            count: r.count,
        }
    }
}

// ── py_adaptive_chunk ─────────────────────────────────────────────────────────

/// Run adaptive chunking, selecting the best method from the configured candidates.
#[pyfunction]
#[pyo3(signature = (text, provider, api_key=None, base_url=None, model=None, config=None))]
pub fn py_adaptive_chunk(
    py: Python<'_>,
    text: &str,
    provider: &Bound<'_, PyAny>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    config: Option<&PyAdaptiveConfig>,
) -> PyResult<PyAdaptiveResult> {
    let rust_config = config.map(RustAdaptiveConfig::from).unwrap_or_default();
    let text_owned = text.to_string();

    // Build optional LLM client
    let llm_client: Option<CompletionClient> = if api_key.is_some() {
        let llm_config = LlmConfig::resolve(&api_key, &base_url, &model).map_err(to_py_err)?;
        Some(CompletionClient::new(llm_config).map_err(to_py_err)?)
    } else {
        None
    };

    let _ = py;

    if let Ok(p) = provider.cast::<PyOllamaProvider>() {
        let provider_ref = p.borrow();
        return run_adaptive(
            &text_owned,
            &provider_ref.inner,
            llm_client.as_ref(),
            &rust_config,
        );
    }
    if let Ok(p) = provider.cast::<PyOpenAiProvider>() {
        let provider_ref = p.borrow();
        return run_adaptive(
            &text_owned,
            &provider_ref.inner,
            llm_client.as_ref(),
            &rust_config,
        );
    }
    if let Ok(p) = provider.cast::<PyOnnxProvider>() {
        let provider_ref = p.borrow();
        return run_adaptive(
            &text_owned,
            &provider_ref.inner,
            llm_client.as_ref(),
            &rust_config,
        );
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "provider must be OllamaProvider, OpenAiProvider, or OnnxProvider",
    ))
}

fn run_adaptive<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    llm_client: Option<&CompletionClient>,
    config: &RustAdaptiveConfig,
) -> PyResult<PyAdaptiveResult> {
    let result = RUNTIME
        .block_on(async { adaptive_chunk(text, provider, llm_client, config).await })
        .map_err(to_py_err)?;

    Ok(PyAdaptiveResult::from(result))
}
