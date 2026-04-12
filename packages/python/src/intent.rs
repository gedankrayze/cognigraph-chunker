//! Python bindings for intent-driven chunking.

use pyo3::prelude::*;

use cognigraph_chunker::embeddings::EmbeddingProvider;
use cognigraph_chunker::llm::{CompletionClient, LlmConfig};
use cognigraph_chunker::semantic::intent_chunk::{
    IntentConfig as RustIntentConfig, intent_chunk, intent_chunk_plain,
};
use cognigraph_chunker::semantic::intent_types::IntentType;

use crate::error::to_py_err;
use crate::semantic::RUNTIME;
use crate::semantic::providers::{PyOllamaProvider, PyOnnxProvider, PyOpenAiProvider};

/// Configuration for intent-driven chunking.
#[pyclass(name = "IntentConfig", from_py_object)]
#[derive(Clone)]
pub struct PyIntentConfig {
    #[pyo3(get, set)]
    pub max_intents: usize,
    #[pyo3(get, set)]
    pub soft_budget: usize,
    #[pyo3(get, set)]
    pub hard_budget: usize,
}

#[pymethods]
impl PyIntentConfig {
    #[new]
    #[pyo3(signature = (*, max_intents=20, soft_budget=512, hard_budget=768))]
    fn new(max_intents: usize, soft_budget: usize, hard_budget: usize) -> Self {
        Self {
            max_intents,
            soft_budget,
            hard_budget,
        }
    }
}

impl From<&PyIntentConfig> for RustIntentConfig {
    fn from(py: &PyIntentConfig) -> Self {
        Self {
            max_intents: py.max_intents,
            soft_budget: py.soft_budget,
            hard_budget: py.hard_budget,
        }
    }
}

/// A predicted user intent with matched chunk indices.
#[pyclass(name = "PredictedIntent", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPredictedIntent {
    #[pyo3(get)]
    pub query: String,
    #[pyo3(get)]
    pub intent_type: String,
    #[pyo3(get)]
    pub matched_chunks: Vec<usize>,
}

/// A chunk produced by intent-driven chunking.
#[pyclass(name = "IntentChunk", skip_from_py_object)]
#[derive(Clone)]
pub struct PyIntentChunk {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub offset_start: usize,
    #[pyo3(get)]
    pub offset_end: usize,
    #[pyo3(get)]
    pub token_estimate: usize,
    #[pyo3(get)]
    pub best_intent: usize,
    #[pyo3(get)]
    pub alignment_score: f64,
    #[pyo3(get)]
    pub heading_path: Vec<String>,
}

/// Result of intent-driven chunking.
#[pyclass(name = "IntentResult")]
pub struct PyIntentResult {
    #[pyo3(get)]
    pub chunks: Vec<PyIntentChunk>,
    #[pyo3(get)]
    pub intents: Vec<PyPredictedIntent>,
    #[pyo3(get)]
    pub partition_score: f64,
    #[pyo3(get)]
    pub block_count: usize,
}

/// Run intent-driven chunking with a provider.
#[pyfunction]
#[pyo3(signature = (text, provider, api_key, base_url=None, model=None, config=None, *, markdown=true))]
#[allow(clippy::too_many_arguments)]
pub fn py_intent_chunk(
    py: Python<'_>,
    text: &str,
    provider: &Bound<'_, PyAny>,
    api_key: String,
    base_url: Option<String>,
    model: Option<String>,
    config: Option<&PyIntentConfig>,
    markdown: bool,
) -> PyResult<PyIntentResult> {
    let rust_config = config.map(RustIntentConfig::from).unwrap_or_default();
    let text_owned = text.to_string();

    let llm_config = LlmConfig::resolve(&Some(api_key), &base_url, &model).map_err(to_py_err)?;
    let llm_client = CompletionClient::new(llm_config).map_err(to_py_err)?;

    let _ = py;
    if let Ok(p) = provider.cast::<PyOllamaProvider>() {
        let provider_ref = p.borrow();
        return run_intent(
            &text_owned,
            &provider_ref.inner,
            &llm_client,
            &rust_config,
            markdown,
        );
    }
    if let Ok(p) = provider.cast::<PyOpenAiProvider>() {
        let provider_ref = p.borrow();
        return run_intent(
            &text_owned,
            &provider_ref.inner,
            &llm_client,
            &rust_config,
            markdown,
        );
    }
    if let Ok(p) = provider.cast::<PyOnnxProvider>() {
        let provider_ref = p.borrow();
        return run_intent(
            &text_owned,
            &provider_ref.inner,
            &llm_client,
            &rust_config,
            markdown,
        );
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "provider must be OllamaProvider, OpenAiProvider, or OnnxProvider",
    ))
}

fn run_intent<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    llm_client: &CompletionClient,
    config: &RustIntentConfig,
    markdown: bool,
) -> PyResult<PyIntentResult> {
    let result = RUNTIME
        .block_on(async {
            if markdown {
                intent_chunk(text, provider, llm_client, config).await
            } else {
                intent_chunk_plain(text, provider, llm_client, config).await
            }
        })
        .map_err(to_py_err)?;

    let chunks = result
        .chunks
        .into_iter()
        .map(|c| PyIntentChunk {
            text: c.text,
            offset_start: c.offset_start,
            offset_end: c.offset_end,
            token_estimate: c.token_estimate,
            best_intent: c.best_intent,
            alignment_score: c.alignment_score,
            heading_path: c.heading_path,
        })
        .collect();

    let intents = result
        .intents
        .into_iter()
        .map(|i| PyPredictedIntent {
            query: i.query,
            intent_type: intent_type_to_str(i.intent_type).to_string(),
            matched_chunks: i.matched_chunks,
        })
        .collect();

    Ok(PyIntentResult {
        chunks,
        intents,
        partition_score: result.partition_score,
        block_count: result.block_count,
    })
}

fn intent_type_to_str(t: IntentType) -> &'static str {
    match t {
        IntentType::Factual => "factual",
        IntentType::Procedural => "procedural",
        IntentType::Conceptual => "conceptual",
        IntentType::Comparative => "comparative",
    }
}
