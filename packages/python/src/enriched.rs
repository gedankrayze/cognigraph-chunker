//! Python bindings for enriched chunking.

use std::collections::HashMap;

use pyo3::prelude::*;

use cognigraph_chunker::llm::{CompletionClient, LlmConfig};
use cognigraph_chunker::semantic::enriched_chunk::{
    EnrichedConfig as RustEnrichedConfig, enriched_chunk, enriched_chunk_plain,
};
use cognigraph_chunker::semantic::enriched_types::{
    EnrichedChunk, EnrichedResult, MergeRecord, TypedEntity,
};

use crate::error::to_py_err;
use crate::semantic::RUNTIME;

/// Configuration for the enriched chunking pipeline.
#[pyclass(name = "EnrichedConfig", from_py_object)]
#[derive(Clone)]
pub struct PyEnrichedConfig {
    #[pyo3(get, set)]
    pub soft_budget: usize,
    #[pyo3(get, set)]
    pub hard_budget: usize,
    #[pyo3(get, set)]
    pub recombine: bool,
    #[pyo3(get, set)]
    pub re_enrich: bool,
}

#[pymethods]
impl PyEnrichedConfig {
    #[new]
    #[pyo3(signature = (*, soft_budget=512, hard_budget=768, recombine=true, re_enrich=true))]
    fn new(soft_budget: usize, hard_budget: usize, recombine: bool, re_enrich: bool) -> Self {
        Self {
            soft_budget,
            hard_budget,
            recombine,
            re_enrich,
        }
    }
}

impl From<&PyEnrichedConfig> for RustEnrichedConfig {
    fn from(py: &PyEnrichedConfig) -> Self {
        Self {
            soft_budget: py.soft_budget,
            hard_budget: py.hard_budget,
            recombine: py.recombine,
            re_enrich: py.re_enrich,
        }
    }
}

/// A typed entity (name + type label).
#[pyclass(name = "TypedEntity", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTypedEntity {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub entity_type: String,
}

/// A record of a merge operation during semantic-key recombination.
#[pyclass(name = "MergeRecord", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMergeRecord {
    #[pyo3(get)]
    pub result_chunk: usize,
    #[pyo3(get)]
    pub source_chunks: Vec<usize>,
    #[pyo3(get)]
    pub shared_key: String,
}

/// A single enriched chunk with full LLM-generated metadata.
#[pyclass(name = "EnrichedChunk", skip_from_py_object)]
#[derive(Clone)]
pub struct PyEnrichedChunk {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub offset_start: usize,
    #[pyo3(get)]
    pub offset_end: usize,
    #[pyo3(get)]
    pub token_estimate: usize,
    #[pyo3(get)]
    pub title: String,
    #[pyo3(get)]
    pub summary: String,
    #[pyo3(get)]
    pub keywords: Vec<String>,
    #[pyo3(get)]
    pub typed_entities: Vec<PyTypedEntity>,
    #[pyo3(get)]
    pub hypothetical_questions: Vec<String>,
    #[pyo3(get)]
    pub semantic_keys: Vec<String>,
    #[pyo3(get)]
    pub category: String,
    #[pyo3(get)]
    pub heading_path: Vec<String>,
}

/// Result of the enriched chunking pipeline.
#[pyclass(name = "EnrichedResult")]
pub struct PyEnrichedResult {
    #[pyo3(get)]
    pub chunks: Vec<PyEnrichedChunk>,
    #[pyo3(get)]
    pub key_dictionary: HashMap<String, Vec<usize>>,
    #[pyo3(get)]
    pub merge_history: Vec<PyMergeRecord>,
    #[pyo3(get)]
    pub block_count: usize,
}

fn convert_result(result: EnrichedResult) -> PyEnrichedResult {
    let chunks = result
        .chunks
        .into_iter()
        .map(|c: EnrichedChunk| PyEnrichedChunk {
            text: c.text,
            offset_start: c.offset_start,
            offset_end: c.offset_end,
            token_estimate: c.token_estimate,
            title: c.title,
            summary: c.summary,
            keywords: c.keywords,
            typed_entities: c
                .typed_entities
                .into_iter()
                .map(|e: TypedEntity| PyTypedEntity {
                    name: e.name,
                    entity_type: e.entity_type,
                })
                .collect(),
            hypothetical_questions: c.hypothetical_questions,
            semantic_keys: c.semantic_keys,
            category: c.category,
            heading_path: c.heading_path,
        })
        .collect();

    let merge_history = result
        .merge_history
        .into_iter()
        .map(|m: MergeRecord| PyMergeRecord {
            result_chunk: m.result_chunk,
            source_chunks: m.source_chunks,
            shared_key: m.shared_key,
        })
        .collect();

    PyEnrichedResult {
        chunks,
        key_dictionary: result.key_dictionary,
        merge_history,
        block_count: result.block_count,
    }
}

/// Run enriched chunking using an LLM (no embedding provider required).
///
/// Args:
///     text: Input text to chunk.
///     api_key: OpenAI-compatible API key.
///     base_url: Optional API base URL (defaults to https://api.openai.com/v1).
///     model: Optional model name (defaults to gpt-4.1-mini).
///     config: Optional EnrichedConfig instance.
///     markdown: If True (default), parse input as markdown; otherwise plain text.
#[pyfunction]
#[pyo3(signature = (text, api_key, base_url=None, model=None, config=None, *, markdown=true))]
pub fn py_enriched_chunk(
    _py: Python<'_>,
    text: &str,
    api_key: String,
    base_url: Option<String>,
    model: Option<String>,
    config: Option<&PyEnrichedConfig>,
    markdown: bool,
) -> PyResult<PyEnrichedResult> {
    let rust_config = config.map(RustEnrichedConfig::from).unwrap_or_default();
    let text_owned = text.to_string();

    let llm_config = LlmConfig::resolve(&Some(api_key), &base_url, &model).map_err(to_py_err)?;
    let llm_client = CompletionClient::new(llm_config).map_err(to_py_err)?;

    let result = RUNTIME
        .block_on(async {
            if markdown {
                enriched_chunk(&text_owned, &llm_client, &rust_config).await
            } else {
                enriched_chunk_plain(&text_owned, &llm_client, &rust_config).await
            }
        })
        .map_err(to_py_err)?;

    Ok(convert_result(result))
}
