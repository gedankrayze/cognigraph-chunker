//! Python bindings for topology-aware chunking.

use pyo3::prelude::*;

use cognigraph_chunker::llm::{CompletionClient, LlmConfig};
use cognigraph_chunker::semantic::topo_chunk::{TopoConfig as RustTopoConfig, topo_chunk};
use cognigraph_chunker::semantic::topo_types::{SectionClass, SectionClassification, TopoChunk};

use crate::error::to_py_err;
use crate::semantic::RUNTIME;

/// Configuration for topology-aware chunking.
#[pyclass(name = "TopoConfig", from_py_object)]
#[derive(Clone)]
pub struct PyTopoConfig {
    #[pyo3(get, set)]
    pub soft_budget: usize,
    #[pyo3(get, set)]
    pub hard_budget: usize,
    #[pyo3(get, set)]
    pub emit_sir: bool,
}

#[pymethods]
impl PyTopoConfig {
    #[new]
    #[pyo3(signature = (*, soft_budget=512, hard_budget=768, emit_sir=false))]
    fn new(soft_budget: usize, hard_budget: usize, emit_sir: bool) -> Self {
        Self {
            soft_budget,
            hard_budget,
            emit_sir,
        }
    }
}

impl From<&PyTopoConfig> for RustTopoConfig {
    fn from(py: &PyTopoConfig) -> Self {
        Self {
            soft_budget: py.soft_budget,
            hard_budget: py.hard_budget,
            emit_sir: py.emit_sir,
        }
    }
}

/// A section classification returned to Python.
#[pyclass(name = "SectionClassification", from_py_object)]
#[derive(Clone)]
pub struct PySectionClassification {
    #[pyo3(get)]
    pub section_id: usize,
    #[pyo3(get)]
    pub class: String,
    #[pyo3(get)]
    pub reason: String,
}

/// A topology-aware chunk returned to Python.
#[pyclass(name = "TopoChunk", from_py_object)]
#[derive(Clone)]
pub struct PyTopoChunk {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub offset_start: usize,
    #[pyo3(get)]
    pub offset_end: usize,
    #[pyo3(get)]
    pub token_estimate: usize,
    #[pyo3(get)]
    pub heading_path: Vec<String>,
    #[pyo3(get)]
    pub section_classification: String,
    #[pyo3(get)]
    pub cross_references: Vec<usize>,
}

/// Result of topology-aware chunking.
#[pyclass(name = "TopoResult")]
pub struct PyTopoResult {
    #[pyo3(get)]
    pub chunks: Vec<PyTopoChunk>,
    #[pyo3(get)]
    pub classifications: Vec<PySectionClassification>,
    #[pyo3(get)]
    pub block_count: usize,
    #[pyo3(get)]
    pub sir_json: Option<String>,
}

fn section_class_to_str(class: SectionClass) -> &'static str {
    match class {
        SectionClass::Atomic => "atomic",
        SectionClass::Splittable => "splittable",
        SectionClass::MergeCandidate => "merge_candidate",
    }
}

fn convert_chunk(c: TopoChunk) -> PyTopoChunk {
    PyTopoChunk {
        text: c.text,
        offset_start: c.offset_start,
        offset_end: c.offset_end,
        token_estimate: c.token_estimate,
        heading_path: c.heading_path,
        section_classification: c.section_classification,
        cross_references: c.cross_references,
    }
}

fn convert_classification(sc: SectionClassification) -> PySectionClassification {
    PySectionClassification {
        section_id: sc.section_id,
        class: section_class_to_str(sc.class).to_string(),
        reason: sc.reason,
    }
}

/// Run topology-aware chunking.
#[pyfunction]
#[pyo3(signature = (text, api_key, base_url=None, model=None, config=None))]
pub fn py_topo_chunk(
    _py: Python<'_>,
    text: &str,
    api_key: String,
    base_url: Option<String>,
    model: Option<String>,
    config: Option<&PyTopoConfig>,
) -> PyResult<PyTopoResult> {
    let rust_config = config.map(RustTopoConfig::from).unwrap_or_default();
    let text_owned = text.to_string();

    let llm_config = LlmConfig::resolve(&Some(api_key), &base_url, &model).map_err(to_py_err)?;
    let llm_client = CompletionClient::new(llm_config).map_err(to_py_err)?;

    let result = RUNTIME
        .block_on(async { topo_chunk(&text_owned, &llm_client, &rust_config).await })
        .map_err(to_py_err)?;

    let sir_json = if rust_config.emit_sir {
        serde_json::to_string(&result.sir).ok()
    } else {
        None
    };

    let chunks = result.chunks.into_iter().map(convert_chunk).collect();
    let classifications = result
        .classifications
        .into_iter()
        .map(convert_classification)
        .collect();

    Ok(PyTopoResult {
        chunks,
        classifications,
        block_count: result.block_count,
        sir_json,
    })
}
