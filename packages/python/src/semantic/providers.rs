use pyo3::prelude::*;

use cognigraph_chunker::embeddings::ollama::OllamaProvider;
use cognigraph_chunker::embeddings::onnx::OnnxProvider as RustOnnxProvider;
use cognigraph_chunker::embeddings::openai::OpenAiProvider;

use crate::error::to_py_err;

#[pyclass(name = "OllamaProvider")]
pub struct PyOllamaProvider {
    pub(crate) inner: OllamaProvider,
}

#[pymethods]
impl PyOllamaProvider {
    #[new]
    #[pyo3(signature = (*, model=None, base_url=None))]
    fn new(model: Option<String>, base_url: Option<String>) -> PyResult<Self> {
        let inner = OllamaProvider::new(base_url, model).map_err(to_py_err)?;
        Ok(Self { inner })
    }
}

#[pyclass(name = "OpenAiProvider")]
pub struct PyOpenAiProvider {
    pub(crate) inner: OpenAiProvider,
}

#[pymethods]
impl PyOpenAiProvider {
    #[new]
    #[pyo3(signature = (api_key, *, model=None, base_url=None))]
    fn new(api_key: String, model: Option<String>, base_url: Option<String>) -> PyResult<Self> {
        let inner = OpenAiProvider::new(api_key, base_url, model).map_err(to_py_err)?;
        Ok(Self { inner })
    }
}

#[pyclass(name = "OnnxProvider")]
pub struct PyOnnxProvider {
    pub(crate) inner: RustOnnxProvider,
}

#[pymethods]
impl PyOnnxProvider {
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let inner = RustOnnxProvider::new(model_path).map_err(to_py_err)?;
        Ok(Self { inner })
    }
}
