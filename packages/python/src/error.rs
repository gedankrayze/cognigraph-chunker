use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pyo3::create_exception!(cognigraph_chunker, CognigraphError, PyRuntimeError);

pub fn to_py_err(e: anyhow::Error) -> PyErr {
    CognigraphError::new_err(format!("{e:#}"))
}
