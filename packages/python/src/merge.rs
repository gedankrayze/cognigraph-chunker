use numpy::PyArray1;
use pyo3::prelude::*;

use cognigraph_chunker::core::{
    find_merge_indices as rust_find_merge_indices, merge_splits as rust_merge_splits,
};

#[pyclass(name = "MergeResult")]
#[derive(Clone)]
pub struct PyMergeResult {
    #[pyo3(get)]
    pub merged: Vec<String>,
    #[pyo3(get)]
    pub token_counts: Vec<usize>,
}

#[pymethods]
impl PyMergeResult {
    /// Return `token_counts` as a numpy array.
    #[getter]
    fn token_counts_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_slice(py, &self.token_counts)
    }
}

#[pyfunction]
pub fn merge_splits(
    splits: Vec<String>,
    token_counts: Vec<usize>,
    chunk_size: usize,
) -> PyMergeResult {
    let split_refs: Vec<&str> = splits.iter().map(|s| s.as_str()).collect();
    let result = rust_merge_splits(&split_refs, &token_counts, chunk_size);
    PyMergeResult {
        merged: result.merged,
        token_counts: result.token_counts,
    }
}

#[pyfunction]
pub fn find_merge_indices(token_counts: Vec<usize>, chunk_size: usize) -> Vec<usize> {
    rust_find_merge_indices(&token_counts, chunk_size)
}
