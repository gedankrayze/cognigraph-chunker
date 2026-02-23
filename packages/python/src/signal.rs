use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use cognigraph_chunker::core::{
    filter_split_indices as rust_filter, find_local_minima_interpolated as rust_minima,
    savgol_filter as rust_savgol, windowed_cross_similarity as rust_cross_sim,
};

/// Extract a `Vec<f64>` from either a numpy array or a Python list.
fn extract_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    // Try numpy array first, then fall back to list.
    if let Ok(arr) = obj.downcast::<PyArray1<f64>>() {
        let readonly = arr.try_readonly().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("cannot borrow array: {e}"))
        })?;
        let slice = readonly.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("array is not contiguous: {e}"))
        })?;
        Ok(slice.to_vec())
    } else {
        obj.extract::<Vec<f64>>()
    }
}

#[pyclass(name = "MinimaResult")]
#[derive(Clone)]
pub struct PyMinimaResult {
    #[pyo3(get)]
    pub indices: Vec<usize>,
    #[pyo3(get)]
    pub values: Vec<f64>,
}

#[pymethods]
impl PyMinimaResult {
    /// Return `values` as a numpy array.
    #[getter]
    fn values_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.values)
    }
}

#[pyclass(name = "FilteredIndices")]
#[derive(Clone)]
pub struct PyFilteredIndices {
    #[pyo3(get)]
    pub indices: Vec<usize>,
    #[pyo3(get)]
    pub values: Vec<f64>,
}

#[pymethods]
impl PyFilteredIndices {
    /// Return `values` as a numpy array.
    #[getter]
    fn values_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.values)
    }
}

#[pyfunction]
#[pyo3(signature = (data, window_length, poly_order, *, deriv=0))]
pub fn savgol_filter<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyAny>,
    window_length: usize,
    poly_order: usize,
    deriv: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = extract_f64_vec(data)?;
    let result = rust_savgol(&data, window_length, poly_order, deriv).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "Invalid parameters: window_length must be odd and > poly_order, data must be non-empty",
        )
    })?;
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
pub fn windowed_cross_similarity<'py>(
    py: Python<'py>,
    embeddings: &Bound<'_, PyAny>,
    n: usize,
    d: usize,
    window_size: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let embeddings = extract_f64_vec(embeddings)?;
    let result = rust_cross_sim(&embeddings, n, d, window_size).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "Invalid parameters: window_size must be odd >= 3, n >= 2, d > 0",
        )
    })?;
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (data, window_size, poly_order, *, tolerance=0.1))]
pub fn find_local_minima(
    data: &Bound<'_, PyAny>,
    window_size: usize,
    poly_order: usize,
    tolerance: f64,
) -> PyResult<PyMinimaResult> {
    let data = extract_f64_vec(data)?;
    let result = rust_minima(&data, window_size, poly_order, tolerance).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Invalid parameters for minima detection")
    })?;
    Ok(PyMinimaResult {
        indices: result.indices,
        values: result.values,
    })
}

#[pyfunction]
#[pyo3(signature = (indices, values, threshold, *, min_distance=1))]
pub fn filter_split_indices(
    indices: Vec<usize>,
    values: Vec<f64>,
    threshold: f64,
    min_distance: usize,
) -> PyFilteredIndices {
    let result = rust_filter(&indices, &values, threshold, min_distance);
    PyFilteredIndices {
        indices: result.indices,
        values: result.values,
    }
}
