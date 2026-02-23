use pyo3::prelude::*;

use cognigraph_chunker::core::{
    filter_split_indices as rust_filter,
    find_local_minima_interpolated as rust_minima,
    savgol_filter as rust_savgol,
    windowed_cross_similarity as rust_cross_sim,
};

#[pyclass(name = "MinimaResult")]
#[derive(Clone)]
pub struct PyMinimaResult {
    #[pyo3(get)]
    pub indices: Vec<usize>,
    #[pyo3(get)]
    pub values: Vec<f64>,
}

#[pyclass(name = "FilteredIndices")]
#[derive(Clone)]
pub struct PyFilteredIndices {
    #[pyo3(get)]
    pub indices: Vec<usize>,
    #[pyo3(get)]
    pub values: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (data, window_length, poly_order, *, deriv=0))]
pub fn savgol_filter(
    data: Vec<f64>,
    window_length: usize,
    poly_order: usize,
    deriv: usize,
) -> PyResult<Vec<f64>> {
    rust_savgol(&data, window_length, poly_order, deriv).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "Invalid parameters: window_length must be odd and > poly_order, data must be non-empty",
        )
    })
}

#[pyfunction]
pub fn windowed_cross_similarity(
    embeddings: Vec<f64>,
    n: usize,
    d: usize,
    window_size: usize,
) -> PyResult<Vec<f64>> {
    rust_cross_sim(&embeddings, n, d, window_size).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "Invalid parameters: window_size must be odd >= 3, n >= 2, d > 0",
        )
    })
}

#[pyfunction]
#[pyo3(signature = (data, window_size, poly_order, *, tolerance=0.1))]
pub fn find_local_minima(
    data: Vec<f64>,
    window_size: usize,
    poly_order: usize,
    tolerance: f64,
) -> PyResult<PyMinimaResult> {
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
