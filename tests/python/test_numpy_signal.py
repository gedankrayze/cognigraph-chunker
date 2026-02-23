import numpy as np
import cognigraph_chunker as cc
import pytest


# --- savgol_filter ---

def test_savgol_filter_returns_numpy_array():
    data = [float(i + 1) for i in range(9)]
    result = cc.savgol_filter(data, 5, 2)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert len(result) == 9


def test_savgol_filter_accepts_numpy_input():
    data = np.array([float(i + 1) for i in range(9)], dtype=np.float64)
    result = cc.savgol_filter(data, 5, 2)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert len(result) == 9


def test_savgol_filter_accepts_list_input():
    data = [float(i + 1) for i in range(9)]
    result = cc.savgol_filter(data, 5, 2)
    assert isinstance(result, np.ndarray)
    assert len(result) == 9


# --- windowed_cross_similarity ---

def test_windowed_cross_similarity_returns_numpy_array():
    embeddings = [
        1.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    ]
    result = cc.windowed_cross_similarity(embeddings, 3, 3, 3)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert len(result) == 2


def test_windowed_cross_similarity_accepts_numpy_input():
    embeddings = np.array([
        1.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    ], dtype=np.float64)
    result = cc.windowed_cross_similarity(embeddings, 3, 3, 3)
    assert isinstance(result, np.ndarray)
    assert len(result) == 2


# --- MinimaResult.values_array ---

def test_minima_values_array_returns_numpy():
    data = [((i - 10) / 3.0) ** 2 for i in range(20)]
    result = cc.find_local_minima(data, 5, 2)
    arr = result.values_array
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert len(arr) == len(result.values)
    np.testing.assert_array_equal(arr, result.values)


def test_find_local_minima_accepts_numpy_input():
    data = np.array([((i - 10) / 3.0) ** 2 for i in range(20)], dtype=np.float64)
    result = cc.find_local_minima(data, 5, 2)
    assert len(result.indices) > 0


# --- FilteredIndices.values_array ---

def test_filtered_indices_values_array_returns_numpy():
    result = cc.filter_split_indices(
        [0, 5, 8, 15, 20],
        [0.1, 0.3, 0.2, 0.5, 0.4],
        0.5,
        min_distance=3,
    )
    arr = result.values_array
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert len(arr) == len(result.values)
    np.testing.assert_array_equal(arr, result.values)


# --- MergeResult.token_counts_array ---

def test_merge_result_token_counts_array():
    result = cc.merge_splits(["a", "b", "c"], [10, 20, 30], 100)
    arr = result.token_counts_array
    assert isinstance(arr, np.ndarray)
    assert len(arr) == len(result.token_counts)
