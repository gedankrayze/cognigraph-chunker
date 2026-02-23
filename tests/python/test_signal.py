import cognigraph_chunker as cc
import pytest


def test_savgol_filter_smoothing():
    data = [float(i + 1) for i in range(9)]
    result = cc.savgol_filter(data, 5, 2)
    assert len(result) == 9
    for i, val in enumerate(result):
        assert abs(val - (i + 1)) < 0.5


def test_savgol_filter_invalid():
    with pytest.raises(ValueError):
        cc.savgol_filter([1.0, 2.0, 3.0], 4, 2)  # even window


def test_windowed_cross_similarity():
    embeddings = [
        1.0, 0.0, 0.0,  # emb 1
        1.0, 0.0, 0.0,  # emb 2 (same)
        0.0, 1.0, 0.0,  # emb 3 (orthogonal)
    ]
    result = cc.windowed_cross_similarity(embeddings, 3, 3, 3)
    assert len(result) == 2
    assert abs(result[0] - 0.5) < 0.1


def test_find_local_minima():
    data = [((i - 10) / 3.0) ** 2 for i in range(20)]
    result = cc.find_local_minima(data, 5, 2)
    assert len(result.indices) > 0
    assert abs(result.indices[0] - 10) <= 2


def test_filter_split_indices():
    result = cc.filter_split_indices([0, 5, 8, 15, 20], [0.1, 0.3, 0.2, 0.5, 0.4], 0.5, min_distance=3)
    assert len(result.indices) > 0
