import cognigraph_chunker as cc


def test_merge_splits_basic():
    result = cc.merge_splits(["a", "b", "c", "d", "e", "f", "g"], [1, 1, 1, 1, 1, 1, 1], 3)
    assert result.merged == ["abc", "def", "g"]
    assert result.token_counts == [3, 3, 1]


def test_merge_splits_empty():
    result = cc.merge_splits([], [], 10)
    assert result.merged == []
    assert result.token_counts == []


def test_merge_splits_all_exceed():
    result = cc.merge_splits(["aaa", "bbb", "ccc"], [50, 60, 70], 30)
    assert result.merged == ["aaa", "bbb", "ccc"]
    assert result.token_counts == [50, 60, 70]


def test_find_merge_indices():
    indices = cc.find_merge_indices([1, 1, 1, 1, 1, 1, 1], 3)
    assert indices == [3, 6, 7]


def test_find_merge_indices_empty():
    indices = cc.find_merge_indices([], 10)
    assert indices == []
