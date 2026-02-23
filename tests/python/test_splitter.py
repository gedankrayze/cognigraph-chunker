import cognigraph_chunker as cc


def test_split_at_delimiters_basic():
    offsets = cc.py_split_at_delimiters("Hello. World. Test.", b".")
    assert len(offsets) == 3
    assert offsets[0] == (0, 6)


def test_split_at_delimiters_include_next():
    offsets = cc.py_split_at_delimiters("Hello. World.", b".", include_delim="next")
    assert offsets[0] == (0, 5)  # "Hello"


def test_split_at_delimiters_include_none():
    offsets = cc.py_split_at_delimiters("Hello.World.", b".", include_delim="none")
    assert len(offsets) >= 2


def test_split_at_patterns():
    offsets = cc.py_split_at_patterns("Hello. World. Test.", [b". "])
    assert len(offsets) == 3


def test_pattern_splitter():
    ps = cc.PatternSplitter([b". ", b"? "])
    offsets = ps.split("Hello. World? Test.")
    assert len(offsets) >= 2


def test_pattern_splitter_min_chars():
    ps = cc.PatternSplitter([b". "])
    offsets = ps.split("Hi. Hello world. Test.", min_chars=10)
    # Short segments get merged
    assert len(offsets) >= 1


def test_empty_text():
    offsets = cc.py_split_at_delimiters("", b".")
    assert offsets == []
