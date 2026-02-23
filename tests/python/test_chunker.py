import cognigraph_chunker as cc


def test_basic_chunking():
    chunker = cc.Chunker("Hello. World. Test.", size=10, delimiters=b".")
    chunks = chunker.collect_chunks()
    assert len(chunks) == 3
    assert chunks[0] == "Hello."
    assert chunks[1] == " World."
    assert chunks[2] == " Test."


def test_iterator():
    chunks = list(cc.Chunker("Hello. World.", size=10, delimiters=b"."))
    assert len(chunks) == 2
    assert chunks[0] == "Hello."


def test_collect_offsets():
    chunker = cc.Chunker("Hello. World. Test.", size=10, delimiters=b".")
    offsets = chunker.collect_offsets()
    assert len(offsets) == 3
    assert offsets[0] == (0, 6)


def test_small_text():
    chunks = cc.Chunker("Small", size=100).collect_chunks()
    assert chunks == ["Small"]


def test_empty_text():
    chunks = cc.Chunker("", size=10).collect_chunks()
    assert chunks == []


def test_prefix_mode():
    chunks = cc.Chunker("Hello World Test", size=8, delimiters=b" ", prefix=True).collect_chunks()
    assert chunks[0] == "Hello"
    assert chunks[1] == " World"
    assert chunks[2] == " Test"


def test_reset():
    chunker = cc.Chunker("Hello. World.", size=10, delimiters=b".")
    first = chunker.collect_chunks()
    chunker.reset()
    second = chunker.collect_chunks()
    assert first == second
