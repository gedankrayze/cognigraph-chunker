import pytest
import cognigraph_chunker as cc


def ollama_available():
    """Check if Ollama is running locally."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
def test_semantic_chunk_ollama():
    text = (
        "Machine learning is a subset of artificial intelligence. "
        "It focuses on algorithms that learn from data. "
        "Neural networks are a popular approach. "
        "The weather today is sunny and warm. "
        "Climate change affects global temperatures. "
        "Rising sea levels threaten coastal cities."
    )
    provider = cc.OllamaProvider(model="nomic-embed-text")
    config = cc.SemanticConfig(threshold=0.3)
    result = cc.py_semantic_chunk(text, provider, config, markdown=False)
    assert len(result.chunks) >= 1
    assert all(isinstance(c, tuple) and len(c) == 2 for c in result.chunks)


@pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
def test_semantic_config_defaults():
    config = cc.SemanticConfig()
    assert config.sim_window == 3
    assert config.sg_window == 11
    assert config.poly_order == 3
    assert config.threshold == 0.5
    assert config.min_distance == 2


def test_provider_type_error():
    with pytest.raises(TypeError):
        cc.py_semantic_chunk("hello", "not_a_provider")
