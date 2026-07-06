# cognigraph-chunker

Fast text chunking toolkit for RAG pipelines, written in Rust with Python bindings.

Eight chunking strategies — fixed-size, delimiter, semantic, cognition-aware,
intent-driven, topology-aware, enriched, and adaptive — plus token-aware
merging, signal-processing primitives, and five intrinsic quality metrics.

## Install

```sh
pip install cognigraph-chunker
```

## Quick start

```python
from cognigraph_chunker import Chunker, OllamaProvider, SemanticConfig, semantic_chunk

# Fixed-size chunking — no external services needed
for chunk in Chunker("Your long document text here...", size=1024):
    print(chunk)

# Semantic chunking with a local Ollama embedding model
provider = OllamaProvider(model="nomic-embed-text")
result = semantic_chunk(open("document.md").read(), provider, SemanticConfig())
for text, offset in result.chunks:
    print(f"[{offset}] {text[:80]}")
```

Cognition-aware chunking (entity/discourse-preserving boundaries with quality
metrics), intent-driven, topology-aware, enriched, and adaptive strategies are
exposed as `cognitive_chunk`, `intent_chunk`, `topo_chunk`, `enriched_chunk`,
and `adaptive_chunk`, alongside `evaluate_chunks` for standalone quality
scoring.

Full documentation, the CLI, and the REST API live in the
[project repository](https://github.com/gedankrayze/cognigraph-chunker).

MIT licensed.
