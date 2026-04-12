# Adaptive Chunking: Automatic Method Selection

No single chunking method works best for all documents. A flat blog post with clear topic transitions is well served by semantic chunking. A deeply nested specification needs topology-aware chunking. A knowledge base article intended for RAG retrieval benefits from intent-driven or enriched chunking. Choosing the right method requires understanding both the document and the downstream use case -- knowledge that is often unavailable when building a chunking pipeline for diverse input.

Adaptive chunking is a meta-router. It runs multiple candidate methods on the same document, scores each method's output using five intrinsic quality metrics, and returns the output from the method that scores highest. The result is per-document method selection without manual tuning.

## The five quality metrics

Each metric is scored from 0.0 to 1.0. Together they measure whether the chunks are well-sized, internally coherent, properly bounded, and free of broken references.

**Size Compliance (SC)** measures whether chunks fall within the target size range. It counts the fraction of chunks whose token count is between half the soft budget and the hard budget. A score of 1.0 means every chunk is appropriately sized. Low scores indicate fragmentation (too many small chunks) or monolithic chunks (too few, too large).

**Intrachunk Cohesion (ICC)** measures whether each chunk is about one thing. For each chunk, the pipeline splits it into sentences, embeds each sentence and the full chunk text, and computes the mean cosine similarity of sentence embeddings to the chunk embedding. A cohesive chunk has high internal similarity; a chunk that covers multiple unrelated topics has low similarity. ICC is the mean of per-chunk cohesion scores.

**Contextual Coherence (DCC)** measures the smoothness of transitions between adjacent chunks. For each pair of consecutive chunks, it computes the cosine similarity of their embeddings. Higher values indicate smoother transitions. Very high values may indicate under-splitting (the chunks are so similar they should have been one chunk), but in practice this metric rewards methods that avoid jarring topic jumps at chunk boundaries.

**Block Integrity (BI)** measures whether structural elements are preserved intact. It counts the fraction of tables, code blocks, lists, and block quotes that are fully contained within a single chunk rather than split across boundaries. A score of 1.0 means no structural element is broken.

**Reference Completeness (RC)** measures whether chunks start with unresolved references. For each chunk boundary, it checks whether the next chunk begins with a pronoun or demonstrative that has no antecedent in the same chunk, and whether entities introduced in the previous chunk are orphaned. RC maps directly to the orphan risk and entity continuity signals from cognitive chunking.

The five metrics are combined into a composite score with configurable weights. The default is equal weighting (0.20 each). Custom weights let you prioritize what matters for your use case: a RAG pipeline might weight ICC and RC higher; a structural preservation pipeline might weight BI higher.

## Pre-screening

Running every candidate method on every document would be wasteful. Pre-screening applies lightweight heuristics to skip methods that are unlikely to help:

- **Topology-aware** is skipped if the document has fewer than two heading levels. A flat document has no topology to preserve.
- **Intent-driven** is skipped if the document is under 500 tokens. Too short for meaningful intent generation.
- **Enriched** is skipped if the document has no markdown structure and is under 1000 tokens. Too simple to benefit from the enrichment overhead.
- **Cognitive** and **semantic** are always included as general-purpose methods.

Pre-screening is advisory. The `--force-candidates` flag overrides it.

## How winner selection works

After all candidate methods have run, each method's output is scored on all five metrics. The composite scores are compared, and the method with the highest score wins. Ties are broken by preferring fewer chunks (less fragmentation).

The full quality report is available in the output (JSON format, `--report` flag). This includes each candidate's per-metric scores, composite score, chunk count, and total tokens. The report is valuable for understanding why the adaptive router chose a particular method and for identifying documents where multiple methods perform similarly.

## Configuration

**`--candidates`** restricts which methods are evaluated. Default: `semantic,cognitive,intent,enriched,topo`. Restricting candidates reduces cost but may miss the best method.

**`--force-candidates`** bypasses pre-screening heuristics, running all specified candidates regardless of document characteristics.

**`--metric-weights`** sets custom weights for the composite score. Format: `sc=0.15,icc=0.25,dcc=0.20,bi=0.20,rc=0.20`. Weights must sum to 1.0.

**`--soft-budget`** and **`--hard-budget`** are passed through to all candidate methods.

## CLI usage

```sh
# Adaptive chunking with all candidates
cognigraph-chunker adaptive -i document.md -p openai --api-key $OPENAI_API_KEY

# Restrict candidates
cognigraph-chunker adaptive -i doc.md -p openai --api-key $KEY \
  --candidates semantic,cognitive,intent

# Custom metric weights (prioritize cohesion and reference completeness)
cognigraph-chunker adaptive -i doc.md -p openai --api-key $KEY \
  --metric-weights sc=0.10,icc=0.30,dcc=0.15,bi=0.15,rc=0.30

# Include full quality report in output
cognigraph-chunker adaptive -i doc.md -p openai --api-key $KEY -f json --report

# Force all candidates (skip pre-screening)
cognigraph-chunker adaptive -i doc.md -p openai --api-key $KEY --force-candidates
```

## API usage

```
POST /api/v1/adaptive

{
  "text": "...",
  "provider": "openai",
  "model": "text-embedding-3-small",
  "api_key": "...",
  "candidates": ["semantic", "cognitive", "intent", "enriched"],
  "soft_budget": 512,
  "hard_budget": 768,
  "metric_weights": { "sc": 0.20, "icc": 0.20, "dcc": 0.20, "bi": 0.20, "rc": 0.20 },
  "include_report": true
}
```

The response includes the winner's name, its chunks, and optionally the full quality report with per-candidate metrics.

## When to use adaptive chunking

Use it when you process documents from diverse sources and cannot predict which chunking method will work best for each one. A pipeline ingesting research papers, blog posts, API documentation, and compliance manuals benefits from adaptive routing because each document type has different structural properties.

Use it for benchmarking. The quality report provides a side-by-side comparison of how different methods perform on the same document. This is valuable when evaluating chunking strategies for a new domain or when fine-tuning method parameters.

It is less useful when you already know which method works best for your documents (use that method directly) or when cost is the primary constraint (adaptive runs multiple methods, multiplying the compute cost).

Adaptive chunking requires an embedding provider (for ICC and DCC metrics) and may require an LLM (depending on which candidates are included). The cost is the sum of running all non-screened candidates plus the metric computation.

## The quality metrics as a standalone module

The five quality metrics are implemented as a standalone module that can evaluate any chunking output, not just adaptive candidates. The evaluation API endpoint (`POST /api/v1/evaluate`) accepts pre-chunked text and returns metric scores. This enables continuous quality monitoring: run your chunking pipeline, feed the output to the evaluator, and track quality over time.

The metrics are also available as assertions in tests. A CI pipeline can chunk a reference document, compute metrics, and assert that size compliance stays above 0.85 or reference completeness stays above 0.90. This catches regressions in chunking quality as the codebase evolves.
