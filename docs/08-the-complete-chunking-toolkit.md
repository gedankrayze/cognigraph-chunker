# The Complete Chunking Toolkit: From Bytes to Meaning

The previous seven articles walked through a progression. We started with why chunking matters for AI retrieval, moved through fixed-size and delimiter strategies, introduced semantic chunking with embeddings and signal processing, showed how markdown-aware parsing preserves document structure, explained how token-aware merging normalizes chunk sizes, brought it all together in a single toolkit, and finally introduced cognition-aware chunking that preserves meaning across boundaries. This article is the complete picture — every chunking strategy CogniGraph Chunker offers, when to use each one, and how they compose into a pipeline that handles anything from a 500-word blog post to a 200-page clinical trial protocol.

## Five strategies, one toolkit

CogniGraph Chunker implements five chunking strategies. They form a hierarchy: each level builds on the one below it, adding capability at the cost of complexity.

### 1. Fixed-size chunking

The simplest strategy. Set a target size in bytes or characters, and the chunker walks through the text producing chunks of that size. Boundary awareness softens the edges — instead of cutting mid-word, the chunker looks for the nearest whitespace or punctuation to break cleanly.

This is the right choice when uniformity matters more than coherence. Embedding models with strict token limits, batch processing pipelines that need predictable chunk sizes, or situations where the text has no meaningful structure to preserve.

```sh
cognigraph-chunker chunk -i document.md -s 1024
```

**Strengths:** Fast, predictable, zero configuration.
**Weaknesses:** Splits mid-paragraph, mid-sentence, mid-thought. No awareness of content.

### 2. Delimiter splitting

Split at every occurrence of a delimiter — sentence-ending punctuation, newlines, custom patterns. Multi-pattern matching uses an Aho-Corasick automaton compiled once and reused, so even complex delimiter sets add no per-character overhead.

This produces natural units (sentences, paragraphs) rather than arbitrary byte ranges. Combine it with token-aware merging to group short sentences into right-sized chunks.

```sh
cognigraph-chunker split -i document.md -d ".?!" --merge --chunk-size 256
```

**Strengths:** Respects sentence boundaries, configurable patterns, fast.
**Weaknesses:** No topic awareness. Two adjacent sentences about completely different subjects stay together if they fall within the merge window.

### 3. Semantic chunking

The first strategy that understands content. Each block gets an embedding vector from an embedding model, and the pipeline computes cosine similarity between adjacent blocks. A Savitzky-Golay filter smooths the similarity curve, and local minima (valleys) become candidate split points. Where similarity drops, topics change, and the chunker splits.

This requires an embedding provider — Ollama, OpenAI, ONNX Runtime, Cloudflare Workers AI, or an OAuth-protected endpoint — but the result is chunks that align with actual topic boundaries rather than arbitrary positions.

```sh
cognigraph-chunker semantic -i document.md --provider ollama
```

**Strengths:** Topic-aware boundaries, works across document types, tunable parameters.
**Weaknesses:** Depends on embedding quality and latency. Treats all boundaries as a single-signal decision. Can split entity chains, causal sequences, and discourse continuations because embedding similarity alone doesn't capture these.

### 4. Cognitive chunking

Extends semantic chunking with eight signals instead of one. Beyond embedding similarity, the boundary scorer evaluates entity continuity, discourse continuation, heading context, structural affinity, topic shift, orphan risk, and budget pressure. Blocks are enriched with named entities, pronouns, demonstratives, discourse markers, and heading ancestry before scoring.

The result is chunks that preserve not just topics but propositions — the "who did what to whom" structure that makes text retrievable and answerable.

```sh
cognigraph-chunker cognitive -i document.md --provider ollama
```

**Strengths:** Preserves entity chains, causal links, and discourse structure. Heading attachment. Cross-chunk entity tracking. Quality metrics on every run. Multilingual support for 14 language groups.
**Weaknesses:** More computation than semantic chunking (though embedding latency still dominates). Not needed for simple, topically distinct documents.

### 5. Cognitive chunking with reranking

For documents where boundary decisions are close calls, ambiguous boundaries (those within half a standard deviation of the mean join score) can be refined by a cross-encoder model. The reranker scores the text pair through a sequence classification model, and the result is blended with the original similarity score.

Typically 10-20% of boundaries qualify as ambiguous, so the reranker processes a fraction of the total — enough to improve precision without O(n) inference cost.

```sh
cognigraph-chunker cognitive -i document.md --reranker models/ms-marco-MiniLM-L-6-v2
```

**Strengths:** Precision improvement on uncertain boundaries. Selective — only processes ambiguous cases.
**Weaknesses:** Requires an ONNX reranker model. Additional latency on ambiguous boundaries.

## How the strategies compose

These strategies aren't mutually exclusive alternatives — they're layers that compose. The cognitive pipeline includes everything below it:

```
Fixed-size         → byte-level splitting with boundary awareness
  + Delimiter      → sentence-level splitting with pattern matching
    + Markdown     → AST-aware block extraction (headings, tables, code, lists)
      + Semantic   → embedding-based topic boundary detection
        + Cognitive → multi-signal boundary scoring with enrichment
          + Reranker → cross-encoder refinement on ambiguous boundaries
```

Token-aware merging can be applied as a post-processing step after any strategy, grouping small chunks into right-sized units for your embedding model.

Markdown-aware parsing runs automatically in the semantic and cognitive pipelines. You don't need to configure it — the parser detects and preserves headings, tables, code blocks, lists, and block quotes as atomic units.

## Choosing a strategy

The decision depends on your documents, your latency budget, and how much retrieval quality matters.

| Scenario | Recommended strategy |
|----------|---------------------|
| High-throughput ingestion, uniform chunk sizes needed | Fixed-size |
| Sentence-level granularity for fine-grained retrieval | Delimiter + merge |
| General documents with clear topic structure | Semantic |
| Documents with entity chains, cross-references, causal reasoning | Cognitive |
| Mission-critical retrieval where boundary precision matters | Cognitive + reranker |

A practical approach is to start with semantic chunking. If retrieval quality is good enough, stay there. If you notice that retrieved chunks start with "It also..." or reference entities defined elsewhere, upgrade to cognitive chunking. The switch is a one-parameter change — the infrastructure, output format, and downstream pipeline stay the same.

## Four interfaces, every strategy

All strategies are available through all four interfaces:

**CLI** — experiment and iterate. Pipe documents through subcommands (`chunk`, `split`, `semantic`, `cognitive`), adjust parameters with flags, and export to plain text, JSON, or JSONL.

```sh
# Fixed-size
cognigraph-chunker chunk -i doc.md -s 1024 -f json

# Delimiter with merge
cognigraph-chunker split -i doc.md -d ".?!" --merge --chunk-size 256 -f json

# Semantic
cognigraph-chunker semantic -i doc.md --provider openai -f json

# Cognitive with relations and graph export
cognigraph-chunker cognitive -i doc.md --provider openai --relations --graph -f json

# Cognitive with reranker and signal diagnostics
cognigraph-chunker cognitive -i doc.md --reranker models/ms-marco-MiniLM-L-6-v2 --emit-signals
```

**REST API** — deploy as a microservice. Start with `cognigraph-chunker serve` and call endpoints:

| Endpoint | Strategy |
|----------|----------|
| `POST /api/v1/chunk` | Fixed-size |
| `POST /api/v1/split` | Delimiter |
| `POST /api/v1/semantic` | Semantic |
| `POST /api/v1/cognitive` | Cognitive |
| `POST /api/v1/merge` | Token-aware merge (post-processing) |

Bearer token authentication, CORS, SSRF protection, request size limits, and timeouts are built in.

**Python bindings** — Rust performance in Python workflows. The `Chunker` class handles fixed-size and delimiter splitting. The `semantic_chunk()` function runs the full semantic pipeline. Signal processing primitives (`savgol_filter()`, `windowed_cross_similarity()`) are exposed individually with NumPy support.

**Docker** — container deployment with a single image. The multi-stage Dockerfile produces a minimal binary. Set `PORT`, `API_KEY`, and provider credentials via environment variables.

## Five embedding providers

The semantic and cognitive strategies require embeddings. CogniGraph Chunker supports five providers through a unified trait — switching between them is a one-parameter change.

| Provider | Use case | Configuration |
|----------|----------|---------------|
| **Ollama** (default) | Local development, air-gapped environments | Zero config if Ollama is running |
| **OpenAI** | Production quality, cloud-based | API key via flag, env var, or `.env.openai` |
| **ONNX Runtime** | Local inference, custom models | Path to `model.onnx` + `tokenizer.json` |
| **Cloudflare Workers AI** | Teams on Cloudflare infrastructure | Auth token + account ID, optional AI Gateway |
| **OAuth** | Enterprise API gateways (Azure, corporate proxies) | Client credentials, token URL, optional TLS override |

## What cognitive chunking adds

The cognitive pipeline is the most capable strategy in the toolkit. Beyond topic-aware boundaries, it provides:

**Enriched metadata per chunk** — heading ancestry, dominant entities (top 5 by frequency), all entities (full list), token estimate, and continuity confidence score.

**Cross-chunk entity tracking** — a `shared_entities` map showing which entities appear in which chunks. Only entities in 2+ chunks are included, creating a focused index of concept threads that span the document.

**Proposition healing** — after initial assembly, a healing pass scans boundaries for incomplete propositions: unresolved pronouns, dangling demonstratives, discourse continuations, and high entity overlap with similar topics. Chunks that would be more coherent together are merged, as long as the combined size stays within budget.

**Evaluation metrics** on every run — entity orphan rate, pronoun boundary rate, heading attachment rate, and discourse break rate. These provide quantitative chunk quality assessment without human evaluation.

**Relation extraction** (optional) — LLM-based subject-predicate-object triple extraction per chunk, producing structured knowledge that supports graph-based retrieval.

**Graph export** (optional) — output as nodes (chunks) and edges (adjacency + shared entity links), ready for import into graph databases or visualization tools.

**Multilingual support** — automatic language detection across 70+ languages, with language-specific enrichment for 14 language groups including English, German, French, Spanish, Portuguese, Italian, Dutch, Russian, Turkish, Polish, Chinese, Japanese, Korean, and Arabic. CJK text gets script-based entity detection (Katakana spans, Latin-in-CJK terms) alongside the standard heuristics.

## Performance characteristics

The core algorithms are Rust, operating on byte slices for zero-copy performance.

| Strategy | Typical latency | Bottleneck |
|----------|----------------|------------|
| Fixed-size | Microseconds | Memory bandwidth |
| Delimiter | Microseconds | Aho-Corasick automaton (compiled once) |
| Semantic | Seconds | Embedding provider API |
| Cognitive | Seconds | Embedding provider API (enrichment adds negligible overhead) |
| Cognitive + reranker | Seconds | Embedding + reranker inference on ambiguous boundaries |

For semantic and cognitive chunking, embedding computation dominates. The signal processing, enrichment, scoring, assembly, and healing stages together take milliseconds even on large documents. With a local ONNX embedding model, end-to-end processing of a 100-page document completes in seconds. With a remote API like OpenAI, network latency is the limiting factor.

## From simple to sophisticated

The value of having all strategies in one toolkit is that you can move along the complexity axis without changing your infrastructure.

Start with fixed-size chunking to get something working. Move to delimiter splitting when you need sentence-level boundaries. Add semantic chunking when you need topic awareness. Upgrade to cognitive chunking when retrieved chunks lack context — when they start with "It" or reference entities defined elsewhere. Add the reranker when boundary precision on ambiguous cases matters.

At each step, the output format stays the same, the interfaces stay the same, and the downstream pipeline doesn't change. The chunking strategy is a parameter, not an architecture decision.

CogniGraph Chunker is open source under the MIT license. The source, documentation, and issue tracker are on GitHub at [gedankrayze/cognigraph-chunker](https://github.com/gedankrayze/cognigraph-chunker).
