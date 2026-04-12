# The Complete Chunking Toolkit: From Bytes to Meaning

The previous articles walked through a progression. We started with why chunking matters for AI retrieval, moved through fixed-size and delimiter strategies, introduced semantic chunking with embeddings and signal processing, showed how markdown-aware parsing preserves document structure, explained how token-aware merging normalizes chunk sizes, brought it all together in a single toolkit, and introduced cognition-aware chunking that preserves meaning across boundaries. Since then, the toolkit has grown to include intent-driven chunking, topology-aware chunking, enriched chunking, and adaptive method selection. This article is the complete picture -- every chunking strategy CogniGraph Chunker offers, when to use each one, and how they compose into a pipeline that handles anything from a 500-word blog post to a 200-page clinical trial protocol.

## Eight strategies, one toolkit

CogniGraph Chunker implements eight chunking strategies. They form a progression along two axes: structural awareness and intelligence source. The first four strategies use heuristics and embeddings. The last four add LLM reasoning.

```
Heuristic / Embedding methods:
  Fixed-size        -> byte-level splitting with boundary awareness
    + Delimiter     -> sentence-level splitting with pattern matching
      + Semantic    -> embedding-based topic boundary detection
        + Cognitive -> multi-signal boundary scoring with enrichment

LLM-augmented methods:
  Intent-driven     -> query-aligned boundaries via DP optimization
  Enriched          -> self-describing chunks with metadata extraction
  Topology-aware    -> SIR tree + dual-agent section classification
  Adaptive          -> meta-router: runs candidates, picks best by quality score
```

Token-aware merging can be applied as a post-processing step after any strategy, grouping small chunks into right-sized units for your embedding model. Markdown-aware parsing runs automatically in the semantic, cognitive, intent, enriched, and topology-aware pipelines.

### 1. Fixed-size chunking

The simplest strategy. Set a target size in bytes or characters, and the chunker walks through the text producing chunks of that size. Boundary awareness softens the edges -- instead of cutting mid-word, the chunker looks for the nearest whitespace or punctuation to break cleanly.

This is the right choice when uniformity matters more than coherence. Embedding models with strict token limits, batch processing pipelines that need predictable chunk sizes, or situations where the text has no meaningful structure to preserve.

```sh
cognigraph-chunker chunk -i document.md -s 1024
```

**Strengths:** Fast, predictable, zero configuration.
**Weaknesses:** Splits mid-paragraph, mid-sentence, mid-thought. No awareness of content.

### 2. Delimiter splitting

Split at every occurrence of a delimiter -- sentence-ending punctuation, newlines, custom patterns. Multi-pattern matching uses an Aho-Corasick automaton compiled once and reused, so even complex delimiter sets add no per-character overhead.

This produces natural units (sentences, paragraphs) rather than arbitrary byte ranges. Combine it with token-aware merging to group short sentences into right-sized chunks.

```sh
cognigraph-chunker split -i document.md -d ".?!" --merge --chunk-size 256
```

**Strengths:** Respects sentence boundaries, configurable patterns, fast.
**Weaknesses:** No topic awareness. Two adjacent sentences about completely different subjects stay together if they fall within the merge window.

### 3. Semantic chunking

The first strategy that understands content. Each block gets an embedding vector from an embedding model, and the pipeline computes cosine similarity between adjacent blocks. A Savitzky-Golay filter smooths the similarity curve, and local minima (valleys) become candidate split points. Where similarity drops, topics change, and the chunker splits.

This requires an embedding provider -- Ollama, OpenAI, ONNX Runtime, Cloudflare Workers AI, or an OAuth-protected endpoint -- but the result is chunks that align with actual topic boundaries rather than arbitrary positions.

```sh
cognigraph-chunker semantic -i document.md --provider ollama
```

**Strengths:** Topic-aware boundaries, works across document types, tunable parameters.
**Weaknesses:** Depends on embedding quality and latency. Treats all boundaries as a single-signal decision. Can split entity chains, causal sequences, and discourse continuations because embedding similarity alone doesn't capture these.

### 4. Cognitive chunking

Extends semantic chunking with eight signals instead of one. Beyond embedding similarity, the boundary scorer evaluates entity continuity, discourse continuation, heading context, structural affinity, topic shift, orphan risk, and budget pressure. Blocks are enriched with named entities, pronouns, demonstratives, discourse markers, and heading ancestry before scoring.

The result is chunks that preserve not just topics but propositions -- the "who did what to whom" structure that makes text retrievable and answerable.

```sh
cognigraph-chunker cognitive -i document.md --provider ollama
```

**Strengths:** Preserves entity chains, causal links, and discourse structure. Heading attachment. Cross-chunk entity tracking. Quality metrics on every run. Multilingual support for 14 language groups. Optional cross-encoder reranking on ambiguous boundaries.
**Weaknesses:** More computation than semantic chunking (though embedding latency still dominates). Not needed for simple, topically distinct documents.

### 5. Intent-driven chunking

Optimizes boundaries for predicted user queries rather than topic transitions. An LLM generates 10-30 hypothetical queries that users might ask about the document, classified by type (factual, procedural, conceptual, comparative). Each candidate chunk is scored by the cosine similarity between its centroid embedding and the intent embeddings. Dynamic programming finds the globally optimal partition -- unlike greedy approaches, the DP evaluates downstream consequences of every boundary decision.

This is the right choice when retrieval quality is the primary objective and you can tolerate the cost of an LLM call plus embedding the generated intents. It works best for reference manuals, knowledge bases, FAQ compilations, and compliance documents where users have diverse, specific information needs. See [Article 10](10-intent-driven-chunking.md) for the full pipeline description.

```sh
cognigraph-chunker intent -i document.md -p openai --api-key $OPENAI_API_KEY
```

**Strengths:** Chunks align with how readers search, not how authors organize. DP optimization avoids greedy traps. Partition score provides a direct quality measure.
**Weaknesses:** Requires both an LLM and an embedding provider -- the most expensive method per document. Less useful for short documents or documents where structure itself is the information.
**Requires:** Embedding provider + LLM.

### 6. Enriched chunking

Produces self-describing chunks. Each chunk carries seven metadata fields extracted in a single LLM call: title, summary, keywords, typed entities, hypothetical questions, semantic keys, and category. Initial grouping is purely structural (no embeddings needed). The semantic keys create a rolling concept dictionary -- when the LLM processes chunks sequentially, it reuses keys for recurring concepts, creating explicit links between chunks. A recombination step merges chunks sharing the same keys using bin-packing.

This is the right choice when your retrieval pipeline supports hybrid search (BM25 + dense vectors), when you need HyDE-style retrieval, or when chunks need to be self-describing for downstream consumers that cannot access the original document. See [Article 12](12-enriched-chunking.md) for the full pipeline description.

```sh
cognigraph-chunker enriched -i document.md --api-key $OPENAI_API_KEY
```

**Strengths:** Rich metadata enables hybrid retrieval (BM25 over keywords, HyDE via hypothetical questions, entity-type filtering, category routing). Semantic key dictionary provides concept-level document index. No embedding provider needed.
**Weaknesses:** LLM cost scales linearly with chunk count. Metadata fields go unused if retrieval is purely dense-vector-based.
**Requires:** LLM only (no embeddings).

### 7. Topology-aware chunking

Builds a Structured Intermediate Representation (SIR) -- a tree mirroring the document's heading hierarchy with content blocks as leaves and cross-reference edges linking blocks that share entities or discourse continuations. Two LLM agents then make boundary decisions. The Inspector classifies each section node as atomic (keep together), splittable (can divide at block boundaries), or merge candidate (too small to stand alone). The Refiner determines optimal split points for splittable sections, merge directions for small sections, and handles cross-section dependencies.

This is the right choice for deeply nested documents where heading hierarchy carries structural meaning: research papers, technical specifications, API documentation, legal documents with articles and sub-articles. See [Article 11](11-topology-aware-chunking.md) for the full pipeline description.

```sh
cognigraph-chunker topo -i document.md --api-key $OPENAI_API_KEY
```

**Strengths:** Preserves heading hierarchy and structural coupling. Cross-reference annotations between dependent sections. SIR construction is purely heuristic (no LLM calls). No embedding provider needed.
**Weaknesses:** Less useful for flat documents without heading structure. LLM calls for two agents add latency.
**Requires:** LLM only (no embeddings).

### 8. Adaptive chunking

A meta-router. It runs multiple candidate methods on the same document, scores each method's output using five intrinsic quality metrics, and returns the output from the method that scores highest. Pre-screening heuristics skip methods unlikely to help for a given document (topology-aware is skipped for flat documents, intent-driven for short documents, enriched for simple unstructured text).

The five quality metrics, each scored 0.0 to 1.0:

- **Size Compliance** -- fraction of chunks within the target size range
- **Intrachunk Cohesion** -- mean sentence-to-chunk cosine similarity (is each chunk about one thing?)
- **Contextual Coherence** -- cosine similarity between adjacent chunks (smooth transitions?)
- **Block Integrity** -- fraction of structural elements (tables, code, lists) preserved intact
- **Reference Completeness** -- absence of orphaned pronouns and dangling entity references at boundaries

The full quality report is available with the `--report` flag, providing side-by-side comparison of how each candidate performed. See [Article 13](13-adaptive-chunking.md) for the full pipeline description.

```sh
cognigraph-chunker adaptive -i document.md -p openai --api-key $OPENAI_API_KEY \
  --candidates semantic,cognitive,intent
```

**Strengths:** Per-document method selection without manual tuning. Quality report useful for benchmarking. Pre-screening keeps cost practical.
**Weaknesses:** Most expensive option (runs multiple methods). Not useful when you already know which method works best.
**Requires:** Embedding provider + optionally LLM (depending on candidates).

## Choosing a strategy

The decision depends on your documents, your latency budget, your cost tolerance, and how much retrieval quality matters.

| Scenario | Recommended strategy |
|----------|---------------------|
| High-throughput ingestion, uniform chunk sizes needed | Fixed-size |
| Sentence-level granularity for fine-grained retrieval | Delimiter + merge |
| General documents with clear topic structure | Semantic |
| Documents with entity chains, cross-references, causal reasoning | Cognitive |
| Mission-critical retrieval where boundary precision matters | Cognitive + reranker |
| Retrieval-optimized chunks aligned with user queries | Intent-driven |
| Hybrid search pipelines needing rich metadata per chunk | Enriched |
| Deeply nested documents with heading hierarchy | Topology-aware |
| Diverse document types where no single method fits all | Adaptive |

A practical approach is to start with semantic chunking. If retrieval quality is good enough, stay there. If retrieved chunks lack context -- they start with "It" or reference entities defined elsewhere -- upgrade to cognitive chunking. If your documents are deeply nested specifications, try topology-aware. If your retrieval pipeline supports hybrid search, enriched chunking provides the metadata to exploit it. If you process diverse document types and cannot predict the best method, use adaptive.

At each step, the output format stays the same, the interfaces stay the same, and the downstream pipeline doesn't change. The chunking strategy is a parameter, not an architecture decision.

## Four interfaces, every strategy

All eight strategies are available through all four interfaces:

**CLI** -- experiment and iterate. Pipe documents through subcommands (`chunk`, `split`, `semantic`, `cognitive`, `intent`, `enriched`, `topo`, `adaptive`), adjust parameters with flags, and export to plain text, JSON, or JSONL.

```sh
# Fixed-size
cognigraph-chunker chunk -i doc.md -s 1024 -f json

# Delimiter with merge
cognigraph-chunker split -i doc.md -d ".?!" --merge --chunk-size 256 -f json

# Semantic
cognigraph-chunker semantic -i doc.md --provider openai -f json

# Cognitive with relations and graph export
cognigraph-chunker cognitive -i doc.md --provider openai --relations --graph -f json

# Intent-driven with custom intent count
cognigraph-chunker intent -i doc.md -p openai --api-key $KEY --max-intents 30

# Enriched with metadata extraction
cognigraph-chunker enriched -i doc.md --api-key $KEY -f json

# Topology-aware with SIR output
cognigraph-chunker topo -i doc.md --api-key $KEY -f json --emit-sir

# Adaptive with quality report
cognigraph-chunker adaptive -i doc.md -p openai --api-key $KEY -f json --report
```

**REST API** -- deploy as a microservice. Start with `cognigraph-chunker serve` and call endpoints:

| Endpoint | Strategy |
|----------|----------|
| `POST /api/v1/chunk` | Fixed-size |
| `POST /api/v1/split` | Delimiter |
| `POST /api/v1/semantic` | Semantic |
| `POST /api/v1/cognitive` | Cognitive |
| `POST /api/v1/intent` | Intent-driven |
| `POST /api/v1/enriched` | Enriched |
| `POST /api/v1/topo` | Topology-aware |
| `POST /api/v1/adaptive` | Adaptive |
| `POST /api/v1/merge` | Token-aware merge (post-processing) |

Bearer token authentication, CORS, SSRF protection, request size limits, and timeouts are built in.

**Python bindings** -- Rust performance in Python workflows. The `Chunker` class handles fixed-size and delimiter splitting. The `semantic_chunk()` function runs the full semantic pipeline. Signal processing primitives (`savgol_filter()`, `windowed_cross_similarity()`) are exposed individually with NumPy support.

**Docker** -- container deployment with a single image. The multi-stage Dockerfile produces a minimal binary. Set `PORT`, `API_KEY`, and provider credentials via environment variables.

## Five embedding providers

The semantic, cognitive, intent, and adaptive strategies require embeddings. CogniGraph Chunker supports five providers through a unified trait -- switching between them is a one-parameter change.

| Provider | Use case | Configuration |
|----------|----------|---------------|
| **Ollama** (default) | Local development, air-gapped environments | Zero config if Ollama is running |
| **OpenAI** | Production quality, cloud-based | API key via flag, env var, or `.env.openai` |
| **ONNX Runtime** | Local inference, custom models | Path to `model.onnx` + `tokenizer.json` |
| **Cloudflare Workers AI** | Teams on Cloudflare infrastructure | Auth token + account ID, optional AI Gateway |
| **OAuth** | Enterprise API gateways (Azure, corporate proxies) | Client credentials, token URL, optional TLS override |

Strategies that need only an LLM (enriched, topo) use an OpenAI-compatible completion API directly. The default model is `gpt-4.1-mini`, configurable via `COGNIGRAPH_LLM_MODEL` or per-call flags.

## LLM enrichment features

Several features use LLM calls to add structured information to chunks. These are available across multiple strategies:

**Relation extraction** (`--relations` flag on cognitive chunking) -- LLM-based subject-predicate-object triple extraction per chunk. Produces structured knowledge that supports graph-based retrieval.

**Chunk synopsis** (`--synopsis` flag on cognitive chunking) -- LLM-generated one-sentence summaries per chunk for preview and navigation.

**Graph export** (`--graph` flag on cognitive chunking) -- output as nodes (chunks) and edges (adjacency + shared entity links), ready for import into graph databases or visualization tools.

**Enriched metadata** (enriched chunking mode) -- title, summary, keywords, typed entities, hypothetical questions, semantic keys, and category per chunk. Enables hybrid retrieval strategies beyond dense vector similarity.

## Performance characteristics

The core algorithms are Rust, operating on byte slices for zero-copy performance.

| Strategy | Typical latency | Bottleneck |
|----------|----------------|------------|
| Fixed-size | Microseconds | Memory bandwidth |
| Delimiter | Microseconds | Aho-Corasick automaton (compiled once) |
| Semantic | Seconds | Embedding provider API |
| Cognitive | Seconds | Embedding provider API (enrichment adds negligible overhead) |
| Cognitive + reranker | Seconds | Embedding + reranker inference on ambiguous boundaries |
| Intent-driven | Seconds | LLM (intent generation) + embedding provider |
| Enriched | Seconds | LLM (one call per chunk for metadata extraction) |
| Topology-aware | Seconds | LLM (two agent calls: Inspector + Refiner) |
| Adaptive | Sum of candidates | Runs multiple methods; pre-screening reduces cost |

For embedding-based strategies, embedding computation dominates. The signal processing, enrichment, scoring, assembly, and healing stages together take milliseconds even on large documents. For LLM-based strategies, the LLM calls dominate. With a local ONNX embedding model, end-to-end processing of a 100-page document completes in seconds. With a remote API, network latency is the limiting factor.

## From simple to sophisticated

The value of having all eight strategies in one toolkit is that you can move along the complexity axis without changing your infrastructure.

Start with fixed-size chunking to get something working. Move to delimiter splitting when you need sentence-level boundaries. Add semantic chunking when you need topic awareness. Upgrade to cognitive chunking when retrieved chunks lack context. Switch to intent-driven when you want chunks optimized for how users actually search. Use enriched chunking when your pipeline can exploit rich metadata. Try topology-aware for deeply nested documents. Deploy adaptive when you process diverse documents and want automatic method selection.

At each step, the output format stays the same, the interfaces stay the same, and the downstream pipeline doesn't change. The chunking strategy is a parameter, not an architecture decision.

CogniGraph Chunker is open source under the MIT license. The source, documentation, and issue tracker are on GitHub at [gedankrayze/cognigraph-chunker](https://github.com/gedankrayze/cognigraph-chunker).
