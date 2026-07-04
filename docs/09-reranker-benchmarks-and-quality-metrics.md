# Reranker Benchmarks and Quality Metrics

When chunking text for retrieval-augmented generation (RAG), the boundaries between chunks matter as much as the chunks themselves. A poorly placed boundary can strand an entity in a single chunk, sever a line of reasoning mid-sentence, or leave a pronoun without its antecedent. Cognition-aware chunking addresses this with multi-signal boundary scoring — but some boundaries remain ambiguous even after scoring. That is where **cross-encoder rerankers** come in.

This article presents benchmark results across five reranker providers, explains what the quality metrics mean for downstream RAG performance, and offers guidance on choosing the right provider for your deployment.

## What Rerankers Do in Cognitive Chunking

The cognitive chunking pipeline assigns a **join score** to every sequential boundary based on eight signals: semantic similarity, entity continuity, discourse continuation, heading context, structural affinity, topic shift, orphan risk, and budget pressure. Most boundaries score clearly high (keep together) or clearly low (split here). But roughly 10–20% of boundaries fall into an **ambiguous band** — close to the mean, where the heuristic signals disagree.

For these ambiguous boundaries, the pipeline optionally sends adjacent text pairs to a **cross-encoder reranker**. Cross-encoders process both texts jointly (rather than comparing independent embeddings), producing a more accurate relevance score. The pipeline then updates the boundary score and re-evaluates whether to split.

This staged approach avoids expensive cross-encoder inference on every boundary. Only the uncertain ones get reranked, keeping total API calls low while improving precision where it matters.

## Quality Metrics

The cognitive pipeline reports five quality metrics after assembly. Understanding them is key to interpreting benchmark results.

### Entity Orphan Rate

**What it measures:** The percentage of chunk boundaries that cut through an entity chain — the same entities continue across the boundary, but the split strands them in separate chunks.

**Why it matters:** In RAG, a query might match a chunk mentioning "Phase 3 clinical trials." If the boundary severed the entity chain, the related context about what Phase 3 entails, who is conducting it, or what the results were lives in the neighboring chunk that retrieval may never return. The entity is *orphaned*.

**Target:** 0%. No boundary should sever a continuing entity chain.

### Pronoun Boundary Rate

**What it measures:** The percentage of chunk boundaries where the next chunk starts with an unresolved pronoun ("it", "they", "this", "these").

**Why it matters:** A chunk beginning with "They concluded that the treatment was effective" is useless without knowing who "they" refers to. If the antecedent is in the previous chunk and retrieval only returns this one, the model hallucinates or hedges.

**Target:** 0%.

### Heading Attachment Rate

**What it measures:** The percentage of headings that stay attached to the content that follows them — no chunk boundary falls between a heading and its first content block.

**Why it matters:** Heading context is metadata gold for retrieval. A chunk that knows it lives under "Architecture > Scoring" can be filtered, faceted, or boosted by section relevance. A heading split from its body produces a stranded title and a chunk that starts mid-section.

**Target:** 100%.

### Discourse Break Rate

**What it measures:** The percentage of chunk boundaries that fall inside a discourse unit — splitting mid-argument, mid-explanation, or mid-comparison.

**Why it matters:** Discourse markers like "however", "therefore", "for example" signal logical relationships. Splitting after "however" severs the contrast. The retrieved chunk has a conclusion without its premise.

**Target:** 0%.

### Triple Severance Rate

**What it measures:** The percentage of subject-verb-object triples (extracted via LLM) that get split across chunk boundaries, separating the subject from its predicate or object.

**Why it matters:** If a chunk contains "The pipeline processes" but the object "ambiguous boundaries using cross-encoders" is in the next chunk, retrieval loses the complete proposition.

**Target:** 0%. (In the current pipeline this is reported as 0.0% by construction: triples are extracted per chunk after assembly, so a triple cannot span a boundary.)

## Priority Order for RAG Quality

Not all metrics are equally impactful. In practice:

1. **Entity orphan rate** — directly determines whether retrieval can follow cross-references
2. **Discourse break rate** — preserves reasoning chains and logical flow
3. **Pronoun boundary rate** — ensures chunks are self-contained
4. **Heading attachment rate** — enables section-aware retrieval
5. **Chunk granularity** — smaller chunks improve retrieval precision, but not at the cost of the above

A reranker that produces 30 tiny chunks with 5% entity orphans is *worse* for RAG than one producing 25 slightly larger chunks with 0% orphans. The orphaned entities become retrieval dead ends.

## Benchmark Setup

All benchmarks use the same conditions:

- **Document:** `docs/08-the-complete-chunking-toolkit.md` (~19.5 KB, 181 blocks)
- **Embedding provider:** OpenAI `text-embedding-3-small`
- **Budgets:** soft=512, hard=768 tokens (defaults)
- **Machine:** macOS, Apple Silicon
- **Measurement:** wall-clock time including embedding + reranking API calls
- **ONNX model:** `ms-marco-MiniLM-L-6-v2` (~22 MB, local inference)
- **Software:** July 2026 build — byte-based token estimation (~4 bytes/token, so token figures read higher than in earlier revisions of this article), correctly weighted reranker blending, and concurrent rerank calls

The baseline runs cognitive chunking without any reranker — pure heuristic boundary scoring.

## Results

| Metric | Baseline | ONNX (local) | NVIDIA Nemotron 1B | Cloudflare BGE | Cohere v3.5\* | OAuth (corporate)\* |
|--------|----------|--------------|--------------------|----------------|--------------|-------------------|
| **Time** | 1.7s | 3.1s | 4.6s | 8.3s | 15.5s | 40.7s |
| **Chunks** | 30 | 43 | 44 | 43 | 33 | 29 |
| **Avg tokens** | 164 | 114 | 112 | 114 | 54 | 62 |
| **Max tokens** | 522 | 272 | 272 | 272 | 130 | 181 |
| **Entity orphan** | 0.0% | 0.0% | 0.0% | 0.0% | 3.1% | 3.6% |
| **Pronoun boundary** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Heading attachment** | 100% | 100% | 100% | 100% | 100% | 100% |
| **Discourse break** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

\* Prior-round measurements (February 2026: whitespace-based token counts, pre-fix reranker blending, smaller revision of the benchmark document) — not directly comparable to the other columns. Pending a credential refresh for a re-run.

The Cloudflare column was measured against the direct Workers AI API
(no AI Gateway routing); routing through an AI Gateway adds a proxy
hop and requires a token with the AI Gateway Run permission.

### Provider Details

- **ONNX** (`ms-marco-MiniLM-L-6-v2`) — local cross-encoder, ~22 MB model
- **NVIDIA Nemotron 1B** (`nvidia/llama-nemotron-rerank-1b-v2`) — NVIDIA NIM API
- **Cohere v3.5** (`rerank-v3.5`) — Cohere Rerank API
- **Cloudflare BGE** (`@cf/baai/bge-reranker-base`) — Cloudflare Workers AI
- **OAuth** — Corporate API gateway behind OAuth2 client credentials, routing through reverse proxy layers (pharmaceutical client deployment)

## Analysis

### ONNX (Local): Best Overall

The surprise winner. Running `ms-marco-MiniLM-L-6-v2` locally produces **near-identical results to NVIDIA Nemotron 1B** — 43 vs 44 chunks, the same 272-token maximum, 0% entity orphans — but in just **3.1 seconds**, only 1.4s more than baseline. No network round-trips, no API keys, no per-call costs.

The 22 MB model is small enough to bundle with deployments. For Docker containers, it adds negligible image size. For CI/CD pipelines, it can be cached. The only requirement is ONNX Runtime available at runtime (`brew install onnxruntime` on macOS, or the system package on Linux).

This is the recommended choice whenever local inference is feasible. It delivers the best quality at near-baseline speed with zero operational overhead.

### NVIDIA Nemotron 1B: Best Cloud Option

Matches ONNX quality almost exactly — **0% entity orphans**, 44 chunks, the same 272 max tokens — making it the best cloud-based reranker. At 4.6 seconds (~2.7x baseline, with ambiguous boundaries now scored concurrently), the latency overhead is moderate and predictable. For batch processing or offline indexing, this is negligible. For real-time chunking, it adds roughly 3 seconds of reranking on top of the embedding phase.

Choose NVIDIA over ONNX when you cannot bundle a local model (e.g., serverless deployments, thin containers) or when you want a managed service with no local dependencies.

### Cloudflare: Matches Local Quality

With the corrected reranker blending (July 2026), Cloudflare BGE produces results **identical to ONNX** — 43 chunks, 272 max tokens, 0% entity orphans — at 8.3 seconds. Notably, the ~3% entity orphan rate reported for Cloudflare in the February round turned out to be an artifact of the earlier over-applied reranker delta, not a property of the BGE model: with the delta correctly weighted, the quality cost disappears entirely.

If you already hold Cloudflare credentials (shared with the embedding provider), this is a solid managed option — slower than NVIDIA but with no additional vendor relationship.

### Cohere: Pending Re-Run

*(Figures in this section and the OAuth section are from the February 2026 round — see the table note.)*

Cohere produced more chunks (33) with smaller average sizes and a ~3.1% entity orphan rate in the prior round. Given that Cloudflare's orphan rate proved to be a blending artifact rather than a model property, Cohere's number deserves the same re-examination once credentials are refreshed.

Cohere v3.5 and v4.0-fast produced identical results on this document, suggesting the models score ambiguous boundaries similarly. The "fast" variant may show advantages on larger documents or higher throughput workloads.

### OAuth (Corporate Gateway): Functional but Slow

The OAuth reranker demonstrates that the pipeline works end-to-end through corporate API gateways with OAuth2 authentication, reverse proxies, and custom endpoint paths. The 40.7-second latency reflects network overhead from multiple proxy layers rather than model quality — the actual reranking quality in the February round (29 chunks, 3.6% entity orphan) was comparable to Cohere's, and like Cohere's may improve under the corrected blending.

This provider exists for enterprises that cannot send data to public APIs and must route through their own infrastructure.

### Baseline: When Reranking Is Not Worth It

The baseline produces excellent quality metrics on its own — zero across all error rates. The only downside is chunk granularity: 30 chunks with a max of 522 tokens means some chunks are large. If your retrieval system handles larger chunks well (e.g., with 512-token windows), the baseline may be sufficient and several times faster than any cloud reranker.

## Choosing a Reranker

| Scenario | Recommendation |
|----------|---------------|
| **Best quality + speed** | ONNX local (`ms-marco-MiniLM-L-6-v2`) |
| **Best cloud option** | NVIDIA Nemotron 1B |
| **Already using Cloudflare** | Cloudflare BGE (shared credentials with embedding provider) |
| **Multi-provider flexibility** | Cohere v3.5 (no infrastructure lock-in) |
| **Corporate/air-gapped** | OAuth (configurable endpoint, works behind proxies) |
| **Latency-sensitive, no model** | No reranker (baseline is excellent on its own) |

## Configuration

Each provider reads credentials from environment variables or `.env.*` files:

```sh
# Baseline — no reranker, pure heuristic boundary scoring
cognigraph-chunker cognitive -i doc.md -p openai -f json

# NVIDIA NIM
echo 'NVIDIA_API_KEY=nvapi-...' > .env.nvidia
echo 'NVIDIA_RERANK_MODEL=nvidia/llama-nemotron-rerank-1b-v2' >> .env.nvidia
cognigraph-chunker cognitive -i doc.md -p openai --reranker nvidia

# Cohere
echo 'COHERE_API_KEY=...' > .env.cohere
cognigraph-chunker cognitive -i doc.md -p openai --reranker cohere

# Cloudflare (shared with embedding provider)
cognigraph-chunker cognitive -i doc.md -p cloudflare --reranker cloudflare

# OAuth (shared with embedding provider, custom endpoint)
echo 'OAUTH_RERANK_PATH=/rerank' >> .env.oauth
echo 'OAUTH_RERANK_MODEL=rerank-model' >> .env.oauth
cognigraph-chunker cognitive -i doc.md -p oauth --reranker oauth

# Local ONNX model
cognigraph-chunker cognitive -i doc.md --reranker onnx:./models/ms-marco-MiniLM-L-6-v2
```

The embedding provider and reranker are independent — any combination works. The benchmark-winning configuration is a local ONNX model with near-zero overhead:

```sh
cognigraph-chunker cognitive -i doc.md -p openai --reranker onnx:./models/ms-marco-MiniLM-L-6-v2
```

When local models are not an option, the best cloud configuration is OpenAI embeddings + NVIDIA reranking:

```sh
cognigraph-chunker cognitive -i doc.md -p openai --reranker nvidia
```

## Conclusion

Cross-encoder reranking is a precision tool, not a necessity. The cognitive pipeline's heuristic scoring already produces excellent results. Rerankers shine when you need finer granularity (smaller chunks for tighter retrieval windows) without sacrificing entity coherence.

The standout finding is that a local 22 MB ONNX model (`ms-marco-MiniLM-L-6-v2`) matches the best cloud APIs in quality while adding only ~1.4 seconds of overhead. If you can bundle the model, there is no reason to pay for cloud reranking. When local inference is not feasible, NVIDIA's Nemotron 1B (fastest) and Cloudflare BGE (shared credentials with the embedding provider) both deliver the same perfect quality metrics — under the corrected reranker blending, the quality gap between providers that the earlier round suggested has essentially vanished.
