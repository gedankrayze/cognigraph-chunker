# CogniGraph Chunker: Putting It All Together

The previous five articles explored the concepts behind text chunking: why it matters for AI retrieval, how fixed-size and delimiter strategies handle structural boundaries, how semantic analysis detects topic shifts using signal processing, how markdown-aware parsing preserves document structure, and how token-aware merging normalizes chunk sizes. CogniGraph Chunker is the toolkit that brings all of these ideas into a single, production-ready package.

## One library, eight strategies

CogniGraph Chunker implements eight chunking strategies from a shared core. They range from byte-level splitting to an adaptive meta-router that picks the best method automatically. All eight share the same output format, the same interfaces, and can all be followed by token-aware merging as a post-processing step.

This matters in practice because most teams start with fixed-size chunking, realize they need better boundary detection, move to delimiter splitting, and eventually adopt one of the more sophisticated strategies for their most important use cases. Having all eight in one toolkit means you can experiment with strategies without changing your infrastructure -- just swap a subcommand or an API parameter.

### Fixed-size chunking

The simplest strategy. Set a target size in bytes or characters, and the chunker walks through the text producing chunks of that size with delimiter-aware boundaries. Fast, predictable, zero configuration. Use it when uniformity matters more than coherence.

```sh
cognigraph-chunker chunk -i document.md -s 1024
```

### Delimiter splitting

Split at every occurrence of a delimiter -- sentence-ending punctuation, newlines, custom patterns. Multi-pattern matching uses an Aho-Corasick automaton compiled once and reused. Combine with token-aware merging to group short sentences into right-sized chunks.

```sh
cognigraph-chunker split -i document.md -d ".?!" --merge --chunk-size 256
```

### Semantic chunking

The first strategy that understands content. Each block gets an embedding vector, and the pipeline computes cosine similarity between adjacent blocks. A Savitzky-Golay filter smooths the similarity curve, and local minima become split points. Where similarity drops, topics change.

```sh
cognigraph-chunker semantic -i document.md --provider ollama
```

### Cognitive chunking

Extends semantic chunking with eight signals instead of one. Beyond embedding similarity, the boundary scorer evaluates entity continuity, discourse continuation, heading context, structural affinity, topic shift, orphan risk, and budget pressure. Blocks are enriched with named entities, pronouns, demonstratives, discourse markers, and heading ancestry before scoring. The result is chunks that preserve not just topics but propositions -- the "who did what to whom" structure that makes text retrievable and answerable. Supports 14 language groups, optional cross-encoder reranking on ambiguous boundaries, LLM-based relation extraction, and graph export.

```sh
cognigraph-chunker cognitive -i document.md --provider ollama
```

### Intent-driven chunking

Optimizes boundaries for predicted user queries rather than topic transitions. An LLM generates hypothetical queries that users might ask about the document. Each candidate chunk is scored by how well its embedding aligns with the predicted intents. Dynamic programming finds the globally optimal partition -- unlike greedy approaches, the DP evaluates downstream consequences of every boundary decision. The result is chunks that align with how readers search, not how authors organize. See [Article 10](10-intent-driven-chunking.md) for full details.

```sh
cognigraph-chunker intent -i document.md -p openai --api-key $OPENAI_API_KEY
```

### Enriched chunking

Produces self-describing chunks. Each chunk carries seven metadata fields extracted in a single LLM call: title, summary, keywords, typed entities, hypothetical questions, semantic keys, and category. The semantic keys create explicit concept links between chunks -- when two chunks discuss the same topic, the LLM assigns matching keys. A recombination step merges chunks sharing the same keys. No embedding provider is needed. See [Article 12](12-enriched-chunking.md) for full details.

```sh
cognigraph-chunker enriched -i document.md --api-key $OPENAI_API_KEY
```

### Topology-aware chunking

Builds a Structured Intermediate Representation (SIR) -- a tree mirroring the document's heading hierarchy with content blocks as leaves and cross-reference edges. Two LLM agents then classify and partition sections: the Inspector labels sections as atomic, splittable, or merge candidates; the Refiner determines optimal split points and merge directions. The result respects heading structure, cross-section dependencies, and structural coupling that flat methods miss. No embedding provider is needed. See [Article 11](11-topology-aware-chunking.md) for full details.

```sh
cognigraph-chunker topo -i document.md --api-key $OPENAI_API_KEY
```

### Adaptive chunking

A meta-router that runs multiple candidate methods on the same document, scores each with five intrinsic quality metrics (Size Compliance, Intrachunk Cohesion, Contextual Coherence, Block Integrity, Reference Completeness), and returns the output from the highest-scoring method. Pre-screening heuristics skip methods unlikely to help for a given document. Use it when you process diverse document types and cannot predict which method works best. See [Article 13](13-adaptive-chunking.md) for full details.

```sh
cognigraph-chunker adaptive -i document.md -p openai --api-key $OPENAI_API_KEY \
  --candidates semantic,cognitive,intent
```

## Four interfaces

The same algorithms are exposed through four interfaces, each suited to a different workflow.

The **CLI** is the fastest way to experiment. Pipe a document through any subcommand (`chunk`, `split`, `semantic`, `cognitive`, `intent`, `enriched`, `topo`, `adaptive`) and see the chunks immediately. Adjust parameters interactively until the output looks right. Export to JSON or JSONL for downstream processing. The CLI reads from files or stdin, so it slots into shell pipelines naturally.

The **REST API** makes the chunker available as a microservice. Start it with `cognigraph-chunker serve`, and any application can send text to the appropriate endpoint and receive structured JSON responses. Bearer token authentication, CORS configuration, SSRF protection on embedding provider URLs, request size limits, and timeouts are all built in.

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

The **Python bindings** bring the Rust performance into Python workflows. The `Chunker` class is an iterator -- loop over chunks in a `for` statement, or call `collect_chunks()` to get them all at once. The `semantic_chunk()` function runs the full semantic pipeline with your choice of embedding provider. Signal processing primitives like `savgol_filter()` and `windowed_cross_similarity()` are exposed individually with NumPy support.

**Docker** wraps the REST API for container-based deployment. The multi-stage Dockerfile produces a minimal image, and the server respects the `PORT` environment variable that platforms like Railway, Render, and Fly.io inject automatically. Set `API_KEY` for authentication and provider credentials via environment variables, and you're running.

## Five embedding providers

Strategies that use embeddings (semantic, cognitive, intent, adaptive) require an embedding model. CogniGraph Chunker supports five providers through a unified trait -- switching between them is a one-parameter change.

**Ollama** is the default. If you're running Ollama locally with a model like `nomic-embed-text`, semantic chunking works with zero configuration. No API keys, no network calls to external services, no cost per request. This is the right choice for development, experimentation, and environments where data can't leave the machine.

**OpenAI** provides access to models like `text-embedding-3-small` via the OpenAI API. The provider handles authentication (API key via flag, environment variable, or `.env.openai` file), batched requests, and error parsing.

**ONNX Runtime** runs sentence-transformer models locally with no Python dependency and no network calls. Point it at a directory containing `model.onnx` and `tokenizer.json`, and it handles tokenization, padding, inference, and mean pooling internally.

**Cloudflare Workers AI** uses Cloudflare's hosted embedding models. Authentication via Cloudflare API token, with optional AI Gateway routing for centralized logging and rate limiting.

**OAuth** covers OpenAI-compatible APIs behind OAuth2 client credentials authentication -- a common pattern in enterprise environments where a corporate API gateway mediates access to LLM services.

Strategies that need only an LLM (enriched, topo) and not embeddings use an OpenAI-compatible completion API directly. The default model is `gpt-4.1-mini`.

## Performance

The core algorithms are written in Rust and operate on byte slices for zero-copy performance. Delimiter search uses SIMD-accelerated `memchr` for one to three delimiters and a lookup table for four or more. Multi-pattern splitting compiles an Aho-Corasick automaton once and reuses it across calls. The Savitzky-Golay filter computes coefficients via matrix operations and applies them through convolution.

For non-semantic strategies, chunking a multi-megabyte document takes microseconds. For embedding-based strategies, embedding computation dominates -- the signal processing and boundary detection add negligible overhead. For LLM-based strategies (intent, enriched, topo), the LLM calls dominate. Adaptive chunking is the most expensive because it runs multiple candidates, but pre-screening keeps the cost practical.

The project includes Criterion benchmarks covering chunking, splitting, merging, signal processing, markdown parsing, and sentence segmentation, so performance regressions are caught before they ship.

## What makes it different

There are other chunking libraries. What sets CogniGraph Chunker apart is the combination of features in a single, cohesive toolkit.

Eight strategies spanning the full complexity spectrum -- from microsecond byte splitting to LLM-powered adaptive selection -- means you can start simple and move up without changing your infrastructure.

Markdown-aware semantic and cognitive chunking parses the AST, keeps structural elements atomic, sentence-splits paragraphs, embeds everything, and finds topic boundaries using signal processing -- all in one pipeline call.

The multi-interface design means you're not locked into one integration pattern. Prototype in the CLI, deploy as a microservice, call from Python, or run in a container -- same algorithms, same parameters, same results.

The embedding provider abstraction means you can start with a free local model and move to a cloud API without changing your chunking code. And it's written in Rust, so it's fast, memory-safe, and compiles to a single binary with no runtime dependencies.

## Getting started

Install from crates.io:

```sh
cargo install cognigraph-chunker
```

Or from PyPI:

```sh
pip install cognigraph-chunker
```

Or build from source:

```sh
git clone https://github.com/gedankrayze/cognigraph-chunker.git
cd cognigraph-chunker
cargo build --release
```

Try it on a document:

```sh
# Fixed-size chunks
cognigraph-chunker chunk -i document.md -s 1024

# Sentence splitting with merge
cognigraph-chunker split -i document.md -d ".?!" --merge --chunk-size 256

# Semantic chunking (requires Ollama or another provider)
cognigraph-chunker semantic -i document.md -f json

# Cognitive chunking with entity preservation
cognigraph-chunker cognitive -i document.md --provider ollama -f json

# Intent-driven chunking optimized for retrieval
cognigraph-chunker intent -i document.md -p openai --api-key $KEY

# Enriched chunking with self-describing metadata
cognigraph-chunker enriched -i document.md --api-key $KEY -f json

# Topology-aware chunking for nested documents
cognigraph-chunker topo -i document.md --api-key $KEY

# Adaptive: let the toolkit pick the best method
cognigraph-chunker adaptive -i document.md -p openai --api-key $KEY
```

The project is open source under the MIT license. The source, documentation, and issue tracker are on GitHub at [gedankrayze/cognigraph-chunker](https://github.com/gedankrayze/cognigraph-chunker).
