# CogniGraph Chunker: Putting It All Together

The previous five articles explored the concepts behind text chunking: why it matters for AI retrieval, how fixed-size and delimiter strategies handle structural boundaries, how semantic analysis detects topic shifts using signal processing, how markdown-aware parsing preserves document structure, and how token-aware merging normalizes chunk sizes. CogniGraph Chunker is the toolkit that brings all of these ideas into a single, production-ready package.

## One library, three strategies

CogniGraph Chunker implements all three chunking strategies from a shared core. Fixed-size chunking with delimiter-aware boundaries, delimiter and pattern splitting with Aho-Corasick multi-pattern matching, and semantic chunking with embedding-based topic boundary detection all live in the same codebase, share the same output format, and can all be followed by token-aware merging as a post-processing step.

This matters in practice because most teams start with fixed-size chunking, realize they need better boundary detection, move to delimiter splitting, and eventually adopt semantic chunking for their most important use cases. Having all three in one toolkit means you can experiment with strategies without changing your infrastructure — just swap a subcommand or an API parameter.

## Four interfaces

The same algorithms are exposed through four interfaces, each suited to a different workflow.

The **CLI** is the fastest way to experiment. Pipe a document through `cognigraph-chunker semantic -i doc.md` and see the chunks immediately. Adjust parameters like `--threshold` and `--sg-window` interactively until the output looks right. Export to JSON or JSONL for downstream processing. The CLI reads from files or stdin, so it slots into shell pipelines naturally.

The **REST API** makes the chunker available as a microservice. Start it with `cognigraph-chunker serve`, and any application can send text to `/api/v1/chunk`, `/api/v1/split`, `/api/v1/semantic`, or `/api/v1/merge` and receive structured JSON responses. Bearer token authentication, CORS configuration, SSRF protection on embedding provider URLs, request size limits, and timeouts are all built in. Deploy it alongside your ingestion pipeline and call it from any language.

The **Python bindings** bring the Rust performance into Python workflows. The `Chunker` class is an iterator — loop over chunks in a `for` statement, or call `collect_chunks()` to get them all at once. The `semantic_chunk()` function runs the full pipeline with your choice of embedding provider. Signal processing primitives like `savgol_filter()` and `windowed_cross_similarity()` are exposed individually for custom pipelines, with NumPy array support for interoperability with the scientific Python ecosystem.

**Docker** wraps the REST API for container-based deployment. The multi-stage Dockerfile produces a minimal image, and the server respects the `PORT` environment variable that platforms like Railway, Render, and Fly.io inject automatically. Set `API_KEY` for authentication and `OPENAI_API_KEY` if you want the OpenAI embedding provider available, and you're running.

## Five embedding providers

Semantic chunking requires an embedding model, and different environments call for different providers. CogniGraph Chunker supports five out of the box.

**Ollama** is the default. If you're running Ollama locally with a model like `nomic-embed-text`, semantic chunking works with zero configuration. No API keys, no network calls to external services, no cost per request. This is the right choice for development, experimentation, and environments where data can't leave the machine.

**OpenAI** provides access to models like `text-embedding-3-small` via the OpenAI API. The provider handles authentication (API key via flag, environment variable, or `.env.openai` file), batched requests, and error parsing. Use this when you want high-quality embeddings and are comfortable with the API cost and latency.

**ONNX Runtime** runs sentence-transformer models locally with no Python dependency and no network calls. Point it at a directory containing `model.onnx` and `tokenizer.json` (compatible with HuggingFace ONNX exports like `all-MiniLM-L6-v2`), and it handles tokenization, padding, inference, and mean pooling internally. This gives you local inference with the flexibility to choose your own model.

**Cloudflare Workers AI** uses Cloudflare's hosted embedding models like `@cf/baai/bge-m3` and `@cf/qwen/qwen3-embedding-0.6b`. Authentication is handled via a Cloudflare API token, which is verified at startup before any embedding requests are made. Credentials can be provided through CLI flags, environment variables, or a `.env.cloudflare` file. For teams already using Cloudflare's infrastructure, the provider also supports routing requests through an AI Gateway for centralized logging, rate limiting, and analytics — just set the gateway name and both the gateway authentication and provider authentication are handled automatically.

**OAuth** covers OpenAI-compatible APIs that sit behind OAuth2 client credentials authentication — a common pattern in enterprise environments where a corporate API gateway mediates access to LLM services. The provider acquires a bearer token automatically, caches it, and refreshes it before expiry. Credentials come from CLI flags, environment variables, or a `.env.oauth` file. For endpoints behind corporate proxies with custom certificate authorities, TLS verification can be disabled with a flag. This means teams running behind an API gateway like Azure API Management or similar can use the same semantic chunking pipeline without any custom integration work.

All five providers implement the same trait, so switching between them is a one-parameter change. The chunking pipeline doesn't know or care which provider is running — it just receives embedding vectors and processes them the same way.

## Performance

The core algorithms are written in Rust and operate on byte slices for zero-copy performance. Delimiter search uses SIMD-accelerated `memchr` for one to three delimiters and a lookup table for four or more. Multi-pattern splitting compiles an Aho-Corasick automaton once and reuses it across calls. The Savitzky-Golay filter computes coefficients via matrix operations and applies them through convolution.

For non-semantic strategies, chunking a multi-megabyte document takes microseconds. The semantic pipeline is dominated by embedding computation — the signal processing and boundary detection add negligible overhead on top of the provider latency. With a local ONNX model, end-to-end semantic chunking of a 100-page document completes in seconds.

The project includes Criterion benchmarks covering chunking, splitting, merging, signal processing, markdown parsing, and sentence segmentation, so performance regressions are caught before they ship.

## What makes it different

There are other chunking libraries. What sets CogniGraph Chunker apart is the combination of features in a single, cohesive toolkit.

Markdown-aware semantic chunking is the headline feature. Most chunkers treat markdown as plain text or offer only fixed-size splitting. CogniGraph Chunker parses the AST, keeps structural elements atomic, sentence-splits paragraphs, embeds everything, and finds topic boundaries using signal processing — all in one pipeline call.

The multi-interface design means you're not locked into one integration pattern. Prototype in the CLI, deploy as a microservice, call from Python, or run in a container — same algorithms, same parameters, same results.

The embedding provider abstraction means you can start with a free local model and move to a cloud API without changing your chunking code. Or run ONNX in production for zero-dependency local inference.

And it's written in Rust, so it's fast, memory-safe, and compiles to a single binary with no runtime dependencies (beyond ONNX Runtime if you use the ONNX provider).

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
```

The project is open source under the MIT license. The source, documentation, and issue tracker are on GitHub at [gedankrayze/cognigraph-chunker](https://github.com/gedankrayze/cognigraph-chunker).
