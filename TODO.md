# CogniGraph Chunker — Implementation Plan

## Overview

**CogniGraph Chunker** is an actively maintained text chunking toolkit delivered as **CLI**, **REST API**, **Python bindings**, and **Docker container**.

Three chunking strategies:

1. **Fixed-size chunking** — size-based with delimiter-aware boundaries
2. **Delimiter splitting** — split at every delimiter/pattern occurrence
3. **Semantic chunking** — Savitzky-Golay smoothed embedding distances for topic boundary detection

Core algorithms are ported from the `chunk` research project (copied into `src/` as first-party code).

---

## Test Documents

Sample documents for testing are available at `/Users/skitsanos/FTP/Products/CogniGraph/samples/` — 24 domain-specific markdown files covering forensics, healthcare, pharma, manufacturing, and finance.

---

## Phase 1: Project Setup & Core Import

- [x] Copy source files from `chunk` library into `src/core/`: `chunk.rs`, `split.rs`, `merge.rs`, `delim.rs`, `savgol.rs`
- [x] Create `src/core/mod.rs` re-exporting all public types
- [x] Add direct dependencies: `memchr`, `daggrs` (previously owned by `chunk`)
- [x] Add CLI dependencies: `clap` (derive), `serde`, `serde_json`, `tokio`
- [x] Verify core modules compile and existing tests pass (32 tests)
- [x] Define shared types and output module

## Phase 2: Fixed-Size Chunking (CLI)

- [x] Implement `chunk` subcommand using `core::Chunker`
- [x] CLI args: `--input <file|->`, `--size <bytes>`, `--delimiters <chars>`, `--pattern <str>`
- [x] Options: `--prefix`, `--consecutive`, `--forward-fallback`
- [x] Output formats: `--format plain|json|jsonl`
- [x] Support stdin piping

## Phase 3: Delimiter Splitting (CLI)

- [x] Implement `split` subcommand using `core::split_at_delimiters` / `core::split_at_patterns`
- [x] CLI args: `--input <file|->`, `--delimiters <chars>`, `--patterns <str,...>`
- [x] Options: `--include-delim prev|next|none`, `--min-chars <n>`
- [x] Same output formats as Phase 2

## Phase 4: Embedding Providers

- [x] Define `EmbeddingProvider` trait: `async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>>`
- [x] Implement **OpenAI** provider (HTTP via `reqwest`, model configurable, default `text-embedding-3-small`)
- [x] Implement **Ollama** provider (local HTTP API, model configurable)
- [x] Implement **ONNX Runtime** provider (local model via `ort` crate, bundled or user-supplied model path)
- [x] Provider selection via `--provider openai|ollama|onnx` flag
- [x] Config: `--model <name>`, `--api-key <key>` (or env `OPENAI_API_KEY`), `--base-url <url>`

## Phase 5: Semantic Chunking (CLI)

- [x] Implement `semantic` subcommand
- [x] Pipeline: sentence split → embed → cosine distance → S-G smooth → peak detection → merge
- [x] Sentence splitting via `unicode-segmentation`
- [x] Use `core::savgol_filter` for smoothing
- [x] Use `core::windowed_cross_similarity` for distance curve
- [x] Use `core::find_local_minima_interpolated` for boundary detection
- [x] Use `core::filter_split_indices` for percentile-based filtering
- [x] CLI args: `--sim-window`, `--sg-window`, `--poly-order`, `--threshold` (percentile), `--min-distance`
- [x] Same output formats + optional `--emit-distances` for debugging the signal

## Phase 6: Token-Aware Merging

- [x] Add `--merge` flag to all subcommands (chunk, split, semantic) to post-process chunks through `core::merge_splits`
- [x] Args: `--chunk-size <tokens>` for merge budget (default: 512)
- [x] Whitespace-based token counting (fast approximation)

## Phase 7: Polish & UX

- [x] `--verbose` / `--quiet` global flags (verbose shows detail, quiet suppresses all info)
- [x] `--stats` global flag: print chunk count, avg/min/max size in bytes and tokens
- [x] Error messages with actionable hints (file not found, missing API key, missing model path)
- [x] Shell completions generation (`clap_complete`) via `completions` subcommand
- [x] `--version` and `--help` with usage examples on all subcommands

---

## Phase 8: REST API

- [x] Add `serve` subcommand using `axum` + `tokio`
- [x] `POST /api/v1/chunk` — fixed-size chunking
- [x] `POST /api/v1/split` — delimiter splitting
- [x] `POST /api/v1/semantic` — semantic chunking
- [x] `POST /api/v1/merge` — token-aware merging
- [x] `GET /api/v1/health` — health check (no auth)
- [x] Configurable bind address: `--host`, `--port`
- [x] Bearer token auth via `--api-key` or `--no-auth`
- [x] CORS configuration via `--cors-origin`
- [x] SSRF protection: private IP validation on embedding provider URLs
- [x] Request body limit (10 MiB) and timeout (120s)

## Phase 9: Python Bindings

- [x] Add `packages/python/` with PyO3 + maturin setup (Cargo workspace)
- [x] Expose `Chunker` class (iterator, collect_chunks, collect_offsets, reset)
- [x] Expose `split_at_delimiters()`, `split_at_patterns()`, `PatternSplitter` class
- [x] Expose `semantic_chunk()` function with OllamaProvider, OpenAiProvider, OnnxProvider
- [x] Expose `merge_splits()`, `find_merge_indices()`, `MergeResult`
- [x] Expose signal processing: `savgol_filter()`, `windowed_cross_similarity()`, `find_local_minima()`, `filter_split_indices()`
- [x] `SemanticConfig` and `SemanticResult` classes with full field access
- [x] Python test suite: 27 tests (chunker, splitter, merge, signal, semantic)
- [x] NumPy integration: signal functions return `numpy.ndarray`, accept array inputs, `*_array()` properties on result types
- [ ] Publish to PyPI as `cognigraph-chunker`

## Phase 10: Production Readiness

- [x] Security hardening: CORS allowlist, SSRF private IP blocking, error categorization, base URL validation
- [x] Reliable error handling: `anyhow::Result` on provider constructors, no silent `unwrap_or_default`
- [x] Clippy clean: zero warnings across workspace
- [x] Criterion benchmarks: 30+ benchmarks across 6 groups (chunking, splitting, merging, signal, markdown, sentences)
- [x] README documentation: CLI, REST API, Python API, Docker deployment
- [x] Dockerfile: multi-stage build, ONNX Runtime `load-dynamic`, Railway/Render/Fly.io compatible
- [x] `.dockerignore` for minimal build context
- [ ] CI/CD pipeline (GitHub Actions: test, clippy, build Docker image)
- [ ] Publish to crates.io
- [ ] Publish Python package to PyPI

---

## Dependencies Summary

| Crate | Purpose |
|-------|---------|
| `memchr` | SIMD-accelerated byte searching (from chunk core) |
| `daggrs` | Aho-Corasick multi-byte pattern matching (from chunk core) |
| `clap` | CLI argument parsing |
| `clap_complete` | Shell completion generation |
| `tokio` | Async runtime |
| `reqwest` | HTTP client (OpenAI, Ollama) |
| `serde` / `serde_json` | Serialization |
| `ort` | ONNX Runtime for local embeddings (load-dynamic) |
| `tokenizers` | Tokenizer for ONNX models |
| `ndarray` | N-dimensional arrays (used by ort) |
| `unicode-segmentation` | Sentence boundary detection |
| `pulldown-cmark` | Markdown AST parsing |
| `axum` | REST API server |
| `tower-http` | CORS, timeout, request limits middleware |
| `url` | URL parsing for SSRF validation |
| `pyo3` | Python bindings |
| `numpy` | NumPy array interop for Python bindings |
| `criterion` | Benchmarks (dev-dependency) |
