# cognigraph-chunker

Fast text chunking toolkit with fixed-size, delimiter-based, and semantic strategies.

[![Crates.io](https://img.shields.io/crates/v/cognigraph-chunker)](https://crates.io/crates/cognigraph-chunker)
[![docs.rs](https://img.shields.io/docsrs/cognigraph-chunker)](https://docs.rs/cognigraph-chunker)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **Three chunking strategies** -- fixed-size with delimiter-aware boundaries, delimiter/pattern splitting, and embedding-based semantic chunking
- **Three interfaces** -- CLI tool, REST API (Axum), and Python bindings (PyO3)
- **Three embedding providers** -- OpenAI, Ollama, and ONNX Runtime (local)
- **Markdown-aware** -- semantic chunker parses markdown AST to preserve tables, code blocks, headings, and lists as atomic units
- **Merge post-processing** -- combine small chunks into token-budget groups across all strategies
- **Output formats** -- plain text, JSON, and JSONL

## Installation

### CLI (from crates.io)

```sh
cargo install cognigraph-chunker
```

### Python (via maturin)

```sh
pip install cognigraph-chunker
```

### From source

```sh
git clone https://github.com/skitsanos/cognigraph-chunker.git
cd cognigraph-chunker
cargo build --release
```

The binary is at `target/release/cognigraph-chunker`.

## Quick Start

### CLI

```sh
# Fixed-size chunks of 1024 bytes
cognigraph-chunker chunk -i document.md -s 1024

# Split on sentence boundaries, JSON output
cognigraph-chunker split -i document.md -d ".?!" -f json

# Semantic chunking with Ollama
cognigraph-chunker semantic -i document.md
```

### REST API

```sh
# Start the server
cognigraph-chunker serve --api-key my-secret --port 3000

# Chunk text
curl -X POST http://localhost:3000/api/v1/chunk \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret" \
  -d '{"text": "Your long document text here...", "size": 1024}'
```

### Python

```python
from cognigraph_chunker import Chunker

for chunk in Chunker("Your long document text here...", size=1024):
    print(chunk)
```

## CLI Reference

### Global Options

These flags apply to all subcommands:

| Flag | Default | Description |
|------|---------|-------------|
| `--verbose` | off | Show detailed processing information |
| `--quiet` | off | Suppress all informational output (conflicts with `--verbose`) |
| `--stats` | off | Print chunk statistics after output (count, avg/min/max size) |
| `--max-input-size` | 52428800 (50 MiB) | Maximum input size in bytes |

### `chunk` -- Fixed-size chunking

Split text into chunks of a target byte size, with delimiter-aware boundary detection.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | `-` (stdin) | Input file path, or `-` for stdin |
| `--size` | `-s` | 4096 | Target chunk size in bytes |
| `--delimiters` | `-d` | none | Single-byte delimiters to split on (e.g., `"\n.?"`) |
| `--pattern` | `-p` | none | Multi-byte pattern to split on |
| `--prefix` | | off | Put delimiter at start of next chunk instead of end of current |
| `--consecutive` | | off | Split at start of consecutive delimiter runs |
| `--forward-fallback` | | off | Search forward if no boundary found in backward window |
| `--format` | `-f` | plain | Output format: `plain`, `json`, `jsonl` |
| `--merge` | | off | Post-process by merging small chunks to fit a token budget |
| `--chunk-size` | | 512 | Target token count per merged chunk (used with `--merge`) |

**Examples:**

```sh
# 2 KB chunks with newline/period boundaries
cognigraph-chunker chunk -i input.txt -s 2048 -d "\n."

# Prefix mode: delimiters go to the start of the next chunk
cognigraph-chunker chunk -i input.txt -d "\n.?" --prefix

# Pipe from stdin, JSON output
cat file.txt | cognigraph-chunker chunk -s 1024 -f json

# Chunk then merge small pieces into ~256 token groups
cognigraph-chunker chunk -i doc.md -s 512 --merge --chunk-size 256
```

### `split` -- Delimiter splitting

Split text at every occurrence of specified delimiters or patterns.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | `-` (stdin) | Input file path, or `-` for stdin |
| `--delimiters` | `-d` | `\n.?` | Single-byte delimiters to split on |
| `--patterns` | `-p` | none | Multi-byte patterns, comma-separated (e.g., `". ,? ,! "`) |
| `--include-delim` | | `prev` | Where to attach delimiter: `prev`, `next`, or `none` |
| `--min-chars` | | 0 | Minimum characters per segment; shorter segments are merged |
| `--format` | `-f` | plain | Output format: `plain`, `json`, `jsonl` |
| `--merge` | | off | Post-process by merging small chunks to fit a token budget |
| `--chunk-size` | | 512 | Target token count per merged chunk (used with `--merge`) |

**Examples:**

```sh
# Split on sentence-ending punctuation
cognigraph-chunker split -i doc.md -d ".?!"

# Multi-byte patterns, attach delimiters to next segment
cognigraph-chunker split -i doc.md -p ". ,? " --include-delim next

# Minimum 100 chars per segment, JSONL output
cognigraph-chunker split -i doc.md --min-chars 100 -f jsonl

# Split then merge into ~512 token groups
cognigraph-chunker split -i doc.md --merge --chunk-size 512
```

### `semantic` -- Semantic chunking

Split text based on embedding similarity using Savitzky-Golay smoothing to detect topic boundaries.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | `-` (stdin) | Input file path, or `-` for stdin |
| `--provider` | `-p` | `ollama` | Embedding provider: `ollama`, `openai`, `onnx` |
| `--model` | `-m` | provider default | Model name (provider-specific) |
| `--api-key` | | none | API key for OpenAI (also reads env/file) |
| `--base-url` | | none | Base URL override for the embedding API |
| `--model-path` | | none | Path to ONNX model directory (required for `onnx` provider) |
| `--sim-window` | | 3 | Window size for cross-similarity computation (must be odd, >= 3) |
| `--sg-window` | | 11 | Savitzky-Golay smoothing window size (must be odd) |
| `--poly-order` | | 3 | Savitzky-Golay polynomial order |
| `--threshold` | | 0.5 | Percentile threshold for split point filtering (0.0--1.0) |
| `--min-distance` | | 2 | Minimum block gap between split points |
| `--format` | `-f` | plain | Output format: `plain`, `json`, `jsonl` |
| `--emit-distances` | | off | Emit raw and smoothed distance curves to stderr |
| `--no-markdown` | | off | Treat input as plain text instead of markdown |
| `--merge` | | off | Post-process by merging small chunks to fit a token budget |
| `--chunk-size` | | 512 | Target token count per merged chunk (used with `--merge`) |

**Examples:**

```sh
# Semantic chunking with Ollama (default)
cognigraph-chunker semantic -i document.md

# Use OpenAI embeddings, JSON output
cognigraph-chunker semantic -i doc.md -p openai -f json

# Tune signal processing parameters
cognigraph-chunker semantic -i doc.md --sg-window 15 --threshold 0.3

# Export distance curves for debugging
cognigraph-chunker semantic -i doc.md --emit-distances 2>distances.tsv

# Plain text mode (no markdown parsing)
cognigraph-chunker semantic -i doc.md --no-markdown

# Local ONNX model
cognigraph-chunker semantic -i doc.md -p onnx --model-path ./models/all-MiniLM-L6-v2
```

### `serve` -- REST API server

Start an HTTP server exposing all chunking operations.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--host` | | `0.0.0.0` | Host address to bind to |
| `--port` | `-p` | 3000 | Port to listen on |
| `--api-key` | | none | API key for bearer token authentication |
| `--no-auth` | | off | Run without authentication (insecure) |
| `--allow-private-urls` | | off | Allow embedding provider base URLs pointing to private/loopback IPs |
| `--cors-origin` | | none | Allowed CORS origins (repeatable; omit for same-origin only) |

**Examples:**

```sh
# Start with authentication
cognigraph-chunker serve --api-key my-secret

# Custom port with CORS
cognigraph-chunker serve --port 8080 --api-key my-secret --cors-origin https://example.com

# Development mode (no auth, allow private URLs)
cognigraph-chunker serve --no-auth --allow-private-urls
```

### `completions` -- Shell completions

```sh
cognigraph-chunker completions bash > ~/.bash_completions/cognigraph-chunker
cognigraph-chunker completions zsh > ~/.zfunc/_cognigraph-chunker
cognigraph-chunker completions fish > ~/.config/fish/completions/cognigraph-chunker.fish
```

## REST API Reference

All endpoints are under `/api/v1`. When `--api-key` is configured, include `Authorization: Bearer <key>` in all requests (except health).

Request body limit: 10 MiB. Request timeout: 120 seconds.

### `GET /api/v1/health`

Health check. Always open (no auth required).

**Response:**
```json
{ "status": "ok" }
```

### `POST /api/v1/chunk`

Fixed-size chunking.

**Request body:**
```json
{
  "text": "string (required)",
  "size": 4096,
  "delimiters": "\n.",
  "pattern": null,
  "prefix": false,
  "consecutive": false,
  "forward_fallback": false,
  "merge": false,
  "chunk_size": 512
}
```

**Response:**
```json
{
  "chunks": [
    { "index": 0, "text": "...", "offset": 0, "length": 1024 },
    { "index": 1, "text": "...", "offset": 1024, "length": 980 }
  ],
  "count": 2
}
```

### `POST /api/v1/split`

Delimiter/pattern splitting.

**Request body:**
```json
{
  "text": "string (required)",
  "delimiters": ".?!",
  "patterns": null,
  "include_delim": "prev",
  "min_chars": 0,
  "merge": false,
  "chunk_size": 512
}
```

- `include_delim`: `"prev"` (default), `"next"`, or `"none"`

**Response:** Same structure as `/api/v1/chunk`.

### `POST /api/v1/semantic`

Semantic chunking with embeddings.

**Request body:**
```json
{
  "text": "string (required)",
  "provider": "ollama",
  "model": null,
  "api_key": null,
  "base_url": null,
  "model_path": null,
  "sim_window": 3,
  "sg_window": 11,
  "poly_order": 3,
  "threshold": 0.5,
  "min_distance": 2,
  "no_markdown": false,
  "merge": false,
  "chunk_size": 512
}
```

- `provider`: `"ollama"` (default), `"openai"`, or `"onnx"`
- `model_path` is required when `provider` is `"onnx"`
- `base_url` is validated against SSRF (private IPs rejected unless `--allow-private-urls` is set)

**Response:** Same structure as `/api/v1/chunk`.

### `POST /api/v1/merge`

Merge pre-split chunks into token-budget groups.

**Request body:**
```json
{
  "chunks": ["chunk one", "chunk two", "chunk three"],
  "chunk_size": 512
}
```

**Response:**
```json
{
  "chunks": [
    { "index": 0, "text": "chunk one chunk two", "offset": 0, "length": 19 }
  ],
  "count": 1,
  "token_counts": [4]
}
```

## Python API Reference

### `Chunker`

Fixed-size chunking. Iterable.

```python
from cognigraph_chunker import Chunker

chunker = Chunker(
    text,                       # str, required
    size=4096,                  # target chunk size in bytes
    delimiters=None,            # bytes, single-byte delimiters
    pattern=None,               # bytes, multi-byte pattern
    prefix=False,               # delimiter at start of next chunk
    consecutive=False,          # split at consecutive delimiter runs
    forward_fallback=False,     # search forward if no backward boundary
)

# Iterate
for chunk in chunker:
    print(chunk)

# Or collect all at once
chunker.reset()
chunks = chunker.collect_chunks()     # list[str]
offsets = chunker.collect_offsets()    # list[tuple[int, int]]
```

### `split_at_delimiters` / `split_at_patterns`

Delimiter and pattern splitting functions.

```python
from cognigraph_chunker import split_at_delimiters, split_at_patterns

# Split on single-byte delimiters
offsets = split_at_delimiters(
    text,                       # str
    delimiters,                 # bytes (e.g., b".?!")
    include_delim="prev",       # "prev", "next", or "none"
    min_chars=0,                # minimum chars per segment
)
# Returns list[tuple[int, int]] -- (start, end) byte offsets

# Split on multi-byte patterns
offsets = split_at_patterns(
    text,
    patterns,                   # list[bytes] (e.g., [b". ", b"? "])
    include_delim="prev",
    min_chars=0,
)
```

### `PatternSplitter`

Reusable pattern splitter (compiles patterns once).

```python
from cognigraph_chunker import PatternSplitter

splitter = PatternSplitter(patterns=[b". ", b"? ", b"! "])
offsets = splitter.split(text, include_delim="prev", min_chars=0)
```

### `merge_splits` / `find_merge_indices`

Merge small chunks into token-budget groups.

```python
from cognigraph_chunker import merge_splits, find_merge_indices

result = merge_splits(
    splits=["chunk one", "chunk two", "chunk three"],
    token_counts=[2, 2, 2],
    chunk_size=5,
)
print(result.merged)         # list[str]
print(result.token_counts)   # list[int]

# Just get merge boundary indices
indices = find_merge_indices(token_counts=[2, 2, 2], chunk_size=5)
```

### Semantic Chunking

```python
from cognigraph_chunker import (
    OllamaProvider, OpenAiProvider, OnnxProvider,
    SemanticConfig, semantic_chunk,
)

# Choose a provider
provider = OllamaProvider(model="nomic-embed-text")
# provider = OpenAiProvider("sk-...", model="text-embedding-3-small")
# provider = OnnxProvider("/path/to/model-dir")

config = SemanticConfig(
    sim_window=3,         # cross-similarity window (odd, >= 3)
    sg_window=11,         # Savitzky-Golay window (odd)
    poly_order=3,         # polynomial order
    threshold=0.5,        # percentile threshold (0.0-1.0)
    min_distance=2,       # minimum block gap between splits
    max_blocks=10000,     # maximum blocks to process
)

result = semantic_chunk(text, provider, config, markdown=True)
for chunk_text, offset in result.chunks:
    print(f"[offset={offset}] {chunk_text[:80]}...")

# Access signal data
print(result.similarities)              # list[float] -- raw distance curve
print(result.smoothed)                  # list[float] -- smoothed curve
print(result.split_indices.indices)     # list[int] -- split point indices
print(result.split_indices.values)      # list[float] -- values at split points
```

### Signal Processing Functions

Low-level signal processing primitives used by the semantic chunker.

```python
from cognigraph_chunker import (
    savgol_filter,
    windowed_cross_similarity,
    find_local_minima,
    filter_split_indices,
)

# Savitzky-Golay filter
smoothed = savgol_filter(data, window_length=11, poly_order=3, deriv=0)

# Cross-similarity between embedding windows
distances = windowed_cross_similarity(embeddings, n=num_blocks, d=dim, window_size=3)

# Find local minima in the distance curve
result = find_local_minima(data, window_size=11, poly_order=3, tolerance=0.1)
print(result.indices, result.values)

# Filter split indices by threshold and minimum distance
filtered = filter_split_indices(indices, values, threshold=0.5, min_distance=2)
print(filtered.indices, filtered.values)
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (used by `openai` provider) |
| `OLLAMA_HOST` | Ollama server URL (default: `http://localhost:11434`) |

### `.env.openai` File

The OpenAI provider reads API keys from a `.env.openai` file in the working directory:

```
OPENAI_API_KEY=sk-...
```

Key resolution order: `--api-key` flag / `api_key` field > `OPENAI_API_KEY` env var > `.env.openai` file.

### Embedding Provider Setup

**Ollama** (default) -- Install [Ollama](https://ollama.ai) and pull a model:

```sh
ollama pull nomic-embed-text
```

**OpenAI** -- Set your API key via any of the methods above. Default model: `text-embedding-3-small`.

**ONNX** -- Download a model directory containing `model.onnx` and `tokenizer.json`. Compatible with Hugging Face ONNX exports (e.g., `all-MiniLM-L6-v2`).

```sh
cognigraph-chunker semantic -i doc.md -p onnx --model-path ./models/all-MiniLM-L6-v2
```

## Architecture

```
cognigraph-chunker/
  src/
    lib.rs              # Library root (public API)
    main.rs             # CLI entry point
    core/               # Core algorithms (chunk, split, merge, signal processing)
    embeddings/         # Embedding providers (OpenAI, Ollama, ONNX)
    semantic/           # Semantic chunking pipeline
    api/                # REST API (Axum handlers, types, middleware)
    cli/                # CLI subcommands and options
    output/             # Output formatting (plain, json, jsonl)
  packages/
    python/             # Python bindings (PyO3 + maturin)
```

The core algorithms operate on byte slices for zero-copy performance. The semantic pipeline splits text into blocks (markdown-aware or sentence-based), computes embeddings, calculates cross-similarity distances, applies Savitzky-Golay smoothing, and detects topic boundaries at local minima.

## License

MIT
