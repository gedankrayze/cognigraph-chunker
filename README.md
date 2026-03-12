# cognigraph-chunker

Fast text chunking toolkit with fixed-size, delimiter-based, semantic, and cognition-aware strategies.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **Four chunking strategies** -- fixed-size with delimiter-aware boundaries, delimiter/pattern splitting, embedding-based semantic chunking, and cognition-aware chunking with multi-signal boundary scoring
- **Cognition-aware chunking** -- 8-signal boundary scoring (semantic similarity, entity continuity, discourse continuation, heading context, structural affinity, topic shift, orphan risk, budget pressure), proposition-aware healing, cross-chunk entity tracking, and automatic quality metrics
- **Multilingual** -- automatic language detection across 70+ languages with language-specific enrichment for 14 language groups (English, German, French, Spanish, Portuguese, Italian, Dutch, Russian, Turkish, Polish, Chinese, Japanese, Korean, Arabic)
- **Four interfaces** -- CLI tool, REST API (Axum), Python bindings (PyO3), and Docker
- **Five embedding providers** -- OpenAI, Ollama, ONNX Runtime (local), Cloudflare Workers AI, and OAuth-authenticated OpenAI-compatible endpoints
- **Markdown-aware** -- parses markdown AST to preserve tables, code blocks, headings, and lists as atomic units
- **Optional LLM enrichment** -- relation triple extraction and chunk synopsis generation via OpenAI-compatible API (post-assembly, no LLM needed for core chunking)
- **Graph export** -- output chunks as nodes with adjacency and shared-entity edges, ready for graph databases
- **Ambiguous boundary refinement** -- optional cross-encoder reranking for precision improvement on uncertain boundaries (NVIDIA NIM, Cohere, Cloudflare Workers AI, OAuth-authenticated endpoints, or local ONNX)
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
git clone https://github.com/gedankrayze/cognigraph-chunker.git
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

# Cognition-aware chunking (preserves entity chains, discourse structure, heading context)
cognigraph-chunker cognitive -i document.md -f json

# Cognitive chunking with graph export
cognigraph-chunker cognitive -i document.md --graph

# Cognitive chunking with LLM-based relation extraction
cognigraph-chunker cognitive -i document.md --relations -f json
```

### REST API

```sh
# Start the server
cognigraph-chunker serve --api-key my-secret --port 3000

# Fixed-size chunking
curl -X POST http://localhost:3000/api/v1/chunk \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret" \
  -d '{"text": "Your long document text here...", "size": 1024}'

# Cognitive chunking
curl -X POST http://localhost:3000/api/v1/cognitive \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret" \
  -d '{"text": "Your long document text here...", "provider": "openai"}'
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
| `--provider` | `-p` | `ollama` | Embedding provider: `ollama`, `openai`, `onnx`, `cloudflare`, `oauth` |
| `--model` | `-m` | provider default | Model name (provider-specific) |
| `--api-key` | | none | API key for OpenAI (also reads env/file) |
| `--base-url` | | none | Base URL override for the embedding API |
| `--model-path` | | none | Path to ONNX model directory (required for `onnx` provider) |
| `--cf-auth-token` | | none | Cloudflare auth token (also reads env/`.env.cloudflare`) |
| `--cf-account-id` | | none | Cloudflare account ID (also reads env/`.env.cloudflare`) |
| `--cf-ai-gateway` | | none | Cloudflare AI Gateway name (optional; routes through gateway) |
| `--oauth-token-url` | | none | OAuth token endpoint URL (also reads env/`.env.oauth`) |
| `--oauth-client-id` | | none | OAuth client ID (also reads env/`.env.oauth`) |
| `--oauth-client-secret` | | none | OAuth client secret (also reads env/`.env.oauth`) |
| `--oauth-scope` | | none | OAuth scope (optional; also reads env/`.env.oauth`) |
| `--oauth-base-url` | | none | Base URL for the OpenAI-compatible API (also reads env/`.env.oauth`) |
| `--danger-accept-invalid-certs` | | off | Accept invalid TLS certificates (for corporate proxies) |
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

# Cloudflare Workers AI (reads credentials from .env.cloudflare)
cognigraph-chunker semantic -i doc.md -p cloudflare

# Cloudflare via AI Gateway
cognigraph-chunker semantic -i doc.md -p cloudflare --cf-ai-gateway my-gateway

# OAuth-authenticated endpoint (reads credentials from .env.oauth)
cognigraph-chunker semantic -i doc.md -p oauth

# OAuth with custom CA (corporate proxy)
cognigraph-chunker semantic -i doc.md -p oauth --danger-accept-invalid-certs
```

### `cognitive` -- Cognition-aware chunking

Split text using multi-signal boundary scoring that preserves entity chains, discourse structure, and heading context. Extends semantic chunking with eight cognitive signals and proposition-aware healing.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | `-` (stdin) | Input file path, or `-` for stdin |
| `--provider` | `-p` | `ollama` | Embedding provider: `ollama`, `openai`, `onnx`, `cloudflare`, `oauth` |
| `--model` | `-m` | provider default | Model name (provider-specific) |
| `--api-key` | | none | API key for OpenAI (also reads env/file) |
| `--base-url` | | none | Base URL override for the embedding API |
| `--model-path` | | none | Path to ONNX model directory (required for `onnx` provider) |
| `--cf-auth-token` | | none | Cloudflare auth token (also reads env/`.env.cloudflare`) |
| `--cf-account-id` | | none | Cloudflare account ID (also reads env/`.env.cloudflare`) |
| `--cf-ai-gateway` | | none | Cloudflare AI Gateway name (optional) |
| `--oauth-token-url` | | none | OAuth token endpoint URL (also reads env/`.env.oauth`) |
| `--oauth-client-id` | | none | OAuth client ID (also reads env/`.env.oauth`) |
| `--oauth-client-secret` | | none | OAuth client secret (also reads env/`.env.oauth`) |
| `--oauth-scope` | | none | OAuth scope (optional) |
| `--oauth-base-url` | | none | Base URL for the OpenAI-compatible API (also reads env/`.env.oauth`) |
| `--danger-accept-invalid-certs` | | off | Accept invalid TLS certificates (for corporate proxies) |
| `--soft-budget` | | 512 | Soft token budget per chunk (assembly prefers to stay under this) |
| `--hard-budget` | | 768 | Hard token ceiling per chunk (never exceeded unless a single block is larger) |
| `--sim-window` | | 3 | Window size for cross-similarity computation (must be odd, >= 3) |
| `--sg-window` | | 11 | Savitzky-Golay smoothing window size (must be odd) |
| `--poly-order` | | 3 | Savitzky-Golay polynomial order |
| `--language` | | auto-detect | Language override (`en`, `de`, `fr`, `es`, `pt`, `it`, `nl`, `ru`, `zh`, `ja`, `ko`, `ar`, `tr`, `pl`) or `auto` |
| `--reranker` | | none | Reranker for ambiguous boundary refinement: `nvidia`, `cohere`, `cloudflare`, `oauth`, `onnx:<path>`, or a bare path |
| `--relations` | | off | Extract relation triples via LLM (requires OpenAI API key) |
| `--synopsis` | | off | Generate LLM-based synopsis for each chunk (requires OpenAI API key) |
| `--graph` | | off | Output as graph structure (nodes + edges) instead of flat chunks |
| `--emit-signals` | | off | Emit full boundary signal diagnostics to stderr |
| `--no-markdown` | | off | Treat input as plain text instead of markdown |
| `--format` | `-f` | plain | Output format: `plain`, `json`, `jsonl` |

**Examples:**

```sh
# Cognitive chunking with Ollama (default)
cognigraph-chunker cognitive -i document.md

# Use OpenAI embeddings, JSON output
cognigraph-chunker cognitive -i doc.md -p openai -f json

# Custom token budgets
cognigraph-chunker cognitive -i doc.md --soft-budget 256 --hard-budget 512

# With NVIDIA NIM reranker (reads .env.nvidia for credentials)
cognigraph-chunker cognitive -i doc.md --reranker nvidia

# With Cohere reranker (reads .env.cohere for credentials)
cognigraph-chunker cognitive -i doc.md --reranker cohere

# With Cloudflare Workers AI reranker (reads .env.cloudflare for credentials)
cognigraph-chunker cognitive -i doc.md --reranker cloudflare

# With OAuth-authenticated reranker (reads .env.oauth for credentials)
cognigraph-chunker cognitive -i doc.md --reranker oauth

# With local ONNX cross-encoder reranker
cognigraph-chunker cognitive -i doc.md --reranker onnx:./models/ms-marco-MiniLM-L-6-v2

# OpenAI embeddings + NVIDIA reranking (best quality/speed combo)
cognigraph-chunker cognitive -i doc.md -p openai --reranker nvidia

# Extract relation triples via LLM
cognigraph-chunker cognitive -i doc.md --relations -f json

# Graph export (nodes + edges with entity links)
cognigraph-chunker cognitive -i doc.md --graph

# Generate chunk synopses via LLM
cognigraph-chunker cognitive -i doc.md --synopsis -f json

# Force language (skip auto-detection)
cognigraph-chunker cognitive -i doc.md --language de

# Full diagnostics with stats
cognigraph-chunker cognitive -i doc.md --emit-signals --stats -f json

# Plain text mode (no markdown parsing)
cognigraph-chunker cognitive -i doc.md --no-markdown
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
  "cf_auth_token": null,
  "cf_account_id": null,
  "cf_ai_gateway": null,
  "oauth_token_url": null,
  "oauth_client_id": null,
  "oauth_client_secret": null,
  "oauth_scope": null,
  "oauth_base_url": null,
  "danger_accept_invalid_certs": false,
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

- `provider`: `"ollama"` (default), `"openai"`, `"onnx"`, `"cloudflare"`, or `"oauth"`
- `model_path` is required when `provider` is `"onnx"`
- `cf_auth_token` and `cf_account_id` are required for `"cloudflare"` (also reads env vars or `.env.cloudflare`)
- `cf_ai_gateway` optionally routes requests through a Cloudflare AI Gateway
- `oauth_*` fields are required for `"oauth"` (also reads env vars or `.env.oauth`)
- `danger_accept_invalid_certs` disables TLS verification for corporate proxies with custom CAs
- `base_url` is validated against SSRF (private IPs rejected unless `--allow-private-urls` is set)

**Response:** Same structure as `/api/v1/chunk`.

### `POST /api/v1/cognitive`

Cognition-aware chunking with multi-signal boundary scoring.

**Request body:**
```json
{
  "text": "string (required)",
  "provider": "ollama",
  "model": null,
  "api_key": null,
  "base_url": null,
  "model_path": null,
  "cf_auth_token": null,
  "cf_account_id": null,
  "cf_ai_gateway": null,
  "oauth_token_url": null,
  "oauth_client_id": null,
  "oauth_client_secret": null,
  "oauth_scope": null,
  "oauth_base_url": null,
  "danger_accept_invalid_certs": false,
  "soft_budget": 512,
  "hard_budget": 768,
  "sim_window": 3,
  "sg_window": 11,
  "poly_order": 3,
  "no_markdown": false,
  "emit_signals": false,
  "relations": false,
  "language": null,
  "reranker_path": null,
  "graph": false
}
```

- `soft_budget` / `hard_budget`: token budget controls (assembly prefers soft, never exceeds hard)
- `language`: override auto-detection (`"en"`, `"de"`, `"fr"`, `"es"`, `"pt"`, `"it"`, `"nl"`, `"ru"`, `"zh"`, `"ja"`, `"ko"`, `"ar"`, `"tr"`, `"pl"`, `"auto"` for explicit auto-detect)
- `reranker_path`: reranker provider for ambiguous boundary refinement — `"nvidia"`, `"cohere"`, `"cloudflare"`, `"oauth"`, `"onnx:<path>"`, or a bare path to an ONNX model directory
- `relations`: extract relation triples via LLM (requires OpenAI API key)
- `graph`: return graph-shaped output (nodes + edges) instead of flat chunks
- All embedding provider fields work the same as `/api/v1/semantic`

**Response (flat mode):**
```json
{
  "chunks": [
    {
      "index": 0,
      "text": "...",
      "offset_start": 0,
      "offset_end": 1024,
      "length": 1024,
      "heading_path": ["Architecture", "Scoring"],
      "dominant_entities": ["CogniGraph", "boundary scorer"],
      "token_estimate": 256,
      "continuity_confidence": 0.85,
      "prev_chunk": null,
      "next_chunk": 1
    }
  ],
  "count": 5,
  "block_count": 23,
  "evaluation": {
    "entity_orphan_rate": 0.0,
    "pronoun_boundary_rate": 0.0,
    "heading_attachment_rate": 1.0,
    "discourse_break_rate": 0.0,
    "triple_severance_rate": 0.0
  },
  "shared_entities": {
    "cognigraph": [0, 2, 4],
    "boundary scorer": [1, 3]
  }
}
```

**Response (graph mode, `"graph": true`):**
```json
{
  "nodes": [
    { "id": 0, "text": "...", "heading_path": [...], "entities": [...], "token_estimate": 256 }
  ],
  "edges": [
    { "source": 0, "target": 1, "edge_type": "adjacency" },
    { "source": 0, "target": 3, "edge_type": "entity", "entity": "CogniGraph" }
  ],
  "metadata": { "node_count": 5, "edge_count": 12 }
}
```

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
| `CLOUDFLARE_AUTH_TOKEN` | Cloudflare API token (used by `cloudflare` provider) |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID (used by `cloudflare` provider) |
| `CLOUDFLARE_AI_GATEWAY` | Cloudflare AI Gateway name (optional; routes through gateway) |
| `OAUTH_TOKEN_URL` | OAuth token endpoint URL (used by `oauth` provider) |
| `OAUTH_CLIENT_ID` | OAuth client ID (used by `oauth` provider) |
| `OAUTH_CLIENT_SECRET` | OAuth client secret (used by `oauth` provider) |
| `OAUTH_SCOPE` | OAuth scope (optional) |
| `OAUTH_BASE_URL` | Base URL for the OpenAI-compatible API (used by `oauth` provider) |
| `OAUTH_MODEL` | Model name (used by `oauth` provider) |
| `COGNIGRAPH_LLM_MODEL` | LLM model for relation extraction and synopsis (default: `gpt-4.1-mini`) |
| `NVIDIA_API_KEY` | NVIDIA NIM API key (used by `nvidia` reranker) |
| `NVIDIA_RERANK_MODEL` | NVIDIA reranker model (default: `nv-rerank-qa-mistral-4b:1`) |
| `NVIDIA_RERANK_BASE_URL` | NVIDIA reranker base URL (default: `https://ai.api.nvidia.com/v1`) |
| `COHERE_API_KEY` | Cohere API key (used by `cohere` reranker) |
| `COHERE_RERANK_MODEL` | Cohere reranker model (default: `rerank-v3.5`) |
| `COHERE_RERANK_BASE_URL` | Cohere reranker base URL (default: `https://api.cohere.com/v2`) |
| `CLOUDFLARE_RERANK_MODEL` | Cloudflare reranker model (default: `@cf/baai/bge-reranker-base`) |
| `OAUTH_RERANK_PATH` | Rerank endpoint path appended to `OAUTH_BASE_URL` (default: `/rerank`) |
| `OAUTH_RERANK_MODEL` | Model name for OAuth reranker |

### `.env.openai` File

The OpenAI provider reads API keys from a `.env.openai` file in the working directory:

```
OPENAI_API_KEY=sk-...
```

Key resolution order: `--api-key` flag / `api_key` field > `OPENAI_API_KEY` env var > `.env.openai` file.

### `.env.cloudflare` File

The Cloudflare provider reads credentials from a `.env.cloudflare` file in the working directory. These credentials are shared between the embedding provider and the `cloudflare` reranker:

```
CLOUDFLARE_AUTH_TOKEN=your-token
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_AI_GATEWAY=your-gateway-name
CLOUDFLARE_RERANK_MODEL=@cf/baai/bge-reranker-base
```

Key resolution order: CLI flags / request fields > environment variables > `.env.cloudflare` file.

### `.env.oauth` File

The OAuth provider reads credentials from a `.env.oauth` file in the working directory. These credentials are shared between the embedding provider and the `oauth` reranker:

```
OAUTH_TOKEN_URL=https://auth.example.com/api/oauth/token
OAUTH_CLIENT_ID=your-client-id
OAUTH_CLIENT_SECRET=your-client-secret
OAUTH_SCOPE=embeddings
OAUTH_BASE_URL=https://api.example.com/llm-api
OAUTH_MODEL=text-embedding-3-small
OAUTH_RERANK_PATH=/rerank
OAUTH_RERANK_MODEL=rerank-model-name
```

The `OAUTH_RERANK_PATH` is appended to `OAUTH_BASE_URL` to form the rerank endpoint (default: `/rerank`). This accommodates corporate API gateways that expose reranking at non-standard paths.

Key resolution order: CLI flags / request fields > environment variables > `.env.oauth` file.

### `.env.nvidia` File

The NVIDIA reranker reads credentials from a `.env.nvidia` file in the working directory:

```
NVIDIA_API_KEY=nvapi-...
NVIDIA_RERANK_MODEL=nvidia/llama-nemotron-rerank-1b-v2
NVIDIA_RERANK_BASE_URL=https://ai.api.nvidia.com/v1
```

Available models include `nvidia/llama-nemotron-rerank-1b-v2` (recommended — fast, high quality), `nv-rerank-qa-mistral-4b:1`, and `nvidia/rerank-qa-mistral-4b`. The endpoint path is derived automatically from the model name.

Key resolution order: environment variables > `.env.nvidia` file.

### `.env.cohere` File

The Cohere reranker reads credentials from a `.env.cohere` file in the working directory:

```
COHERE_API_KEY=your-key
COHERE_RERANK_MODEL=rerank-v3.5
```

Available models: `rerank-v3.5`, `rerank-english-v3.0`, `rerank-multilingual-v3.0`.

Key resolution order: environment variables > `.env.cohere` file.

### Embedding Provider Setup

**Ollama** (default) -- Install [Ollama](https://ollama.ai) and pull a model:

```sh
ollama pull nomic-embed-text
```

**OpenAI** -- Set your API key via any of the methods above. Default model: `text-embedding-3-small`.

**ONNX** -- Download a model directory containing `model.onnx` and `tokenizer.json`. Compatible with Hugging Face ONNX exports (e.g., `all-MiniLM-L6-v2`).

ONNX Runtime must be available at runtime when using ONNX providers. Install it first (for example, `brew install onnxruntime`), and set `ORT_DYLIB_PATH` only when needed.

```sh
cognigraph-chunker semantic -i doc.md -p onnx --model-path ./models/all-MiniLM-L6-v2
```

**Cloudflare Workers AI** -- Uses Cloudflare's hosted embedding models (e.g., `@cf/baai/bge-m3`, `@cf/qwen/qwen3-embedding-0.6b`). Set credentials via environment variables or `.env.cloudflare` file. The token is verified at startup. Optionally route requests through an AI Gateway for logging and rate limiting.

```sh
cognigraph-chunker semantic -i doc.md -p cloudflare
cognigraph-chunker semantic -i doc.md -p cloudflare --cf-ai-gateway my-gateway -m @cf/qwen/qwen3-embedding-0.6b
```

**OAuth** -- For OpenAI-compatible APIs behind OAuth2 client credentials authentication (e.g., corporate API gateways). Set credentials via environment variables or `.env.oauth` file. The token is acquired automatically, cached, and refreshed before expiry. Use `--danger-accept-invalid-certs` for endpoints behind corporate proxies with custom CAs.

```sh
cognigraph-chunker semantic -i doc.md -p oauth
cognigraph-chunker semantic -i doc.md -p oauth --danger-accept-invalid-certs
```

## Docker

### Build

```sh
docker build -t cognigraph-chunker .
```

### Run

```sh
# With API key authentication
docker run -p 3000:3000 -e API_KEY=my-secret cognigraph-chunker

# Without authentication (development)
docker run -p 3000:3000 -e NO_AUTH=1 cognigraph-chunker

# With OpenAI embeddings and CORS
docker run -p 3000:3000 \
  -e API_KEY=my-secret \
  -e OPENAI_API_KEY=sk-... \
  -e CORS_ORIGINS=https://example.com \
  cognigraph-chunker
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PORT` | Server port (default: `3000`). Automatically set by Railway, Render, Fly.io. |
| `API_KEY` | Bearer token for API authentication |
| `NO_AUTH` | Set to `1` to disable authentication |
| `CORS_ORIGINS` | Allowed CORS origins |
| `OPENAI_API_KEY` | OpenAI API key for the `openai` embedding provider |
| `CLOUDFLARE_AUTH_TOKEN` | Cloudflare API token for the `cloudflare` embedding provider |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID for the `cloudflare` embedding provider |
| `CLOUDFLARE_AI_GATEWAY` | Cloudflare AI Gateway name (optional) |
| `OAUTH_TOKEN_URL` | OAuth token endpoint URL for the `oauth` embedding provider |
| `OAUTH_CLIENT_ID` | OAuth client ID for the `oauth` embedding provider |
| `OAUTH_CLIENT_SECRET` | OAuth client secret for the `oauth` embedding provider |
| `OAUTH_SCOPE` | OAuth scope (optional) |
| `OAUTH_BASE_URL` | Base URL for the OpenAI-compatible API |
| `OAUTH_MODEL` | Model name for the `oauth` embedding provider |
| `ORT_DYLIB_PATH` | Custom path to ONNX Runtime shared library (only used when the runtime is not on default system paths). Not bundled by this crate. |
| `COGNIGRAPH_LLM_MODEL` | LLM model for `--relations` and `--synopsis` (default: `gpt-4.1-mini`) |
| `NVIDIA_API_KEY` | NVIDIA NIM API key for the `nvidia` reranker |
| `NVIDIA_RERANK_MODEL` | NVIDIA reranker model (default: `nv-rerank-qa-mistral-4b:1`) |
| `NVIDIA_RERANK_BASE_URL` | NVIDIA reranker base URL |
| `COHERE_API_KEY` | Cohere API key for the `cohere` reranker |
| `COHERE_RERANK_MODEL` | Cohere reranker model (default: `rerank-v3.5`) |

### Deploy on Railway / Render / Fly.io

The Dockerfile is ready for container platforms that inject a `PORT` environment variable. Push to your Git repository and connect it to your platform of choice. Set `API_KEY` (or `NO_AUTH=1`) in the platform's environment variable settings.

## Architecture

```
cognigraph-chunker/
  src/
    lib.rs              # Library root (public API)
    main.rs             # CLI entry point
    core/               # Core algorithms (chunk, split, merge, signal processing)
    embeddings/         # Embedding providers (OpenAI, Ollama, ONNX, Cloudflare, OAuth)
      reranker.rs       # Cross-encoder rerankers (NVIDIA NIM, Cohere, Cloudflare, OAuth, ONNX) for boundary refinement
    semantic/           # Semantic and cognitive chunking pipelines
      enrichment/       # Cognitive enrichment (entities, discourse, heading context, language)
      cognitive_*.rs    # Cognitive scoring, assembly, and reranking
      proposition_heal.rs # Proposition-aware chunk healing
      graph_export.rs   # Graph export format (nodes + edges)
      evaluation.rs     # Quality metrics
    llm/                # LLM integration (relation extraction, synopsis generation)
    api/                # REST API (Axum handlers, types, middleware)
    cli/                # CLI subcommands and options
    output/             # Output formatting (plain, json, jsonl)
  packages/
    python/             # Python bindings (PyO3 + maturin)
```

The core algorithms operate on byte slices for zero-copy performance. The semantic pipeline splits text into blocks (markdown-aware or sentence-based), computes embeddings, calculates cross-similarity distances, applies Savitzky-Golay smoothing, and detects topic boundaries at local minima.

The cognitive pipeline extends this with block-level enrichment (entity detection, discourse markers, heading context, continuation flags), weighted multi-signal boundary scoring, valley-based assembly with soft/hard token budgets, and proposition-aware healing that merges chunks with broken cross-references. Language detection runs automatically, selecting appropriate heuristics for 14 language groups.

## License

MIT
