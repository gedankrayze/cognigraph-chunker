# New Chunking Methods Design

**Date:** 2026-04-10
**Status:** Draft
**Scope:** 4 new first-class chunking methods + quality metrics module

## Overview

Four new chunking methods join the existing modes (`chunk`, `split`, `semantic`, `cognitive`) as peer-level CLI subcommands and API endpoints. Each method addresses a distinct gap identified in recent research (March 2026 preprints) and reuses shared infrastructure where possible.

| Method | CLI Command | Key Innovation | LLM Required | Embedding Required |
|--------|-------------|----------------|--------------|-------------------|
| Intent-Driven | `cognigraph intent` | Boundaries optimized for predicted user queries | Yes (intent generation) | Yes (alignment scoring) |
| Topology-Aware | `cognigraph topo` | Hierarchical SIR + dual-agent refinement | Yes (2 agent calls) | No (structure-based) |
| Enriched | `cognigraph enriched` | Single-call 7-field metadata + key-based recombination | Yes (enrichment) | No (structure + LLM) |
| Adaptive | `cognigraph adaptive` | Meta-router: 5 quality metrics select best method per document | Depends on candidates | Yes (for metrics) |

### Architecture: Flat Peer Methods

Each new method follows the exact pattern of existing modes:
- CLI subcommand in `src/cli/{method}_cmd.rs` with clap args
- API handler in `src/api/{method}.rs` with serde request/response
- Core logic in `src/semantic/{method}_chunk.rs`
- LLM prompts in `src/llm/{method}.rs` where applicable
- Registered in `main.rs` Commands enum and `src/api/mod.rs` router

No refactoring of existing methods is required.

### Research References

- **Intent-Driven Dynamic Chunking** (arXiv:2602.14784, Feb 2026) -- LLM intent prediction + DP boundary optimization
- **TopoChunker** (arXiv:2603.18409, Mar 2026) -- topology-aware dual-agent framework with SIR
- **MDKeyChunker** (arXiv:2603.23533, Mar 2026) -- single-call LLM enrichment + semantic-key recombination
- **Adaptive Chunking** (arXiv:2603.25333, Mar 2026, LREC 2026) -- 5 intrinsic quality metrics + per-document method selection
- **Systematic Chunking Study** (arXiv:2603.06976, Mar 2026) -- 36 strategies, 6 domains, paragraph grouping insights
- **GraLC-RAG** (arXiv:2603.22633, Mar 2026) -- KG-infused late chunking for biomedical literature
- **EntiGraph** (arXiv:2409.07431, ICLR 2025) -- entity-centric synthetic data augmentation

---

## Method 1: Intent-Driven Chunking (`intent`)

### Purpose

Optimize chunk boundaries for retrieval by aligning them with predicted user information needs. Instead of asking "where does the topic change?" this asks "what will users search for, and which partition best serves those searches?"

### Pipeline

```
Document
  |
  v
[1] Block extraction (reuse split_blocks / split_sentences)
  |
  v
[2] Intent generation (single LLM call -> 10-30 hypothetical queries)
  |
  v
[3] Block embedding (reuse EmbeddingProvider)
  |
  v
[4] Intent embedding (same provider)
  |
  v
[5] Alignment scoring (chunk centroid <-> best-matching intent)
  |
  v
[6] Dynamic programming (globally optimal partition maximizing alignment)
  |
  v
[7] IntentResult with chunks + intents + alignment scores
```

### Step Details

**Step 2 -- Intent Generation:**
- Uses `CompletionClient` with structured JSON output (`response_format: json_schema`)
- Prompt sends document text (or summary for long docs exceeding context window)
- Schema: `{ intents: [{ query: string, intent_type: "factual" | "procedural" | "conceptual" | "comparative" }] }`
- Default: 20 intents, configurable via `--max-intents`
- Model: configurable via `--intent-model` (default: `gpt-4.1-mini`)

**Step 5 -- Alignment Scoring:**
- For a candidate chunk (contiguous block range), compute centroid embedding (mean of block embeddings)
- Score = max cosine similarity between chunk centroid and any intent embedding
- Total partition score = sum of per-chunk alignment scores, normalized by chunk count

**Step 6 -- Dynamic Programming:**
- State: `dp[i]` = best total alignment score for blocks `0..i`
- Transition: for each valid chunk ending at block `i`, try all start positions `j` where `(i-j)` is within `[min_blocks, max_blocks]`
- `min_blocks` derived from `soft_budget / avg_block_tokens`
- `max_blocks` derived from `hard_budget / avg_block_tokens`
- Time complexity: O(n * max_blocks) where n = number of blocks
- Backtrack to recover the optimal partition

### Data Structures

```rust
// src/semantic/intent_types.rs

pub struct IntentResult {
    pub chunks: Vec<IntentChunk>,
    pub intents: Vec<PredictedIntent>,
    pub partition_score: f64,
}

pub struct IntentChunk {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub token_estimate: usize,
    pub best_intent: usize,          // index into intents vec
    pub alignment_score: f64,
    pub heading_path: Vec<String>,
}

pub struct PredictedIntent {
    pub query: String,
    pub intent_type: IntentType,     // Factual, Procedural, Conceptual, Comparative
    pub matched_chunks: Vec<usize>,  // chunk indices that align with this intent
}
```

### CLI Interface

```
cognigraph intent <FILE> [OPTIONS]

Required:
  <FILE>                    Input file (or - for stdin)

Embedding provider (same flags as semantic/cognitive):
  --provider <PROVIDER>     ollama|openai|onnx|cloudflare|oauth
  --model <MODEL>           Embedding model name
  --base-url <URL>          Provider base URL

LLM configuration:
  --intent-model <MODEL>    Model for intent generation [default: gpt-4.1-mini]
  --api-key <KEY>           API key for LLM (or OPENAI_API_KEY env)
  --llm-base-url <URL>      LLM endpoint [default: https://api.openai.com/v1]

Method parameters:
  --max-intents <N>         Maximum intents to generate [default: 20]
  --soft-budget <N>         Target tokens per chunk [default: 512]
  --hard-budget <N>         Maximum tokens per chunk [default: 768]

Output:
  --format <FMT>            plain|json|jsonl [default: plain]
  --merge                   Post-merge small chunks
  --chunk-size <N>          Merge target size [default: 512]
```

### API Endpoint

```
POST /api/v1/intent

Request:
{
  "text": "...",
  "provider": "openai",
  "model": "text-embedding-3-small",
  "intent_model": "gpt-4.1-mini",
  "max_intents": 20,
  "soft_budget": 512,
  "hard_budget": 768
}

Response:
{
  "chunks": [ { "text": "...", "offset_start": 0, "offset_end": 1234, "token_estimate": 450, "best_intent": 2, "alignment_score": 0.87, "heading_path": ["Introduction"] } ],
  "intents": [ { "query": "What are the side effects of compound XR-7742?", "intent_type": "factual", "matched_chunks": [3, 7] } ],
  "partition_score": 0.82,
  "count": 12
}
```

### New Files

| File | Purpose |
|------|---------|
| `src/semantic/intent_chunk.rs` | DP algorithm, alignment scoring, pipeline orchestration |
| `src/semantic/intent_types.rs` | IntentResult, IntentChunk, PredictedIntent structs |
| `src/llm/intents.rs` | Intent generation prompt + JSON schema |
| `src/cli/intent_cmd.rs` | CLI subcommand definition + run function |
| `src/api/intent.rs` | API handler |

### Reused Components

- `src/semantic/blocks.rs` -- `split_blocks()` for markdown block extraction
- `src/semantic/sentence.rs` -- `split_sentences()` for plain text
- `src/embeddings/*` -- all 5 embedding providers
- `src/llm/mod.rs` -- `CompletionClient` for structured LLM calls
- `src/core/merge.rs` -- optional post-merge

---

## Method 2: Topology-Aware Chunking (`topo`)

### Purpose

Preserve hierarchical document structure during chunking by building an explicit Structured Intermediate Representation (SIR) before making boundary decisions. Two LLM agents inspect and refine the structure, producing chunks that maintain cross-section dependencies.

### Pipeline

```
Document
  |
  v
[1] Block extraction (reuse split_blocks)
  |
  v
[2] Heuristic SIR construction (heading tree + entity co-reference edges)
  |
  v
[3] Inspector Agent (LLM call #1: classify sections, detect cross-references)
  |
  v
[4] Refiner Agent (LLM call #2: resolve ambiguities, produce final partition)
  |
  v
[5] Assembly (map node groups back to text spans)
  |
  v
[6] TopoResult with chunks + SIR + classifications
```

### Step Details

**Step 2 -- SIR Construction:**

The SIR is a tree built from the heading hierarchy with content blocks as leaves:

```
Document (root)
  |-- Section "Introduction" (h1)
  |     |-- Paragraph block 1
  |     |-- Paragraph block 2
  |-- Section "Methods" (h1)
  |     |-- Section "Data Collection" (h2)
  |     |     |-- Paragraph block 3
  |     |     |-- Table block 4
  |     |-- Section "Analysis" (h2)
  |           |-- Paragraph block 5
  |           |-- Code block 6
  |-- Section "Results" (h1)
        |-- Paragraph block 7
        |-- Table block 8
```

Additionally, cross-reference edges are added:
- Entity co-reference: if blocks 3 and 7 both mention "Compound XR-7742", an edge links them
- Discourse continuation: if block 5 starts with "As described above," it gets an edge to the preceding section

This construction reuses our existing `heading_context.rs` for the tree and `entities.rs` + `discourse.rs` for edges.

**Context Window Handling:**
- If the SIR JSON exceeds 80% of the model's context window, large content blocks are summarized (first/last 100 chars + token count) to fit
- Cross-reference edges are preserved even when block text is truncated
- The Refiner receives full text only for sections classified as `splittable` (not the entire document)

**Step 3 -- Inspector Agent:**
- Receives the SIR as a JSON tree (section titles, block types, block lengths -- not full text to keep prompt short)
- Classifies each section node as:
  - `atomic` -- must stay together as one chunk (e.g., a short definition section)
  - `splittable` -- can be divided at block boundaries
  - `merge_candidate` -- should be merged with an adjacent section (e.g., a tiny section with only one paragraph)
- Identifies cross-section dependencies: pairs of sections that reference each other
- Schema: `{ classifications: [{ section_id, class, reason }], dependencies: [{ from, to, type }] }`

**Step 4 -- Refiner Agent:**
- Receives Inspector's output + the SIR + full text of ambiguous sections only
- For `splittable` sections: determines optimal split points within the section
- For `merge_candidate` pairs: decides direction of merge
- For cross-section dependencies: ensures dependent content stays in the same chunk or has explicit cross-references in output
- Outputs: `{ partition: [{ chunk_id, section_ids, block_ranges }] }`

**Step 5 -- Assembly:**
- Map the Refiner's partition back to byte ranges in the original document
- Each chunk carries its heading ancestry, the Inspector's classification, and any cross-chunk references

### Data Structures

```rust
// src/semantic/sir.rs

pub struct SirNode {
    pub id: usize,
    pub node_type: SirNodeType,        // Section, ContentBlock
    pub heading: Option<String>,
    pub heading_level: Option<u8>,
    pub block_type: Option<BlockType>,  // Sentence, Table, Code, etc.
    pub block_range: (usize, usize),   // start/end block indices
    pub children: Vec<usize>,          // child node IDs
    pub text_preview: String,          // first 200 chars for LLM context
    pub token_estimate: usize,
}

pub struct SirEdge {
    pub from: usize,
    pub to: usize,
    pub edge_type: SirEdgeType,        // EntityCoref, DiscourseContinuation, HeadingNesting
}

pub struct Sir {
    pub nodes: Vec<SirNode>,
    pub edges: Vec<SirEdge>,
    pub root: usize,
}

// src/semantic/topo_types.rs

pub struct TopoResult {
    pub chunks: Vec<TopoChunk>,
    pub sir: Sir,
    pub classifications: Vec<SectionClassification>,
}

pub struct TopoChunk {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub token_estimate: usize,
    pub heading_path: Vec<String>,
    pub section_classification: String,  // atomic|splittable|merged
    pub cross_references: Vec<usize>,    // other chunk indices this depends on
}

pub struct SectionClassification {
    pub section_id: usize,
    pub class: SectionClass,           // Atomic, Splittable, MergeCandidate
    pub reason: String,
}
```

### CLI Interface

```
cognigraph topo <FILE> [OPTIONS]

Required:
  <FILE>                    Input file (or - for stdin)

LLM configuration:
  --topo-model <MODEL>      Model for both agents [default: gpt-4.1-mini]
  --api-key <KEY>           API key for LLM
  --llm-base-url <URL>      LLM endpoint [default: https://api.openai.com/v1]

Method parameters:
  --soft-budget <N>         Target tokens per chunk [default: 512]
  --hard-budget <N>         Maximum tokens per chunk [default: 768]

Output:
  --format <FMT>            plain|json|jsonl [default: plain]
  --emit-sir                Include SIR structure in JSON output
```

### API Endpoint

```
POST /api/v1/topo

Request:
{
  "text": "...",
  "topo_model": "gpt-4.1-mini",
  "soft_budget": 512,
  "hard_budget": 768,
  "emit_sir": false
}

Response:
{
  "chunks": [ { "text": "...", "offset_start": 0, "offset_end": 2048, "token_estimate": 480, "heading_path": ["Methods", "Data Collection"], "section_classification": "atomic", "cross_references": [4] } ],
  "count": 8
}
```

### New Files

| File | Purpose |
|------|---------|
| `src/semantic/sir.rs` | SIR data structures (SirNode, SirEdge, Sir) |
| `src/semantic/topo_chunk.rs` | SIR builder, assembly, pipeline orchestration |
| `src/semantic/topo_types.rs` | TopoResult, TopoChunk structs |
| `src/llm/topo_agents.rs` | Inspector + Refiner prompts and JSON schemas |
| `src/cli/topo_cmd.rs` | CLI subcommand |
| `src/api/topo.rs` | API handler |

### Reused Components

- `src/semantic/blocks.rs` -- `split_blocks()` and `BlockType` enum
- `src/semantic/enrichment/heading_context.rs` -- heading hierarchy
- `src/semantic/enrichment/entities.rs` -- entity detection for co-reference edges
- `src/semantic/enrichment/discourse.rs` -- discourse markers for continuation edges
- `src/llm/mod.rs` -- `CompletionClient`

---

## Method 3: Enriched Chunking (`enriched`)

### Purpose

Structure-preserving chunking combined with rich LLM-generated metadata per chunk, followed by semantic-key-based recombination. Produces chunks that are self-describing and retrieval-optimized without requiring embeddings.

### Pipeline

```
Document
  |
  v
[1] Block extraction (reuse split_blocks)
  |
  v
[2] Initial grouping (size-based, soft budget target)
  |
  v
[3] Single-call LLM enrichment per chunk (7 metadata fields, rolling key dict)
  |
  v
[4] Key-based recombination (bin-packing on shared semantic keys)
  |
  v
[5] Optional re-enrichment (update title/summary for merged chunks)
  |
  v
[6] EnrichedResult with chunks + metadata + key dictionary
```

### Step Details

**Step 2 -- Initial Grouping:**
- Simple greedy accumulation: add blocks to current chunk until soft budget is reached
- Respect atomic blocks (tables, code blocks, lists stay whole)
- Heading blocks start a new chunk (same as semantic mode)
- No embeddings needed -- purely structural

**Step 3 -- Single-Call LLM Enrichment:**

Each chunk gets one LLM call that extracts all 7 fields simultaneously:

```json
{
  "title": "Compound XR-7742 Dosing Protocol",
  "summary": "Describes the three-phase dosing schedule for XR-7742 in adult patients, including titration windows and contraindications.",
  "keywords": ["XR-7742", "dosing", "titration", "contraindications", "phase I"],
  "typed_entities": [
    {"name": "XR-7742", "type": "compound"},
    {"name": "Phase I", "type": "study_phase"},
    {"name": "FDA", "type": "organization"}
  ],
  "hypothetical_questions": [
    "What is the recommended starting dose for XR-7742?",
    "What are the contraindications for XR-7742?",
    "How long is the titration window for XR-7742?"
  ],
  "semantic_keys": ["xr-7742-dosing", "clinical-protocol"],
  "category": "methodology"
}
```

**Rolling Key Dictionary:**
- Maintained across chunks during sequential processing
- Each enrichment call receives the current dictionary as context
- LLM can reuse existing keys (creating shared-key links) or create new ones
- Dictionary: `HashMap<String, Vec<usize>>` -- key name to list of chunk indices

**Step 4 -- Key-Based Recombination:**
- Identify chunks sharing identical semantic keys
- For each group of same-key chunks, attempt bin-packing merge subject to hard budget
- Priority: merge chunks that are both same-key AND adjacent in document order
- Non-adjacent same-key chunks: merge only if combined size fits within hard budget
- Chunks with unique keys (no sharing) remain untouched

**Step 5 -- Re-Enrichment:**
- Only runs for chunks that were actually merged (content changed)
- Lightweight call: update `title` and `summary` only (not all 7 fields)
- Preserves original keywords, entities, questions, keys from constituent chunks (union)

### Data Structures

```rust
// src/semantic/enriched_types.rs

pub struct EnrichedResult {
    pub chunks: Vec<EnrichedChunk>,
    pub key_dictionary: HashMap<String, Vec<usize>>,
    pub merge_history: Vec<MergeRecord>,
}

pub struct EnrichedChunk {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub token_estimate: usize,
    pub title: String,
    pub summary: String,
    pub keywords: Vec<String>,
    pub typed_entities: Vec<TypedEntity>,
    pub hypothetical_questions: Vec<String>,
    pub semantic_keys: Vec<String>,
    pub category: String,
    pub heading_path: Vec<String>,
}

pub struct TypedEntity {
    pub name: String,
    pub entity_type: String,
}

pub struct MergeRecord {
    pub result_chunk: usize,
    pub source_chunks: Vec<usize>,
    pub shared_key: String,
}
```

### CLI Interface

```
cognigraph enriched <FILE> [OPTIONS]

Required:
  <FILE>                        Input file (or - for stdin)

LLM configuration:
  --enrichment-model <MODEL>    Model for enrichment [default: gpt-4.1-mini]
  --api-key <KEY>               API key for LLM
  --llm-base-url <URL>          LLM endpoint [default: https://api.openai.com/v1]

Method parameters:
  --soft-budget <N>             Target tokens per chunk [default: 512]
  --hard-budget <N>             Maximum tokens per chunk [default: 768]
  --no-recombine                Skip key-based recombination step
  --no-re-enrich                Skip re-enrichment of merged chunks

Output:
  --format <FMT>                plain|json|jsonl [default: plain]
```

### API Endpoint

```
POST /api/v1/enriched

Request:
{
  "text": "...",
  "enrichment_model": "gpt-4.1-mini",
  "soft_budget": 512,
  "hard_budget": 768,
  "recombine": true,
  "re_enrich": true
}

Response:
{
  "chunks": [ { "text": "...", "title": "...", "summary": "...", "keywords": [...], "typed_entities": [...], "hypothetical_questions": [...], "semantic_keys": [...], "category": "...", "heading_path": [...] } ],
  "key_dictionary": { "xr-7742-dosing": [0, 3], "clinical-protocol": [0, 1, 3] },
  "count": 10
}
```

### New Files

| File | Purpose |
|------|---------|
| `src/semantic/enriched_chunk.rs` | Pipeline orchestration, initial grouping, key-based recombination |
| `src/semantic/enriched_types.rs` | EnrichedResult, EnrichedChunk, TypedEntity structs |
| `src/llm/enrichment.rs` | 7-field enrichment prompt, JSON schema, rolling key logic |
| `src/cli/enriched_cmd.rs` | CLI subcommand |
| `src/api/enriched.rs` | API handler |

### Reused Components

- `src/semantic/blocks.rs` -- `split_blocks()` and block types
- `src/semantic/enrichment/heading_context.rs` -- heading path for chunks
- `src/llm/mod.rs` -- `CompletionClient`
- No embedding provider needed

---

## Method 4: Adaptive Chunking (`adaptive`)

### Purpose

Meta-router that automatically selects the best chunking method for each document by evaluating 5 intrinsic quality metrics on the output of candidate methods. No single chunking strategy works best for all documents -- adaptive routing closes this gap.

### Pipeline

```
Document
  |
  v
[1] Pre-screening (heuristic: which candidates are worth trying?)
  |
  v
[2] Run candidate methods (parallel where possible)
  |
  v
[3] Score each candidate's output (5 intrinsic quality metrics)
  |
  v
[4] Select winner (highest composite score)
  |
  v
[5] AdaptiveResult = winner's output + quality report
```

### Step Details

**Step 1 -- Pre-Screening:**

Lightweight heuristics to skip expensive methods on unsuitable documents:

| Method | Skip If |
|--------|---------|
| `topo` | Document has < 2 heading levels (flat structure, nothing to topology-ize) |
| `intent` | Document < 500 tokens (too short for meaningful intent generation) |
| `enriched` | Document has no markdown structure AND < 1000 tokens (too simple to benefit from enrichment overhead) |
| `cognitive` | (always run -- it's the general-purpose heavyweight) |
| `semantic` | (always run -- it's the general-purpose baseline) |

Pre-screening is advisory: `--force-candidates` overrides it.

**Step 2 -- Running Candidates:**
- Default candidates: `semantic`, `cognitive`, `intent`, `enriched`, `topo`
- User can restrict: `--candidates semantic,cognitive,intent`
- Methods are run sequentially (LLM-dependent ones can't easily parallelize due to shared API key rate limits)
- Results cached in memory -- if adaptive is called after a manual cognitive run in the same API session, the cached result is reused

**Step 3 -- Quality Metrics:**

Five intrinsic metrics, each scored 0.0 to 1.0:

**Size Compliance (SC):**
```
SC = count(chunks where soft_budget * 0.5 <= tokens <= hard_budget) / total_chunks
```
Measures whether chunks are appropriately sized. Too small = fragmented, too large = monolithic.

**Intrachunk Cohesion (ICC):**
```
For each chunk:
  1. Split into sentences
  2. Embed each sentence + embed full chunk text
  3. Compute mean cosine similarity of sentence embeddings to chunk embedding
ICC = mean of per-chunk cohesion scores
```
Requires embedding provider. Measures whether each chunk is about one thing.

**Contextual Coherence (DCC):**
```
For each adjacent chunk pair (i, i+1):
  1. Embed both chunk texts
  2. Compute cosine similarity
DCC = mean of adjacent-pair similarities
```
Higher = smoother transitions. Very high values may indicate under-splitting.

**Block Integrity (BI):**
```
BI = count(structural_elements_fully_contained) / total_structural_elements
```
Structural elements = tables, code blocks, lists, block quotes. Checks that none are split across chunk boundaries. Uses byte offset tracking from block extraction.

**Reference Completeness (RC):**
```
For each chunk boundary:
  1. Check if next chunk starts with pronoun/demonstrative without antecedent in same chunk
  2. Check if entities introduced in prev chunk are referenced in next chunk (entity orphan)
RC = 1.0 - (orphan_count / total_boundary_count)
```
Maps directly to our existing orphan risk and entity continuity signals from cognitive mode.

**Composite Score:**
```
composite = w_sc * SC + w_icc * ICC + w_dcc * DCC + w_bi * BI + w_rc * RC
```
Default weights: equal (0.20 each). Configurable via `--metric-weights sc=0.15,icc=0.25,dcc=0.20,bi=0.20,rc=0.20`.

**Step 4 -- Winner Selection:**
- Method with highest composite score wins
- Ties broken by: fewer chunks (prefer less fragmentation)
- Output includes full quality report for all candidates

### Data Structures

```rust
// src/semantic/quality_metrics.rs

pub struct QualityMetrics {
    pub size_compliance: f64,
    pub intrachunk_cohesion: f64,
    pub contextual_coherence: f64,
    pub block_integrity: f64,
    pub reference_completeness: f64,
    pub composite: f64,
}

pub struct MetricWeights {
    pub sc: f64,
    pub icc: f64,
    pub dcc: f64,
    pub bi: f64,
    pub rc: f64,
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self { sc: 0.20, icc: 0.20, dcc: 0.20, bi: 0.20, rc: 0.20 }
    }
}

// src/semantic/adaptive_types.rs

pub struct AdaptiveResult {
    pub winner: String,                          // method name
    pub chunks: Vec<serde_json::Value>,          // winner's chunk output (polymorphic -- each method has different metadata, so we use Value for the unified response)
    pub report: AdaptiveReport,
}

pub struct AdaptiveReport {
    pub candidates: Vec<CandidateScore>,
    pub pre_screening: Vec<ScreeningDecision>,
    pub metric_weights: MetricWeights,
}

pub struct CandidateScore {
    pub method: String,
    pub metrics: QualityMetrics,
    pub chunk_count: usize,
    pub total_tokens: usize,
}

pub struct ScreeningDecision {
    pub method: String,
    pub included: bool,
    pub reason: String,
}
```

### CLI Interface

```
cognigraph adaptive <FILE> [OPTIONS]

Required:
  <FILE>                        Input file (or - for stdin)

Embedding provider (needed for ICC/DCC metrics + embedding-based candidates):
  --provider <PROVIDER>         ollama|openai|onnx|cloudflare|oauth
  --model <MODEL>               Embedding model name
  --base-url <URL>              Provider base URL

LLM configuration (needed for LLM-based candidates):
  --api-key <KEY>               API key for LLM
  --llm-base-url <URL>          LLM endpoint [default: https://api.openai.com/v1]
  --llm-model <MODEL>           Model for LLM-based methods [default: gpt-4.1-mini]

Method parameters:
  --candidates <LIST>           Comma-separated method names [default: semantic,cognitive,intent,enriched,topo]
  --force-candidates            Bypass pre-screening heuristics
  --metric-weights <WEIGHTS>    Metric weights as key=value pairs [default: equal]
  --soft-budget <N>             Target tokens per chunk [default: 512]
  --hard-budget <N>             Maximum tokens per chunk [default: 768]

Output:
  --format <FMT>                plain|json|jsonl [default: plain]
  --report                      Include quality report in output (JSON only)
```

### API Endpoint

```
POST /api/v1/adaptive

Request:
{
  "text": "...",
  "provider": "openai",
  "model": "text-embedding-3-small",
  "api_key": "...",
  "candidates": ["semantic", "cognitive", "intent", "enriched"],
  "soft_budget": 512,
  "hard_budget": 768,
  "metric_weights": { "sc": 0.15, "icc": 0.25, "dcc": 0.20, "bi": 0.20, "rc": 0.20 },
  "include_report": true
}

Response:
{
  "winner": "cognitive",
  "chunks": [ ... ],
  "count": 12,
  "report": {
    "candidates": [
      { "method": "semantic", "metrics": { "size_compliance": 0.85, "intrachunk_cohesion": 0.72, "contextual_coherence": 0.68, "block_integrity": 0.90, "reference_completeness": 0.65, "composite": 0.76 }, "chunk_count": 15, "total_tokens": 6200 },
      { "method": "cognitive", "metrics": { "size_compliance": 0.92, "intrachunk_cohesion": 0.78, "contextual_coherence": 0.71, "block_integrity": 0.95, "reference_completeness": 0.88, "composite": 0.85 }, "chunk_count": 12, "total_tokens": 5800 }
    ],
    "pre_screening": [
      { "method": "topo", "included": false, "reason": "Document has < 2 heading levels" }
    ]
  }
}
```

### New Files

| File | Purpose |
|------|---------|
| `src/semantic/quality_metrics.rs` | 5 metric implementations (standalone, reusable for benchmarking) |
| `src/semantic/adaptive_chunk.rs` | Orchestrator: pre-screening, candidate dispatch, scoring, selection |
| `src/semantic/adaptive_types.rs` | AdaptiveResult, AdaptiveReport, CandidateScore structs |
| `src/cli/adaptive_cmd.rs` | CLI subcommand |
| `src/api/adaptive.rs` | API handler |

### Reused Components

- All existing chunking methods as callable functions
- `src/embeddings/*` -- for ICC and DCC metric computation
- `src/semantic/enrichment/entities.rs` -- for RC metric
- `src/semantic/blocks.rs` -- for BI metric (block type tracking)
- `src/semantic/enrichment/heading_context.rs` -- for orphan detection in RC

---

## Quality Metrics Module (Standalone)

The 5 quality metrics are implemented as a standalone module (`src/semantic/quality_metrics.rs`) that can evaluate ANY chunking output, not just adaptive candidates. This enables:

- **Benchmarking:** compare chunking methods on a corpus
- **CI integration:** assert minimum quality scores in tests
- **API endpoint:** `POST /api/v1/evaluate` accepts pre-chunked output and returns metrics

```rust
// Standalone evaluation function
pub async fn evaluate_chunks<P: EmbeddingProvider>(
    original_text: &str,
    chunks: &[ChunkForEval],
    provider: &P,
    config: &MetricConfig,
) -> Result<QualityMetrics>

pub struct ChunkForEval {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
}

pub struct MetricConfig {
    pub soft_budget: usize,
    pub hard_budget: usize,
    pub weights: MetricWeights,
}
```

---

## Shared Infrastructure Summary

| Component | Used By |
|-----------|---------|
| `split_blocks()` | intent, topo, enriched, adaptive (via candidates) |
| `EmbeddingProvider` trait | intent, adaptive (metrics) |
| `CompletionClient` | intent, topo, enriched |
| `heading_context.rs` | topo (SIR), enriched (heading paths) |
| `entities.rs` | topo (co-reference edges), adaptive (RC metric) |
| `discourse.rs` | topo (continuation edges) |
| `merge_splits()` | intent (optional post-merge), enriched (recombination) |
| Block type tracking | adaptive (BI metric), enriched (atomic blocks) |

---

## Registration Changes

### main.rs -- Commands Enum

Add 4 variants:
```rust
Intent(Box<cli::intent_cmd::IntentArgs>),
Topo(Box<cli::topo_cmd::TopoArgs>),
Enriched(Box<cli::enriched_cmd::EnrichedArgs>),
Adaptive(Box<cli::adaptive_cmd::AdaptiveArgs>),
```

### main.rs -- Match Arms

```rust
Commands::Intent(args) => cli::intent_cmd::run(args, &cli.global).await,
Commands::Topo(args) => cli::topo_cmd::run(args, &cli.global).await,
Commands::Enriched(args) => cli::enriched_cmd::run(args, &cli.global).await,
Commands::Adaptive(args) => cli::adaptive_cmd::run(args, &cli.global).await,
```

### src/api/mod.rs -- Router

```rust
.route("/api/v1/intent", axum::routing::post(intent::intent_handler))
.route("/api/v1/topo", axum::routing::post(topo::topo_handler))
.route("/api/v1/enriched", axum::routing::post(enriched::enriched_handler))
.route("/api/v1/adaptive", axum::routing::post(adaptive::adaptive_handler))
.route("/api/v1/evaluate", axum::routing::post(evaluate::evaluate_handler))
```

### lib.rs -- Module Declarations

No new top-level modules needed. New files go under existing `semantic`, `llm`, `cli`, `api` modules.

---

## Testing Strategy

Each method gets:

1. **Unit tests** in its core module (pipeline logic, scoring, assembly)
2. **Integration test** with a sample markdown file (end-to-end through CLI)
3. **Quality metric validation** -- assert metrics are computed correctly on known inputs

Adaptive gets additional tests:
- Pre-screening logic (flat doc skips topo, short doc skips intent)
- Winner selection with known metric values
- Composite score computation with custom weights

Estimated new tests: ~40-50 across all methods.

---

## Documentation

Each method gets a doc article in `docs/`:
- `10-intent-driven-chunking.md`
- `11-topology-aware-chunking.md`
- `12-enriched-chunking.md`
- `13-adaptive-chunking.md`

Plus update `README.md` with the new modes in the CLI reference table.

---

## Implementation Order

Recommended sequence based on dependency graph:

1. **Quality Metrics module** -- needed by Adaptive, useful for validating other methods
2. **Intent-Driven** -- independent, reuses embeddings + LLM
3. **Enriched** -- independent, reuses blocks + LLM
4. **Topology-Aware** -- independent, reuses enrichment + LLM
5. **Adaptive** -- depends on all other methods being available as candidates
6. **Documentation** -- articles + README update
7. **Benchmarks** -- run quality metrics across methods on sample corpus
