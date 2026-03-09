# CogniGraph Chunker — Cognition-Aware Chunking Implementation Plan

> **Thesis:** Treat text as a graph of propositions. Blocks are candidate nodes, relations are candidate edges, chunking is subgraph assembly, and token budgets are merely deployment constraints.

## Objective

CogniGraph cognitive chunking groups text into self-contained cognitive units by preserving entity integrity, relational completeness, and concept continuity, while still respecting soft token budgets.

This is not "better chunking." It is a shift from boundary detection over text to boundary detection over meaning-bearing units.

---

## Architecture Overview

The pipeline shape remains familiar:

```
parse → blocks → enrich → score boundaries → assemble chunks → merge if needed
```

This extends the existing semantic pipeline rather than replacing it. The current system already has the right substrate: block extraction, markdown-aware structural preservation, embedding-based boundary detection, signal smoothing, and token-aware merge normalization.

### Four Layers

| Layer | Description | Status |
|-------|-------------|--------|
| **1. Structural Segmentation** | Markdown-aware block extraction (headings, tables, code, lists, quotes, sentence-split prose) | ✅ Exists |
| **2. Cognitive Enrichment** | Annotate each block with entities, discourse markers, heading path, multilingual support (14 languages) | ✅ Complete |
| **3. Continuity Scoring** | Weighted cognitive boundary cost function with valley detection | ✅ Complete |
| **4. Constrained Assembly** | Chunk formation under soft/hard token budgets with graceful degradation | ✅ Complete |

---

## Data Structures

### BlockEnvelope (enriched block)

```rust
pub struct BlockEnvelope {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub block_type: BlockKind,          // Heading, Sentence, Code, Table, List, Quote
    pub heading_path: Vec<String>,      // ["CogniGraph Project", "Architecture", "Scoring"]
    pub embedding: Option<Vec<f64>>,

    // Cognitive signals
    pub entities: Vec<NormalizedEntity>,
    pub noun_phrases: Vec<String>,
    pub discourse_markers: Vec<DiscourseMarker>,
    pub continuation_flags: ContinuationFlags,
    pub token_estimate: usize,
}
```

### NormalizedEntity

```rust
pub struct NormalizedEntity {
    pub surface_form: String,           // "the chunker", "it", "CogniGraph"
    pub normalized: String,             // "cognigraph_chunker"
    pub entity_type: EntityType,        // Named, NounPhrase, Pronoun, Demonstrative
    pub span: (usize, usize),
}
```

### RelationTriple (LLM-extracted, post-assembly)

```rust
// src/llm/relations.rs
pub struct RelationTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}
```

Relations are extracted per-chunk via LLM structured JSON output after chunk assembly,
not during boundary scoring. See `src/llm/relations.rs`.

### DiscourseMarker

```rust
pub enum DiscourseMarker {
    Continuation,   // "furthermore", "additionally", "also", "moreover"
    Contrast,       // "however", "but", "on the other hand", "in contrast"
    Causation,      // "therefore", "because", "thus", "consequently"
    Exemplification,// "for example", "such as", "e.g.", "for instance"
    Elaboration,    // "specifically", "in particular", "namely"
    Conclusion,     // "in summary", "finally", "in conclusion"
}
```

### ContinuationFlags

```rust
pub struct ContinuationFlags {
    pub starts_with_pronoun: bool,      // "It", "They", "This"
    pub starts_with_demonstrative: bool,// "This model", "These results", "Such systems"
    pub starts_with_discourse: bool,    // "Furthermore", "However"
    pub continues_list: bool,           // numbered/bulleted continuation
    pub shares_subject: bool,           // same subject as previous block
}
```

### CognitiveChunk (output)

```rust
pub struct CognitiveChunk {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub heading_path: Vec<String>,
    pub dominant_entities: Vec<String>,
    pub dominant_relations: Vec<crate::llm::relations::RelationTriple>,
    pub token_estimate: usize,
    pub boundary_reasons_start: Vec<BoundaryReason>,
    pub boundary_reasons_end: Vec<BoundaryReason>,
    pub continuity_confidence: f64,
}
```

### BoundaryReason

```rust
pub enum BoundaryReason {
    TopicShift { similarity_drop: f64 },
    HeadingChange { from: String, to: String },
    EntityDiscontinuity { orphaned: Vec<String> },
    PropositionComplete,
    BudgetCeiling { tokens: usize },
    DiscourseBreak { marker: DiscourseMarker },
    StructuralBarrier { block_type: BlockType },
}
```

---

## Boundary Scoring Formula

### Join Score

A boundary is modeled as a cost. High join score = keep together. Low join score = candidate break.

```
join_score(i, i+1) =
    w_sem   * semantic_similarity
  + w_ent   * entity_continuity
  + w_rel   * relation_continuity    # 0.0 — relations extracted post-assembly
  + w_disc  * discourse_continuation
  + w_head  * heading_context_continuity
  + w_struct * structural_affinity
  + w_orphan * orphan_risk           # high risk = keep together
  - w_shift * topic_shift_penalty
  - w_budget * budget_pressure
```

Then: `break_score = 1 - normalized(join_score)`

Splits occur at **local valleys** in the join score curve, not at a flat threshold. The existing Savitzky-Golay smoothing and local-minima detection should be reused here.

### Component Definitions

| Component | Signal | Source |
|-----------|--------|--------|
| **semantic_similarity** | Cosine similarity of adjacent block embeddings | Existing |
| **entity_continuity** | Overlap of normalized entities/noun phrases/aliases between blocks | New |
| **relation_continuity** | Reserved (always 0.0 — relations extracted post-assembly via LLM, not used as boundary signal) | Phase 2 |
| **discourse_continuation** | Presence and type of discourse markers ("furthermore" → high, "in conclusion" → low) | New |
| **heading_context_continuity** | Both blocks under same heading path; or block is first child of a heading | Derivable from existing parser |
| **structural_affinity** | Known cohesive patterns: heading+intro, paragraph+code example, paragraph+table | New |
| **topic_shift_penalty** | Large embedding distance between blocks (inverse of semantic similarity) | Existing (inverted) |
| **orphan_risk** | Split would leave pronoun without antecedent, heading without explanation, or incomplete triple | New |
| **budget_pressure** | Soft penalty that increases as accumulated tokens approach ceiling | New |

### Default Weights (tunable)

```rust
pub struct CognitiveWeights {
    pub w_sem: f64,     // 0.30
    pub w_ent: f64,     // 0.20
    pub w_rel: f64,     // 0.0   (relations extracted post-assembly via LLM)
    pub w_disc: f64,    // 0.10
    pub w_head: f64,    // 0.10
    pub w_struct: f64,  // 0.05
    pub w_shift: f64,   // 0.15
    pub w_orphan: f64,  // 0.20
    pub w_budget: f64,  // 0.10
}
```

---

## Entity Detection Strategy (No Heavy NLP)

Three tiers of evidence, gracefully degrading:

### Level A — Cheap Lexical (Phase 1) ✅

- Repeated capitalized spans across blocks
- Repeated noun phrases (simple POS-like heuristic)
- Heading terms propagated downward
- Known domain entities from a user-supplied glossary (optional)
- Exact normalized string recurrence

### Level B — Structural (Phase 1) ✅

- Subject repetition across sentences
- Demonstratives: "this model", "that service", "the tool" (14 languages)
- Pronoun-start detection: "It", "They", "This", "These", "Such" (14 languages)
- Discourse marker detection: 6 categories, 14 languages

### Level B+ — Script-Based (Phase 1.5) ✅

- Katakana span detection for Japanese proper nouns/loanwords (with nakaguro ・ and prolonged sound mark ー support)
- Latin-in-CJK detection for foreign names, acronyms, technical terms (e.g., "ONNX" in Chinese text)
- Language auto-detection via `whatlang` trigram analysis (~70 languages → 14 language groups)
- Per-language stopword filtering for entity span cleanup

### Level C — Model-Derived (Phase 3+, optional)

- NER via ONNX model (GLiNER-Multi or WikiNeural)
- Alias linking
- Light coreference resolution

---

## Multilingual Support ✅

The enrichment pipeline automatically detects the document language and applies language-appropriate heuristics. No configuration required — detection is transparent and the pipeline degrades gracefully for unsupported languages.

### Implementation

- **Language detection**: `whatlang` crate (trigram-based, ~70 languages, lightweight)
- **Language groups**: 14 supported groups — English, German, French, Spanish, Portuguese, Italian, Dutch, Russian, Turkish, Polish, Chinese, Japanese, Korean, Arabic, plus `Other` fallback
- **Discourse markers**: Per-language pattern tables covering all 6 discourse categories
- **Pronouns/demonstratives**: Per-language word lists for continuation flag detection
- **Script-based entities**: Unicode script properties via `unicode-script` crate:
  - Katakana runs → proper nouns/loanwords in Japanese
  - Latin runs in CJK/Arabic text → foreign names, acronyms, technical terms
- **Stopwords**: Per-language function word lists for entity span filtering
- **Fallback**: Unsupported languages still get embedding-based similarity, heading awareness, structural affinity, and budget management — only discourse/pronoun/demonstrative signals are empty

### Supported Languages

| Language | Discourse | Pronouns | Demonstratives | Script entities |
|----------|-----------|----------|----------------|-----------------|
| English | ✅ 70+ patterns | ✅ | ✅ | — |
| German | ✅ 40 patterns | ✅ | ✅ | — |
| French | ✅ 46 patterns | ✅ | ✅ | — |
| Spanish | ✅ 35 patterns | ✅ | ✅ | — |
| Portuguese | ✅ 25 patterns | ✅ | ✅ | — |
| Italian | ✅ 28 patterns | ✅ | ✅ | — |
| Dutch | ✅ 29 patterns | ✅ | ✅ | — |
| Russian | ✅ 30 patterns | ✅ | ✅ | — |
| Turkish | ✅ 24 patterns | ✅ | ✅ | — |
| Polish | ✅ 28 patterns | ✅ | ✅ | — |
| Chinese | ✅ 36 patterns | — | — | ✅ Latin-in-CJK |
| Japanese | ✅ 33 patterns | — | — | ✅ Katakana + Latin-in-CJK |
| Korean | ✅ 17 patterns | — | — | ✅ Latin-in-CJK |
| Arabic | ✅ 28 patterns | — | — | ✅ Latin detection |

---

## Reranker/Cross-Encoder Strategy

Cross-encoders should **not** be the primary engine for all adjacent decisions.

### Staged Scoring

1. Compute heuristic join score for all boundaries (fast)
2. Identify **ambiguous boundaries** where score falls in uncertainty band
3. Only send ambiguous pairs to cross-encoder/reranker for refinement
4. Update join score with reranker result

This avoids O(n) expensive inference calls. In practice, ~10-20% of boundaries will be ambiguous, so the reranker processes a fraction of the total.

### Provider Integration

Model reranking as a sibling trait to `EmbeddingProvider`:

```rust
pub trait RerankerProvider: Send + Sync {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>>;
}
```

Supported backends: ONNX (local), OpenAI, Cloudflare, OAuth — reusing existing provider infrastructure.

---

## Module Layout

Extend the existing `src/semantic/` area:

```
src/semantic/
├── blocks.rs                    # Existing — markdown-aware block extraction
├── sentence.rs                  # Existing — sentence splitting
├── mod.rs                       # Existing — semantic_chunk + cognitive_chunk entry points
│
├── enrichment/
│   ├── mod.rs                   # BlockEnvelope construction + language-aware pipeline
│   ├── entities.rs              # Entity extraction (Level A + B, capitalized spans)
│   ├── discourse.rs             # English discourse marker detection (legacy, still used by tests)
│   ├── heading_context.rs       # Heading path propagation
│   ├── language.rs              # Language detection (whatlang), per-language pronouns/demonstratives/stopwords
│   ├── multilingual_discourse.rs # Discourse markers for 14 languages (EN,DE,FR,ES,PT,IT,NL,RU,TR,PL,ZH,JA,KO,AR)
│   └── script_entities.rs       # Unicode script-based entity extraction (Katakana, Latin-in-CJK)
│
├── cognitive_types.rs           # All cognitive data structures
├── cognitive_score.rs           # Boundary scoring function
├── cognitive_assemble.rs        # Chunk formation with soft constraints
├── diagnostics.rs               # Score breakdown output
└── evaluation.rs                # Quality metrics (entity orphan, pronoun boundary, heading attachment, discourse break)

src/llm/
├── mod.rs                       # OpenAI-compatible completion client (structured JSON output)
└── relations.rs                 # LLM-based relation triple extraction (per-chunk, post-assembly)
```

CLI subcommand: `src/cli/cognitive_cmd.rs`
API route: `src/api/cognitive.rs`
Python binding: `packages/python/src/cognitive.rs`
Benchmark: `tests/benchmark_comparison.rs`

---

## CLI Surface

```bash
# Basic cognitive chunking
cognigraph-chunker cognitive -i doc.md

# With provider
cognigraph-chunker cognitive -i doc.md --provider ollama

# Heuristic-only (no embeddings needed for Level A signals)
cognigraph-chunker cognitive -i doc.md --lite

# Enable relation awareness (Phase 2)
cognigraph-chunker cognitive -i doc.md --relations

# Reranker for ambiguous boundaries (Phase 3)
cognigraph-chunker cognitive -i doc.md --reranker onnx --reranker-model path/to/model

# Token budget controls
cognigraph-chunker cognitive -i doc.md --soft-budget 512 --hard-budget 768

# Full diagnostic output
cognigraph-chunker cognitive -i doc.md --emit-signals

# Output formats (same as existing)
cognigraph-chunker cognitive -i doc.md -o json
cognigraph-chunker cognitive -i doc.md -o jsonl
```

### Diagnostic Output (--emit-signals)

For each candidate boundary, emit:

```json
{
  "boundary_index": 5,
  "block_a": "Entity extraction uses...",
  "block_b": "It also supports...",
  "scores": {
    "semantic_similarity": 0.72,
    "entity_continuity": 0.85,
    "relation_continuity": 0.0,
    "discourse_continuation": 0.60,
    "heading_continuity": 1.0,
    "structural_affinity": 0.0,
    "topic_shift_penalty": 0.08,
    "orphan_risk": 0.70,
    "budget_pressure": 0.05
  },
  "join_score": 0.78,
  "break_score": 0.22,
  "decision": "join",
  "reason": "high entity continuity + pronoun lead in block B"
}
```

## REST API Surface

```
POST /api/v1/cognitive
```

Request body follows existing patterns, adding cognitive-specific fields:

```json
{
  "text": "...",
  "provider": "ollama",
  "model": "nomic-embed-text",
  "lite": false,
  "relations": false,
  "soft_budget": 512,
  "hard_budget": 768,
  "emit_signals": false,
  "weights": {
    "w_sem": 0.30,
    "w_ent": 0.20,
    "w_orphan": 0.20
  }
}
```

---

## Phased Roadmap

### Phase 1 — Cognitive-Lite (highest priority)

Ship a cognitive boundary scorer that reduces entity orphaning and preserves claim continuity better than plain semantic chunking.

No heavy NLP. No full graph extraction. No cross-encoder in the hot path.

- [x] **1.1** Define `cognitive_types.rs` — all data structures (`BlockEnvelope`, `CognitiveChunk`, `BoundaryReason`, `CognitiveWeights`, etc.)
- [x] **1.2** Implement `enrichment/entities.rs` — Level A + B entity detection (capitalized spans, noun phrases, pronoun detection, demonstratives, heading terms)
- [x] **1.3** Implement `enrichment/discourse.rs` — discourse marker classification (continuation, contrast, causation, exemplification, elaboration, conclusion)
- [x] **1.4** Implement `enrichment/heading_context.rs` — heading path propagation from markdown parser
- [x] **1.5** Implement `enrichment/mod.rs` — `BlockEnvelope` construction from existing blocks + enrichment signals
- [x] **1.6** Implement `cognitive_score.rs` — weighted boundary cost function with join/break scoring (w_rel = 0.0 for now)
- [x] **1.7** Implement `cognitive_assemble.rs` — chunk formation with soft/hard token budget constraints
- [x] **1.8** Implement `diagnostics.rs` — emit full score breakdown per boundary
- [x] **1.9** Add `cognitive` CLI subcommand (`cognitive_cmd.rs`) with `--emit-signals`, `--soft-budget`, `--hard-budget`
- [x] **1.10** Add `POST /api/v1/cognitive` endpoint
- [x] **1.11** Add Python binding for cognitive mode
- [x] **1.12** Write evaluation: entity orphan rate, pronoun unresolved boundary rate, heading attachment quality
- [x] **1.13** Benchmark against current semantic mode on sample documents

### Phase 1.5 — Multilingual Support

Automatic language detection and language-appropriate enrichment heuristics.

- [x] **1.5.1** Add `whatlang` for trigram-based language detection (~70 languages)
- [x] **1.5.2** Add `unicode-script` for script property detection
- [x] **1.5.3** Implement `enrichment/language.rs` — language detection, per-language pronoun lists, demonstrative prefixes, stopwords (14 language groups)
- [x] **1.5.4** Implement `enrichment/multilingual_discourse.rs` — discourse marker tables for English, German, French, Spanish, Portuguese, Italian, Dutch, Russian, Turkish, Polish, Chinese, Japanese, Korean, Arabic
- [x] **1.5.5** Implement `enrichment/script_entities.rs` — Unicode script-based entity extraction (Katakana spans, Latin-in-CJK detection)
- [x] **1.5.6** Integrate language detection into enrichment pipeline (auto-detect from content, select appropriate heuristics)
- [x] **1.5.7** Optional: expose language override flag in CLI (`--language`) and API (`language` field)
- [x] **1.5.8** Optional: add per-language stopword filtering to capitalized-span entity extractor

### Phase 2 — Relation Awareness

The chunker becomes knowledge-graph-friendly. Relations are extracted post-assembly via LLM
(structured JSON output) rather than code-based heuristics, for high-precision results.

- [x] **2.1** ~~Implement `enrichment/relations.rs`~~ → Replaced with LLM-based extraction (`src/llm/relations.rs`)
- [x] **2.2** ~~Add relation continuity to boundary scorer~~ → Relations are post-assembly metadata, not boundary signals
- [x] **2.3** Add `--relations` flag to CLI and API (triggers LLM extraction)
- [x] **2.4** Add `dominant_relations` to chunk output metadata (up to 10 triples per chunk)
- [x] **2.5** `triple_severance_rate` evaluation metric (placeholder — requires LLM context)
- [x] **2.6** ~~Add incomplete-triple orphan risk penalty~~ → Not needed with LLM approach (relations extracted per-chunk, not across boundaries)

**Implementation:** `src/llm/mod.rs` (OpenAI-compatible completion client with `response_format: json_schema`),
`src/llm/relations.rs` (structured prompt, per-chunk extraction). Reuses existing OpenAI API key;
model configurable via `COGNIGRAPH_LLM_MODEL` env var or `.env.openai` file.

### Phase 3 — Ambiguous Boundary Reranking

Precision improvement via cross-encoder on uncertain boundaries.

- [x] **3.1** Define `RerankerProvider` trait
- [x] **3.2** Implement ONNX reranker provider (reuse `ort` infrastructure)
- [x] **3.3** Implement staged scoring: heuristic → identify ambiguous → rerank → update
- [x] **3.4** Add `--reranker` flag to CLI and API
- [ ] **3.5** Optional: add Level C entity detection (NER via ONNX model)
- [x] **3.6** Benchmark throughput impact and accuracy gain (`tests/benchmark_reranker.rs`)

### Phase 4 — Graph-Shaped Output

Chunks become retrieval-ready subgraphs of meaning.

- [x] **4.1** Add adjacency links between chunks (predecessor/successor)
- [x] **4.2** Add local concept threads (cross-chunk entity tracking)
- [x] **4.3** Machine-generated chunk synopsis via LLM (`--synopsis` flag)
- [x] **4.4** Graph export format (`--graph` flag, JSON nodes+edges)
- [ ] **4.5** Proposition-aware assembly: chunks as "claim bundles" with mutual support

---

## Evaluation Metrics

Standard metrics (keep):
- Chunk size variance
- Throughput (blocks/sec, docs/sec)

Cognition-aware metrics (new):

| Metric | Definition |
|--------|------------|
| **Entity orphan rate** | How often a split separates an entity introduction from its primary attribute or explanation |
| **Triple severance rate** | How often a subject-predicate-object candidate is split across chunk boundaries |
| **Pronoun unresolved boundary rate** | How often a chunk starts with unresolved deixis or pronouns |
| **Heading attachment quality** | How often the first explanatory sentence stays with its heading |
| **Retrieval faithfulness uplift** | Comparative answerability and retrieval relevance vs. semantic mode |

---

## Design Principles

1. **Enriched blocks, not raw sentences.** Cognitive chunking operates on blocks that carry semantic identity, not plain text spans.

2. **Valleys, not thresholds.** Boundaries are local drops in continuity, not absolute numbers. Reuse the existing signal-processing approach.

3. **Staged cost, not uniform inference.** Cheap heuristics first, expensive models only on ambiguous boundaries.

4. **Entity persistence over coreference correctness.** Full coref is brittle and expensive. Normalized noun phrases + pronoun flags + heading anchors capture 80% of the value at 1% of the complexity.

5. **Token budgets are constraints, not definitions.** Assemble around cognitive cohesion first, apply token budget as a soft constraint, degrade gracefully only under hard ceiling pressure.

6. **Observability is not optional.** Every boundary decision should be explainable via `--emit-signals`. Without this, tuning is impossible.

7. **Extend, don't replace.** The `semantic` mode stays as-is. `cognitive` is a new mode that shares infrastructure.

8. **Multilingual by default.** Language detection is automatic and transparent. The pipeline degrades gracefully — unsupported languages still get structural and embedding-based signals. No language configuration required.

---

## Key Corrections from Review

**On Cross-Encoders:** They should not be the primary scoring engine. A cross-encoder over every neighboring pair raises cost, reduces throughput, and creates calibration problems. Use staged scoring: cheap local signals first, cross-encoder only on ambiguous candidates.

**On NER and Coreference:** They help but should not be hard requirements. Full coreference resolution is brittle, expensive, and language-sensitive. Start with "entity persistence" (normalized noun phrases, named entities, heading anchors, continuation markers) rather than "coreference correctness."

---

## First Milestone ✅

> Ship a cognitive boundary scorer that reduces entity orphaning and preserves claim continuity better than plain semantic chunking.

Phase 1 is fully complete (all tasks 1.1–1.13). The cognitive mode is operational via CLI, REST API, and Python bindings, with multilingual support for 14 language groups, evaluation metrics, and benchmarks. 104 tests pass.

### Benchmark Results (OpenAI text-embedding-3-small, 5 domain-specific docs)

| Metric | Result |
|--------|--------|
| Entity orphan rate | 0% (5/5 docs) |
| Pronoun boundary rate | 0% (5/5 docs) |
| Heading attachment rate | 100% (5/5 docs) |
| Discourse break rate | 0% (5/5 docs) |
| Cognitive chunks vs semantic | ~2x more chunks, ~50% smaller avg size (budget-respecting) |
| Latency overhead | Negligible (embedding API latency dominates) |

### Bug Fix: Orphan Risk Sign

The `orphan_risk` signal was incorrectly subtracted in the join_score formula, causing breaks at heading→content and pronoun boundaries instead of preventing them. Fixed by changing `- w_orphan * orphan` to `+ w_orphan * orphan`. This improved heading attachment from 9–12% to 100%.

**Next:** Phase 3.5 (optional Level C NER) or Phase 4.5 (proposition-aware assembly).
