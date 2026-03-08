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
| **1. Structural Segmentation** | Markdown-aware block extraction (headings, tables, code, lists, quotes, sentence-split prose) | Exists |
| **2. Cognitive Enrichment** | Annotate each block with entities, relations, discourse markers, heading path | New |
| **3. Continuity Scoring** | Weighted cognitive boundary cost function replacing plain similarity threshold | New |
| **4. Constrained Assembly** | Chunk formation under soft token budgets with graceful degradation | Partially exists (merge layer) |

---

## Data Structures

### BlockEnvelope (enriched block)

```rust
pub struct BlockEnvelope {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub block_type: BlockType,          // Heading, Paragraph, Code, Table, List, Quote
    pub heading_path: Vec<String>,      // ["CogniGraph Project", "Architecture", "Scoring"]
    pub embedding: Option<Vec<f64>>,

    // Cognitive signals
    pub entities: Vec<NormalizedEntity>,
    pub entity_mentions: usize,
    pub relation_candidates: Vec<RelationCandidate>,
    pub discourse_markers: Vec<DiscourseMarker>,
    pub continuation_flags: ContinuationFlags,
    pub pronoun_lead: bool,             // block starts with pronoun/demonstrative
    pub noun_phrases: Vec<String>,
    pub dominant_concepts: Vec<String>,
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

### RelationCandidate

```rust
pub struct RelationCandidate {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub complete: bool,                 // false if object/predicate extends into next block
}
```

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
    pub dominant_relations: Vec<RelationCandidate>,
    pub parent_concept: Option<String>,
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
  + w_rel   * relation_continuity
  + w_disc  * discourse_continuation
  + w_head  * heading_context_continuity
  + w_struct * structural_affinity
  - w_shift * topic_shift_penalty
  - w_orphan * orphan_risk
  - w_budget * budget_pressure
```

Then: `break_score = 1 - normalized(join_score)`

Splits occur at **local valleys** in the join score curve, not at a flat threshold. The existing Savitzky-Golay smoothing and local-minima detection should be reused here.

### Component Definitions

| Component | Signal | Source |
|-----------|--------|--------|
| **semantic_similarity** | Cosine similarity of adjacent block embeddings | Existing |
| **entity_continuity** | Overlap of normalized entities/noun phrases/aliases between blocks | New |
| **relation_continuity** | Whether a relation triple spans the boundary (incomplete subject-predicate-object) | New (Phase 2) |
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
    pub w_rel: f64,     // 0.10  (Phase 2, 0.0 in Phase 1)
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

### Level A — Cheap Lexical (Phase 1)

- Repeated capitalized spans across blocks
- Repeated noun phrases (simple POS-like heuristic)
- Heading terms propagated downward
- Known domain entities from a user-supplied glossary (optional)
- Exact normalized string recurrence

### Level B — Structural (Phase 1)

- Apposition patterns: "X, a Y that..."
- Definitional patterns: "X is a...", "X refers to..."
- Subject repetition across sentences
- Demonstratives: "this model", "that service", "the tool"
- Pronoun-start detection: "It", "They", "This", "These", "Such"

### Level C — Model-Derived (Phase 3+, optional)

- NER via ONNX model
- Alias linking
- Light coreference resolution

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
├── mod.rs                       # Existing — semantic_chunk entry point
│
├── enrichment/
│   ├── mod.rs                   # BlockEnvelope construction
│   ├── entities.rs              # Entity extraction (Level A + B)
│   ├── relations.rs             # Lightweight relation candidates
│   ├── discourse.rs             # Discourse marker detection
│   └── heading_context.rs       # Heading path propagation
│
├── cognitive_types.rs           # All cognitive data structures
├── cognitive_score.rs           # Boundary scoring function
├── cognitive_assemble.rs        # Chunk formation with soft constraints
└── diagnostics.rs               # Score breakdown output
```

New CLI subcommand: `src/cli/cognitive_cmd.rs`
New API route: `src/api/cognitive.rs`

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

- [ ] **1.1** Define `cognitive_types.rs` — all data structures (`BlockEnvelope`, `CognitiveChunk`, `BoundaryReason`, `CognitiveWeights`, etc.)
- [ ] **1.2** Implement `enrichment/entities.rs` — Level A + B entity detection (capitalized spans, noun phrases, pronoun detection, demonstratives, heading terms)
- [ ] **1.3** Implement `enrichment/discourse.rs` — discourse marker classification (continuation, contrast, causation, exemplification, elaboration, conclusion)
- [ ] **1.4** Implement `enrichment/heading_context.rs` — heading path propagation from markdown parser
- [ ] **1.5** Implement `enrichment/mod.rs` — `BlockEnvelope` construction from existing blocks + enrichment signals
- [ ] **1.6** Implement `cognitive_score.rs` — weighted boundary cost function with join/break scoring (w_rel = 0.0 for now)
- [ ] **1.7** Implement `cognitive_assemble.rs` — chunk formation with soft/hard token budget constraints
- [ ] **1.8** Implement `diagnostics.rs` — emit full score breakdown per boundary
- [ ] **1.9** Add `cognitive` CLI subcommand (`cognitive_cmd.rs`) with `--lite`, `--emit-signals`, `--soft-budget`, `--hard-budget`
- [ ] **1.10** Add `POST /api/v1/cognitive` endpoint
- [ ] **1.11** Add Python binding for cognitive mode
- [ ] **1.12** Write evaluation: entity orphan rate, pronoun unresolved boundary rate, heading attachment quality
- [ ] **1.13** Benchmark against current semantic mode on sample documents

### Phase 2 — Relation Awareness

The chunker becomes knowledge-graph-friendly.

- [ ] **2.1** Implement `enrichment/relations.rs` — lightweight triple extraction (subject-verb-object patterns, copular definitions, "X uses Y" etc.)
- [ ] **2.2** Add relation continuity to boundary scorer (enable w_rel)
- [ ] **2.3** Add `--relations` flag to CLI and API
- [ ] **2.4** Add `dominant_relations` to chunk output metadata
- [ ] **2.5** Measure triple severance rate before/after
- [ ] **2.6** Add incomplete-triple orphan risk penalty

### Phase 3 — Ambiguous Boundary Reranking

Precision improvement via cross-encoder on uncertain boundaries.

- [ ] **3.1** Define `RerankerProvider` trait
- [ ] **3.2** Implement ONNX reranker provider (reuse `ort` infrastructure)
- [ ] **3.3** Implement staged scoring: heuristic → identify ambiguous → rerank → update
- [ ] **3.4** Add `--reranker` flag to CLI and API
- [ ] **3.5** Optional: add Level C entity detection (NER via ONNX model)
- [ ] **3.6** Benchmark throughput impact and accuracy gain

### Phase 4 — Graph-Shaped Output

Chunks become retrieval-ready subgraphs of meaning.

- [ ] **4.1** Add adjacency links between chunks (predecessor/successor)
- [ ] **4.2** Add local concept threads (cross-chunk entity tracking)
- [ ] **4.3** Optional machine-generated chunk synopsis
- [ ] **4.4** Graph export format (JSON-LD or custom)
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

---

## Key Corrections from Review

**On Cross-Encoders:** They should not be the primary scoring engine. A cross-encoder over every neighboring pair raises cost, reduces throughput, and creates calibration problems. Use staged scoring: cheap local signals first, cross-encoder only on ambiguous candidates.

**On NER and Coreference:** They help but should not be hard requirements. Full coreference resolution is brittle, expensive, and language-sensitive. Start with "entity persistence" (normalized noun phrases, named entities, heading anchors, continuation markers) rather than "coreference correctness."

---

## First Milestone

> Ship a cognitive boundary scorer that reduces entity orphaning and preserves claim continuity better than plain semantic chunking.

This is Phase 1. It is concrete, measurable, and achievable with the current foundation.
