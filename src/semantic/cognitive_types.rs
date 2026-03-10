//! Data types for cognition-aware chunking.

use super::blocks::BlockKind;

// ── Block-level enrichment ──────────────────────────────────────────

/// A block enriched with cognitive signals for boundary scoring.
#[derive(Debug, Clone)]
pub struct BlockEnvelope {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub block_type: BlockKind,
    /// Heading ancestry path, e.g. ["Architecture", "Scoring"].
    pub heading_path: Vec<String>,
    /// Embedding vector (populated during pipeline).
    pub embedding: Option<Vec<f64>>,

    // Cognitive signals
    pub entities: Vec<NormalizedEntity>,
    pub noun_phrases: Vec<String>,
    pub discourse_markers: Vec<DiscourseMarker>,
    pub continuation_flags: ContinuationFlags,
    pub token_estimate: usize,
}

/// An entity mention normalized for overlap detection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NormalizedEntity {
    /// Original surface text, e.g. "the chunker", "CogniGraph".
    pub surface_form: String,
    /// Lowercased key for matching, e.g. "cognigraph".
    pub normalized: String,
    /// Evidence level.
    pub entity_type: EntityType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    /// Capitalized multi-word span, e.g. "CogniGraph Chunker".
    Named,
    /// Repeated noun phrase, e.g. "the chunker".
    NounPhrase,
    /// Pronoun reference, e.g. "it", "they".
    Pronoun,
    /// Demonstrative reference, e.g. "this model", "these results".
    Demonstrative,
}

/// A discourse marker detected at the start of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscourseMarker {
    /// "furthermore", "additionally", "also", "moreover"
    Continuation,
    /// "however", "but", "on the other hand", "in contrast"
    Contrast,
    /// "therefore", "because", "thus", "consequently"
    Causation,
    /// "for example", "such as", "e.g.", "for instance"
    Exemplification,
    /// "specifically", "in particular", "namely"
    Elaboration,
    /// "in summary", "finally", "in conclusion"
    Conclusion,
}

/// Flags indicating whether this block is likely a continuation of the previous.
#[derive(Debug, Clone, Default)]
pub struct ContinuationFlags {
    /// Block starts with a pronoun ("It", "They", "He", "She").
    pub starts_with_pronoun: bool,
    /// Block starts with a demonstrative ("This model", "These results", "Such systems").
    pub starts_with_demonstrative: bool,
    /// Block starts with a discourse marker.
    pub starts_with_discourse: bool,
    /// Block continues a numbered or bulleted list.
    pub continues_list: bool,
}

impl ContinuationFlags {
    /// True if any continuation signal is present.
    pub fn any(&self) -> bool {
        self.starts_with_pronoun
            || self.starts_with_demonstrative
            || self.starts_with_discourse
            || self.continues_list
    }
}

// ── Boundary scoring ────────────────────────────────────────────────

/// Weights for the cognitive boundary cost function.
#[derive(Debug, Clone)]
pub struct CognitiveWeights {
    pub w_sem: f64,
    pub w_ent: f64,
    pub w_rel: f64,
    pub w_disc: f64,
    pub w_head: f64,
    pub w_struct: f64,
    pub w_shift: f64,
    pub w_orphan: f64,
    pub w_budget: f64,
}

impl Default for CognitiveWeights {
    fn default() -> Self {
        Self {
            w_sem: 0.30,
            w_ent: 0.20,
            w_rel: 0.0, // Relations extracted post-assembly via LLM
            w_disc: 0.10,
            w_head: 0.10,
            w_struct: 0.05,
            w_shift: 0.15,
            w_orphan: 0.20,
            w_budget: 0.10,
        }
    }
}

/// Configuration for cognitive chunking.
#[derive(Debug, Clone)]
pub struct CognitiveConfig {
    /// Weights for the boundary scoring function.
    pub weights: CognitiveWeights,
    /// Soft token budget per chunk — assembly prefers to stay under this.
    pub soft_budget: usize,
    /// Hard token ceiling — never exceed unless a single block is larger.
    pub hard_budget: usize,
    /// Window size for embedding similarity (must be odd, >= 3).
    pub sim_window: usize,
    /// Savitzky-Golay smoothing window (must be odd).
    pub sg_window: usize,
    /// Savitzky-Golay polynomial order.
    pub poly_order: usize,
    /// Maximum block count (O(n²) protection).
    pub max_blocks: usize,
    /// Emit full diagnostic signals.
    pub emit_signals: bool,
    /// Language override. `None` = auto-detect from content.
    pub language: Option<super::enrichment::language::LanguageGroup>,
}

impl Default for CognitiveConfig {
    fn default() -> Self {
        Self {
            weights: CognitiveWeights::default(),
            soft_budget: 512,
            hard_budget: 768,
            sim_window: 3,
            sg_window: 11,
            poly_order: 3,
            max_blocks: 10_000,
            emit_signals: false,
            language: None,
        }
    }
}

/// Detailed score breakdown for a single boundary between blocks i and i+1.
#[derive(Debug, Clone)]
pub struct BoundarySignal {
    /// Index of the boundary (between block i and block i+1).
    pub index: usize,
    pub semantic_similarity: f64,
    pub entity_continuity: f64,
    pub relation_continuity: f64,
    pub discourse_continuation: f64,
    pub heading_continuity: f64,
    pub structural_affinity: f64,
    pub topic_shift_penalty: f64,
    pub orphan_risk: f64,
    pub budget_pressure: f64,
    /// Weighted join score (higher = keep together).
    pub join_score: f64,
    /// Whether this boundary was selected as a split.
    pub is_break: bool,
    /// Human-readable reasons for the decision.
    pub reasons: Vec<String>,
}

/// Reason why a chunk boundary was placed or suppressed.
#[derive(Debug, Clone)]
pub enum BoundaryReason {
    TopicShift { similarity_drop: f64 },
    HeadingChange { from: String, to: String },
    EntityDiscontinuity { orphaned: Vec<String> },
    PropositionComplete,
    BudgetCeiling { tokens: usize },
    DiscourseBreak,
    StructuralBarrier { block_type: BlockKind },
    ContinuationGlue { flags: String },
}

// ── Chunk output ────────────────────────────────────────────────────

/// A cognitive chunk with rich metadata.
#[derive(Debug, Clone)]
pub struct CognitiveChunk {
    /// The chunk text.
    pub text: String,
    /// Zero-based chunk index within the result.
    pub chunk_index: usize,
    /// Byte offset in source document.
    pub offset_start: usize,
    pub offset_end: usize,
    /// Heading ancestry for this chunk.
    pub heading_path: Vec<String>,
    /// Dominant entities in this chunk.
    pub dominant_entities: Vec<String>,
    /// All normalized entities encountered in the chunk.
    pub all_entities: Vec<String>,
    /// Dominant relation triples in this chunk (populated by LLM post-assembly).
    pub dominant_relations: Vec<crate::llm::relations::RelationTriple>,
    /// Estimated token count.
    pub token_estimate: usize,
    /// Confidence that boundaries are correct (average join score within chunk).
    pub continuity_confidence: f64,
    /// Why the chunk starts here.
    pub boundary_reasons_start: Vec<BoundaryReason>,
    /// Why the chunk ends here.
    pub boundary_reasons_end: Vec<BoundaryReason>,
    /// LLM-generated 1-2 sentence summary (populated when --synopsis flag is used).
    pub synopsis: Option<String>,
    /// Index of the previous chunk, if any.
    pub prev_chunk: Option<usize>,
    /// Index of the next chunk, if any.
    pub next_chunk: Option<usize>,
}

/// Result of cognitive chunking, including optional diagnostic signals.
#[derive(Debug)]
pub struct CognitiveResult {
    /// The assembled chunks.
    pub chunks: Vec<CognitiveChunk>,
    /// Per-boundary diagnostic signals (only populated when emit_signals is true).
    pub signals: Vec<BoundarySignal>,
    /// Number of blocks processed.
    pub block_count: usize,
    /// Evaluation metrics (always computed).
    pub evaluation: super::evaluation::EvaluationMetrics,
    /// Cross-chunk entity tracking: entity name → list of chunk indices where it appears.
    pub shared_entities: std::collections::HashMap<String, Vec<usize>>,
}
