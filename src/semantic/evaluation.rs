//! Evaluation metrics for cognitive chunking quality.
//!
//! Measures how well chunk boundaries preserve cognitive coherence:
//! - Entity orphan rate: entities split from their explanatory context
//! - Pronoun boundary rate: chunks starting with unresolved pronouns/demonstratives
//! - Heading attachment quality: headings staying with their first content block

use super::blocks::BlockKind;
use super::cognitive_types::{BlockEnvelope, BoundarySignal, CognitiveChunk};

/// Evaluation metrics for a cognitive chunking result.
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    /// Fraction of chunk boundaries that separate entities shared across the boundary (0.0–1.0).
    /// Lower is better.
    pub entity_orphan_rate: f64,
    /// Fraction of chunks that start with a pronoun ("It", "They") or demonstrative ("This X").
    /// Lower is better — these indicate the chunk depends on prior context.
    pub pronoun_boundary_rate: f64,
    /// Fraction of headings that remain attached to their first content block.
    /// Higher is better.
    pub heading_attachment_rate: f64,
    /// Fraction of chunks starting with a discourse continuation marker ("Furthermore", "Moreover").
    /// Lower is better — these indicate the chunk continues a thought from before.
    pub discourse_break_rate: f64,
    /// Fraction of break boundaries that sever a relation triple (subject in A, object in B).
    /// Lower is better.
    pub triple_severance_rate: f64,
    /// Total number of chunks evaluated.
    pub chunk_count: usize,
    /// Total number of blocks.
    pub block_count: usize,
}

impl Default for EvaluationMetrics {
    fn default() -> Self {
        Self {
            entity_orphan_rate: 0.0,
            pronoun_boundary_rate: 0.0,
            heading_attachment_rate: 1.0,
            discourse_break_rate: 0.0,
            triple_severance_rate: 0.0,
            chunk_count: 0,
            block_count: 0,
        }
    }
}

/// Compute evaluation metrics from enriched blocks, boundary signals, and assembled chunks.
pub fn evaluate(
    blocks: &[BlockEnvelope],
    signals: &[BoundarySignal],
    chunks: &[CognitiveChunk],
) -> EvaluationMetrics {
    let chunk_count = chunks.len();
    let block_count = blocks.len();

    if chunk_count == 0 || block_count == 0 {
        return EvaluationMetrics {
            entity_orphan_rate: 0.0,
            pronoun_boundary_rate: 0.0,
            heading_attachment_rate: 1.0,
            discourse_break_rate: 0.0,
            triple_severance_rate: 0.0,
            chunk_count,
            block_count,
        };
    }

    let entity_orphan_rate = compute_entity_orphan_rate(signals);
    let pronoun_boundary_rate = compute_pronoun_boundary_rate(blocks, signals);
    let heading_attachment_rate = compute_heading_attachment_rate(blocks, signals);
    let discourse_break_rate = compute_discourse_break_rate(blocks, signals);
    // Triple severance rate computed post-LLM extraction, not during assembly
    let triple_severance_rate = 0.0;

    EvaluationMetrics {
        entity_orphan_rate,
        pronoun_boundary_rate,
        heading_attachment_rate,
        discourse_break_rate,
        triple_severance_rate,
        chunk_count,
        block_count,
    }
}

/// Entity orphan rate: fraction of actual breaks where entities are shared across the boundary.
fn compute_entity_orphan_rate(signals: &[BoundarySignal]) -> f64 {
    let breaks: Vec<_> = signals.iter().filter(|s| s.is_break).collect();
    if breaks.is_empty() {
        return 0.0;
    }

    let orphaned = breaks.iter().filter(|s| s.entity_continuity > 0.2).count();

    orphaned as f64 / breaks.len() as f64
}

/// Pronoun boundary rate: fraction of chunks whose first block starts with a pronoun or demonstrative.
fn compute_pronoun_boundary_rate(blocks: &[BlockEnvelope], signals: &[BoundarySignal]) -> f64 {
    // Find which blocks start chunks (block 0 always starts a chunk, plus blocks after each break)
    let mut chunk_start_indices: Vec<usize> = vec![0];
    for signal in signals {
        if signal.is_break {
            chunk_start_indices.push(signal.index + 1);
        }
    }

    if chunk_start_indices.len() <= 1 {
        return 0.0;
    }

    // Skip the first chunk (it always starts the document, no prior context)
    let non_first_starts = &chunk_start_indices[1..];

    let pronoun_starts = non_first_starts
        .iter()
        .filter(|&&idx| {
            if let Some(block) = blocks.get(idx) {
                block.continuation_flags.starts_with_pronoun
                    || block.continuation_flags.starts_with_demonstrative
            } else {
                false
            }
        })
        .count();

    pronoun_starts as f64 / non_first_starts.len() as f64
}

/// Heading attachment rate: fraction of headings that are NOT separated from their next content block.
fn compute_heading_attachment_rate(blocks: &[BlockEnvelope], signals: &[BoundarySignal]) -> f64 {
    // Find all headings that have a following content block
    let mut heading_count = 0;
    let mut attached_count = 0;

    for i in 0..blocks.len().saturating_sub(1) {
        if blocks[i].block_type == BlockKind::Heading
            && blocks[i + 1].block_type != BlockKind::Heading
        {
            heading_count += 1;
            // Check if the boundary between this heading and next block is NOT a break
            if let Some(signal) = signals.get(i)
                && !signal.is_break
            {
                attached_count += 1;
            }
        }
    }

    if heading_count == 0 {
        return 1.0;
    }

    attached_count as f64 / heading_count as f64
}

/// Discourse break rate: fraction of chunk boundaries that split a discourse continuation.
fn compute_discourse_break_rate(blocks: &[BlockEnvelope], signals: &[BoundarySignal]) -> f64 {
    let mut chunk_start_indices: Vec<usize> = vec![0];
    for signal in signals {
        if signal.is_break {
            chunk_start_indices.push(signal.index + 1);
        }
    }

    if chunk_start_indices.len() <= 1 {
        return 0.0;
    }

    let non_first_starts = &chunk_start_indices[1..];

    let discourse_starts = non_first_starts
        .iter()
        .filter(|&&idx| {
            blocks
                .get(idx)
                .is_some_and(|b| b.continuation_flags.starts_with_discourse)
        })
        .count();

    discourse_starts as f64 / non_first_starts.len() as f64
}

/// Format evaluation metrics as a human-readable summary.
pub fn format_metrics(metrics: &EvaluationMetrics) -> String {
    format!(
        "Evaluation ({} blocks → {} chunks):\n  \
         Entity orphan rate:      {:.1}%\n  \
         Pronoun boundary rate:   {:.1}%\n  \
         Heading attachment rate: {:.1}%\n  \
         Discourse break rate:    {:.1}%\n  \
         Triple severance rate:   {:.1}%",
        metrics.block_count,
        metrics.chunk_count,
        metrics.entity_orphan_rate * 100.0,
        metrics.pronoun_boundary_rate * 100.0,
        metrics.heading_attachment_rate * 100.0,
        metrics.discourse_break_rate * 100.0,
        metrics.triple_severance_rate * 100.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::cognitive_types::ContinuationFlags;

    fn make_block(text: &str, kind: BlockKind) -> BlockEnvelope {
        BlockEnvelope {
            text: text.to_string(),
            offset_start: 0,
            offset_end: text.len(),
            block_type: kind,
            heading_path: vec![],
            embedding: None,
            entities: vec![],
            noun_phrases: vec![],
            discourse_markers: vec![],
            continuation_flags: ContinuationFlags::default(),
            token_estimate: text.len() / 4,
        }
    }

    fn make_signal(index: usize, is_break: bool, entity_cont: f64) -> BoundarySignal {
        BoundarySignal {
            index,
            semantic_similarity: 0.5,
            entity_continuity: entity_cont,
            relation_continuity: 0.0,
            discourse_continuation: 0.0,
            heading_continuity: 1.0,
            structural_affinity: 0.0,
            topic_shift_penalty: 0.5,
            orphan_risk: 0.0,
            budget_pressure: 0.0,
            join_score: 0.3,
            is_break,
            reasons: vec![],
        }
    }

    #[test]
    fn test_entity_orphan_rate() {
        let signals = vec![
            make_signal(0, true, 0.5),  // break with shared entities
            make_signal(1, false, 0.8), // no break
            make_signal(2, true, 0.0),  // break without shared entities
        ];
        let rate = compute_entity_orphan_rate(&signals);
        assert!(
            (rate - 0.5).abs() < 0.01,
            "1 of 2 breaks has entity overlap: {rate}"
        );
    }

    #[test]
    fn test_pronoun_boundary_rate() {
        let mut blocks = vec![
            make_block("The system processes text.", BlockKind::Sentence),
            make_block("It also supports chunking.", BlockKind::Sentence),
            make_block("The output is JSON.", BlockKind::Sentence),
        ];
        blocks[1].continuation_flags.starts_with_pronoun = true;

        let signals = vec![
            make_signal(0, true, 0.0), // break before "It also..."
            make_signal(1, false, 0.0),
        ];

        let rate = compute_pronoun_boundary_rate(&blocks, &signals);
        assert!(
            (rate - 1.0).abs() < 0.01,
            "100% of non-first chunk starts have pronoun: {rate}"
        );
    }

    #[test]
    fn test_heading_attachment() {
        let blocks = vec![
            make_block("## Architecture", BlockKind::Heading),
            make_block("The system is modular.", BlockKind::Sentence),
            make_block("## Scoring", BlockKind::Heading),
            make_block("Scores are computed.", BlockKind::Sentence),
        ];

        // First heading attached, second heading detached
        let signals = vec![
            make_signal(0, false, 0.0), // heading stays with content
            make_signal(1, false, 0.0),
            make_signal(2, true, 0.0), // break separates heading from content!
        ];

        let rate = compute_heading_attachment_rate(&blocks, &signals);
        assert!(
            (rate - 0.5).abs() < 0.01,
            "1 of 2 headings attached: {rate}"
        );
    }

    #[test]
    fn test_perfect_metrics() {
        let blocks = vec![
            make_block("## Section", BlockKind::Heading),
            make_block("Content here.", BlockKind::Sentence),
            make_block("More content.", BlockKind::Sentence),
        ];
        let signals = vec![
            make_signal(0, false, 0.0),
            make_signal(1, true, 0.0), // clean break, no entities, no pronouns
        ];
        let chunks = vec![CognitiveChunk {
            text: "test".to_string(),
            chunk_index: 0,
            offset_start: 0,
            offset_end: 10,
            heading_path: vec![],
            dominant_entities: vec![],
            dominant_relations: vec![],
            token_estimate: 10,
            continuity_confidence: 1.0,
            boundary_reasons_start: vec![],
            boundary_reasons_end: vec![],
            synopsis: None,
            prev_chunk: None,
            next_chunk: None,
        }];

        let metrics = evaluate(&blocks, &signals, &chunks);
        assert_eq!(metrics.entity_orphan_rate, 0.0);
        assert_eq!(metrics.pronoun_boundary_rate, 0.0);
        assert_eq!(metrics.heading_attachment_rate, 1.0);
        assert_eq!(metrics.discourse_break_rate, 0.0);
    }
}
