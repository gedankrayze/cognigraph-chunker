//! Cognitive boundary scoring.
//!
//! Computes a join score between adjacent blocks using weighted signals:
//! semantic similarity, entity continuity, discourse markers, heading context,
//! structural affinity, orphan risk, and budget pressure.

use super::blocks::BlockKind;
use super::cognitive_types::{BlockEnvelope, BoundarySignal, CognitiveWeights};
use super::enrichment::discourse::discourse_continuation_score;
use super::enrichment::entities::{entity_overlap, noun_phrase_overlap};
use super::enrichment::heading_context::heading_continuity;

/// Compute boundary signals for all adjacent block pairs.
///
/// Returns one `BoundarySignal` per boundary (n-1 for n blocks).
/// `similarities` is the raw semantic similarity curve from embedding comparison.
pub fn score_boundaries(
    blocks: &[BlockEnvelope],
    similarities: &[f64],
    weights: &CognitiveWeights,
    soft_budget: usize,
) -> Vec<BoundarySignal> {
    let n = blocks.len();
    if n < 2 {
        return vec![];
    }

    let mut signals = Vec::with_capacity(n - 1);
    let mut accumulated_tokens: usize = blocks[0].token_estimate;

    for i in 0..n - 1 {
        let a = &blocks[i];
        let b = &blocks[i + 1];

        // Semantic similarity (from embedding-based curve)
        let semantic_sim = similarities.get(i).copied().unwrap_or(0.5);

        // Entity continuity: overlap of named entities + noun phrases
        let ent_overlap = entity_overlap(&a.entities, &b.entities);
        let np_overlap = noun_phrase_overlap(&a.noun_phrases, &b.noun_phrases);
        let entity_cont = (ent_overlap + np_overlap) / 2.0;

        // Discourse continuation
        let discourse_cont = discourse_continuation_score(&b.discourse_markers);

        // Heading context continuity
        let heading_cont = heading_continuity(&a.heading_path, &b.heading_path);

        // Relation continuity: placeholder (relations extracted post-assembly via LLM)
        let relation_cont = 0.0;

        // Structural affinity: known cohesive patterns
        let struct_affinity = structural_affinity_score(a.block_type, b.block_type);

        // Topic shift penalty (inverse of semantic similarity)
        let topic_shift = 1.0 - semantic_sim;

        // Orphan risk: splitting would leave dangling references
        let orphan = orphan_risk_score(a, b);

        // Budget pressure: increases as accumulated tokens approach soft budget
        accumulated_tokens += b.token_estimate;
        let budget_press = if soft_budget > 0 {
            (accumulated_tokens as f64 / soft_budget as f64 - 0.5).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Weighted join score
        // Positive terms = reasons to keep together (join)
        // Negative terms = reasons to split (break)
        let join_score = weights.w_sem * semantic_sim
            + weights.w_ent * entity_cont
            + weights.w_rel * relation_cont
            + weights.w_disc * discourse_cont
            + weights.w_head * heading_cont
            + weights.w_struct * struct_affinity
            + weights.w_orphan * orphan // high orphan risk = discourage breaking
            - weights.w_shift * topic_shift
            - weights.w_budget * budget_press;

        // Collect reasons
        let mut reasons = Vec::new();
        if orphan > 0.5 {
            reasons.push("high orphan risk".to_string());
        }
        if discourse_cont > 0.7 {
            reasons.push(format!("discourse continuation ({discourse_cont:.2})"));
        }
        if entity_cont > 0.5 {
            reasons.push(format!("entity continuity ({entity_cont:.2})"));
        }
        if relation_cont > 0.3 {
            reasons.push(format!("relation continuity ({relation_cont:.2})"));
        }
        if heading_cont < 0.5 {
            reasons.push("heading change".to_string());
        }
        if topic_shift > 0.6 {
            reasons.push(format!("topic shift ({topic_shift:.2})"));
        }
        if budget_press > 0.5 {
            reasons.push(format!("budget pressure ({accumulated_tokens} tokens)"));
        }

        signals.push(BoundarySignal {
            index: i,
            semantic_similarity: semantic_sim,
            entity_continuity: entity_cont,
            relation_continuity: relation_cont,
            discourse_continuation: discourse_cont,
            heading_continuity: heading_cont,
            structural_affinity: struct_affinity,
            topic_shift_penalty: topic_shift,
            orphan_risk: orphan,
            budget_pressure: budget_press,
            join_score,
            is_break: false, // Will be determined by assembler
            reasons,
        });

        // Reset accumulator on break (will be updated by assembler)
        // For now, accumulate linearly — assembler will use final break decisions.
    }

    signals
}

/// Score structural affinity between adjacent block types (0.0–1.0).
///
/// Higher = these block types naturally belong together.
fn structural_affinity_score(a: BlockKind, b: BlockKind) -> f64 {
    match (a, b) {
        // Heading followed by its first content block — very strong affinity
        (BlockKind::Heading, BlockKind::Sentence) => 0.9,
        (BlockKind::Heading, BlockKind::List) => 0.85,
        (BlockKind::Heading, BlockKind::Table) => 0.8,
        (BlockKind::Heading, BlockKind::CodeBlock) => 0.8,

        // Paragraph followed by an illustrative block
        (BlockKind::Sentence, BlockKind::CodeBlock) => 0.6,
        (BlockKind::Sentence, BlockKind::Table) => 0.6,
        (BlockKind::Sentence, BlockKind::List) => 0.5,

        // Consecutive sentences — neutral (let other signals decide)
        (BlockKind::Sentence, BlockKind::Sentence) => 0.0,

        // Code/table followed by explanation
        (BlockKind::CodeBlock, BlockKind::Sentence) => 0.4,
        (BlockKind::Table, BlockKind::Sentence) => 0.4,

        // New heading after any block — low affinity (natural break point)
        (_, BlockKind::Heading) => 0.0,

        _ => 0.0,
    }
}

/// Score orphan risk for a boundary between blocks a and b (0.0–1.0).
///
/// Higher = splitting here would orphan references or incomplete propositions.
fn orphan_risk_score(a: &BlockEnvelope, b: &BlockEnvelope) -> f64 {
    let mut risk = 0.0;

    // Block B starts with a pronoun — it refers back to A
    if b.continuation_flags.starts_with_pronoun {
        risk += 0.8;
    }

    // Block B starts with a demonstrative — it refers to something in A
    if b.continuation_flags.starts_with_demonstrative {
        risk += 0.6;
    }

    // Block B has a discourse marker indicating continuation
    if b.continuation_flags.starts_with_discourse {
        risk += 0.4;
    }

    // Heading A followed by content B — orphaning the heading
    if a.block_type == BlockKind::Heading && b.block_type != BlockKind::Heading {
        risk += 0.9;
    }

    // Block A introduces entities that B immediately references
    let shared_entities = entity_overlap(&a.entities, &b.entities);
    if shared_entities > 0.3 {
        risk += shared_entities * 0.3;
    }

    risk.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::cognitive_types::ContinuationFlags;

    fn make_block(text: &str, kind: BlockKind, heading_path: Vec<String>) -> BlockEnvelope {
        use crate::semantic::enrichment::discourse::detect_discourse_markers;
        use crate::semantic::enrichment::entities::{
            extract_entities, extract_noun_phrases, starts_with_demonstrative, starts_with_pronoun,
        };
        use crate::semantic::enrichment::language::{LanguageGroup, stopwords_for};
        use std::collections::HashSet;

        let sw = stopwords_for(LanguageGroup::English);
        let entities = extract_entities(text, &HashSet::new(), sw);
        let noun_phrases = extract_noun_phrases(text, sw);
        let discourse_markers = detect_discourse_markers(text);
        let continuation_flags = ContinuationFlags {
            starts_with_pronoun: starts_with_pronoun(text),
            starts_with_demonstrative: starts_with_demonstrative(text),
            starts_with_discourse: !discourse_markers.is_empty(),
            continues_list: false,
        };

        BlockEnvelope {
            text: text.to_string(),
            offset_start: 0,
            offset_end: text.len(),
            block_type: kind,
            heading_path,
            embedding: None,
            entities,
            noun_phrases,
            discourse_markers,
            continuation_flags,
            token_estimate: text.len() / 4,
        }
    }

    #[test]
    fn test_heading_content_affinity() {
        let blocks = vec![
            make_block(
                "## Architecture",
                BlockKind::Heading,
                vec!["Architecture".into()],
            ),
            make_block(
                "The architecture is modular.",
                BlockKind::Sentence,
                vec!["Architecture".into()],
            ),
        ];
        let sims = vec![0.5];
        let signals = score_boundaries(&blocks, &sims, &CognitiveWeights::default(), 512);
        assert_eq!(signals.len(), 1);
        // High join score: heading+content affinity + heading continuity + orphan risk penalty
        assert!(
            signals[0].join_score > 0.0,
            "Heading-content should have positive join score"
        );
    }

    #[test]
    fn test_pronoun_orphan_risk() {
        let blocks = vec![
            make_block(
                "The CogniGraph Chunker processes text efficiently.",
                BlockKind::Sentence,
                vec![],
            ),
            make_block(
                "It also supports multiple providers.",
                BlockKind::Sentence,
                vec![],
            ),
        ];
        let sims = vec![0.7];
        let signals = score_boundaries(&blocks, &sims, &CognitiveWeights::default(), 512);
        assert!(
            signals[0].orphan_risk > 0.5,
            "Pronoun start should raise orphan risk"
        );
    }

    #[test]
    fn test_topic_shift() {
        let blocks = vec![
            make_block(
                "Machine learning is a broad field.",
                BlockKind::Sentence,
                vec!["ML".into()],
            ),
            make_block(
                "Cooking requires patience and skill.",
                BlockKind::Sentence,
                vec!["Cooking".into()],
            ),
        ];
        let sims = vec![0.1]; // Very low similarity
        let signals = score_boundaries(&blocks, &sims, &CognitiveWeights::default(), 512);
        assert!(
            signals[0].topic_shift_penalty > 0.8,
            "Low similarity should produce high shift penalty"
        );
    }

    #[test]
    fn test_discourse_continuation() {
        let blocks = vec![
            make_block("The system uses embeddings.", BlockKind::Sentence, vec![]),
            make_block(
                "Furthermore, it supports reranking.",
                BlockKind::Sentence,
                vec![],
            ),
        ];
        let sims = vec![0.6];
        let signals = score_boundaries(&blocks, &sims, &CognitiveWeights::default(), 512);
        assert!(
            signals[0].discourse_continuation > 0.8,
            "Furthermore should signal strong continuation"
        );
    }
}
