//! Proposition-aware chunk healing.
//!
//! After initial assembly, scans chunk boundaries for incomplete propositions
//! and merges chunks that would be more coherent together. A "broken proposition"
//! is a chunk boundary where:
//!
//! - The second chunk starts with an unresolved pronoun or demonstrative
//! - The second chunk starts with a discourse continuation marker
//! - The two chunks share high entity overlap (same entities, likely same argument)
//!
//! Merging is only performed if the combined chunk stays within the hard budget.

use super::cognitive_types::{BlockEnvelope, BoundarySignal, CognitiveChunk};

/// Result of the proposition healing pass.
#[derive(Debug)]
pub struct HealResult {
    /// Number of merges performed.
    pub merges: usize,
    /// Descriptions of each merge for diagnostics.
    pub merge_reasons: Vec<String>,
}

/// Heal broken propositions by merging chunks with unresolved dependencies.
///
/// Operates on the assembled chunks and original signals/blocks.
/// Returns the healed chunk list and merge diagnostics.
pub fn heal_proposition_breaks(
    chunks: Vec<CognitiveChunk>,
    blocks: &[BlockEnvelope],
    signals: &[BoundarySignal],
    hard_budget: usize,
) -> (Vec<CognitiveChunk>, HealResult) {
    if chunks.len() <= 1 {
        return (
            chunks,
            HealResult {
                merges: 0,
                merge_reasons: vec![],
            },
        );
    }

    let mut result: Vec<CognitiveChunk> = Vec::with_capacity(chunks.len());
    let mut merge_reasons: Vec<String> = Vec::new();
    let mut merges = 0;

    let mut i = 0;
    while i < chunks.len() {
        let mut current = chunks[i].clone();

        // Look ahead: should we merge with the next chunk?
        while i + 1 < chunks.len() {
            let next = &chunks[i + 1];
            let combined_tokens = current.token_estimate + next.token_estimate;

            // Don't exceed hard budget
            if combined_tokens > hard_budget {
                break;
            }

            // Find the boundary signal between these chunks
            let boundary_block_idx = find_boundary_block_index(&current, blocks);
            let reason = if let Some(idx) = boundary_block_idx {
                diagnose_broken_proposition(idx, blocks, signals)
            } else {
                None
            };

            if let Some(reason) = reason {
                merge_reasons.push(format!(
                    "Merged chunk {} ← chunk {}: {}",
                    current.chunk_index, next.chunk_index, reason
                ));
                current = merge_chunks(&current, next);
                merges += 1;
                i += 1;
            } else {
                break;
            }
        }

        result.push(current);
        i += 1;
    }

    (
        result,
        HealResult {
            merges,
            merge_reasons,
        },
    )
}

/// Find the block index where a chunk ends (the boundary between this chunk and the next).
fn find_boundary_block_index(chunk: &CognitiveChunk, blocks: &[BlockEnvelope]) -> Option<usize> {
    // The chunk's offset_end matches the end of its last block.
    // Find the block whose offset_end matches.
    blocks.iter().position(|b| b.offset_end == chunk.offset_end)
}

/// Check if the boundary after block `idx` represents a broken proposition.
/// Returns a reason string if it does, None otherwise.
fn diagnose_broken_proposition(
    boundary_idx: usize,
    blocks: &[BlockEnvelope],
    signals: &[BoundarySignal],
) -> Option<String> {
    // The signal at `boundary_idx` represents the boundary between blocks[boundary_idx] and blocks[boundary_idx+1]
    if boundary_idx >= signals.len() || boundary_idx + 1 >= blocks.len() {
        return None;
    }

    let next_block = &blocks[boundary_idx + 1];
    let signal = &signals[boundary_idx];

    // Only heal boundaries that were actual breaks
    if !signal.is_break {
        return None;
    }

    // Check 1: Next chunk starts with unresolved pronoun
    if next_block.continuation_flags.starts_with_pronoun {
        return Some("unresolved pronoun at chunk start".to_string());
    }

    // Check 2: Next chunk starts with demonstrative
    if next_block.continuation_flags.starts_with_demonstrative {
        return Some("unresolved demonstrative at chunk start".to_string());
    }

    // Check 3: Next chunk starts with a strong discourse continuation
    // (continuation, causation, or elaboration markers — not contrast which can start new ideas)
    if next_block.continuation_flags.starts_with_discourse {
        let has_strong_continuation = next_block.discourse_markers.iter().any(|m| {
            matches!(
                m,
                super::cognitive_types::DiscourseMarker::Continuation
                    | super::cognitive_types::DiscourseMarker::Causation
                    | super::cognitive_types::DiscourseMarker::Elaboration
            )
        });
        if has_strong_continuation {
            return Some("discourse continuation at chunk start".to_string());
        }
    }

    // Check 4: High entity overlap (same subject under discussion)
    // If entity_continuity > 0.5, the same entities dominate both sides
    if signal.entity_continuity > 0.5 && signal.semantic_similarity > 0.7 {
        return Some(format!(
            "high entity continuity ({:.2}) with similar topic ({:.2})",
            signal.entity_continuity, signal.semantic_similarity
        ));
    }

    None
}

/// Merge two adjacent chunks into one.
fn merge_chunks(a: &CognitiveChunk, b: &CognitiveChunk) -> CognitiveChunk {
    let mut text = a.text.clone();
    text.push_str(&b.text);

    // Merge dominant entities: combine, re-sort by occurrence would require counts,
    // so take union of both top-5 lists, capped at 5
    let mut dominant = a.dominant_entities.clone();
    for e in &b.dominant_entities {
        if !dominant.contains(e) {
            dominant.push(e.clone());
        }
    }
    dominant.truncate(5);

    // Merge all_entities: sorted union
    let mut all = a.all_entities.clone();
    all.extend(b.all_entities.iter().cloned());
    all.sort();
    all.dedup();

    CognitiveChunk {
        text,
        chunk_index: a.chunk_index, // will be reassigned
        offset_start: a.offset_start,
        offset_end: b.offset_end,
        heading_path: a.heading_path.clone(),
        dominant_entities: dominant,
        all_entities: all,
        dominant_relations: {
            let mut rels = a.dominant_relations.clone();
            rels.extend(b.dominant_relations.iter().cloned());
            rels
        },
        token_estimate: a.token_estimate + b.token_estimate,
        // Use the lower confidence (conservative estimate)
        continuity_confidence: a.continuity_confidence.min(b.continuity_confidence),
        boundary_reasons_start: a.boundary_reasons_start.clone(),
        boundary_reasons_end: b.boundary_reasons_end.clone(),
        synopsis: None, // will be regenerated if requested
        prev_chunk: None,
        next_chunk: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::blocks::BlockKind;
    use crate::semantic::cognitive_types::ContinuationFlags;

    fn make_block(
        text: &str,
        pronoun: bool,
        demonstrative: bool,
        discourse: bool,
    ) -> BlockEnvelope {
        BlockEnvelope {
            text: text.to_string(),
            offset_start: 0,
            offset_end: text.len(),
            block_type: BlockKind::Sentence,
            heading_path: vec![],
            embedding: None,
            entities: vec![],
            noun_phrases: vec![],
            discourse_markers: if discourse {
                vec![crate::semantic::cognitive_types::DiscourseMarker::Continuation]
            } else {
                vec![]
            },
            continuation_flags: ContinuationFlags {
                starts_with_pronoun: pronoun,
                starts_with_demonstrative: demonstrative,
                starts_with_discourse: discourse,
                continues_list: false,
            },
            token_estimate: text.len() / 4,
        }
    }

    fn make_signal(index: usize, is_break: bool) -> BoundarySignal {
        BoundarySignal {
            index,
            semantic_similarity: 0.5,
            entity_continuity: 0.0,
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

    fn make_chunk(
        index: usize,
        text: &str,
        offset_start: usize,
        offset_end: usize,
    ) -> CognitiveChunk {
        CognitiveChunk {
            text: text.to_string(),
            chunk_index: index,
            offset_start,
            offset_end,
            heading_path: vec![],
            dominant_entities: vec![],
            all_entities: vec![],
            dominant_relations: vec![],
            token_estimate: text.len() / 4,
            continuity_confidence: 0.8,
            boundary_reasons_start: vec![],
            boundary_reasons_end: vec![],
            synopsis: None,
            prev_chunk: if index > 0 { Some(index - 1) } else { None },
            next_chunk: Some(index + 1),
        }
    }

    #[test]
    fn test_heal_pronoun_break() {
        let blocks = vec![
            make_block("The system processes text.", false, false, false),
            make_block("It also handles tables.", true, false, false),
        ];
        let signals = vec![make_signal(0, true)];
        let chunks = vec![
            make_chunk(0, "The system processes text.", 0, blocks[0].text.len()),
            make_chunk(
                1,
                "It also handles tables.",
                blocks[0].text.len(),
                blocks[0].text.len() + blocks[1].text.len(),
            ),
        ];

        let (healed, result) = heal_proposition_breaks(chunks, &blocks, &signals, 1000);
        assert_eq!(result.merges, 1, "Should merge pronoun-starting chunk");
        assert_eq!(healed.len(), 1, "Should produce 1 merged chunk");
        assert!(
            result.merge_reasons[0].contains("unresolved pronoun"),
            "Reason should mention pronoun: {:?}",
            result.merge_reasons
        );
    }

    #[test]
    fn test_heal_demonstrative_break() {
        let blocks = vec![
            make_block("We designed a new algorithm.", false, false, false),
            make_block("This approach improves performance.", false, true, false),
        ];
        let signals = vec![make_signal(0, true)];
        let chunks = vec![
            make_chunk(0, "We designed a new algorithm.", 0, blocks[0].text.len()),
            make_chunk(
                1,
                "This approach improves performance.",
                blocks[0].text.len(),
                blocks[0].text.len() + blocks[1].text.len(),
            ),
        ];

        let (healed, result) = heal_proposition_breaks(chunks, &blocks, &signals, 1000);
        assert_eq!(
            result.merges, 1,
            "Should merge demonstrative-starting chunk"
        );
        assert_eq!(healed.len(), 1);
    }

    #[test]
    fn test_heal_respects_budget() {
        let blocks = vec![
            make_block("First sentence.", false, false, false),
            make_block("It continues.", true, false, false),
        ];
        let signals = vec![make_signal(0, true)];
        let chunks = vec![
            make_chunk(0, "First sentence.", 0, blocks[0].text.len()),
            make_chunk(
                1,
                "It continues.",
                blocks[0].text.len(),
                blocks[0].text.len() + blocks[1].text.len(),
            ),
        ];

        // Set hard budget too low to allow merge
        let (healed, result) = heal_proposition_breaks(chunks, &blocks, &signals, 5);
        assert_eq!(result.merges, 0, "Should not merge when budget exceeded");
        assert_eq!(healed.len(), 2);
    }

    #[test]
    fn test_heal_no_merge_for_clean_break() {
        let blocks = vec![
            make_block("Section A content.", false, false, false),
            make_block("Section B content.", false, false, false),
        ];
        let signals = vec![make_signal(0, true)];
        let chunks = vec![
            make_chunk(0, "Section A content.", 0, blocks[0].text.len()),
            make_chunk(
                1,
                "Section B content.",
                blocks[0].text.len(),
                blocks[0].text.len() + blocks[1].text.len(),
            ),
        ];

        let (healed, result) = heal_proposition_breaks(chunks, &blocks, &signals, 1000);
        assert_eq!(result.merges, 0, "Should not merge clean breaks");
        assert_eq!(healed.len(), 2);
    }

    #[test]
    fn test_heal_discourse_continuation() {
        let blocks = vec![
            make_block("The data shows improvement.", false, false, false),
            make_block("Furthermore, the trend continues.", false, false, true),
        ];
        let signals = vec![make_signal(0, true)];
        let chunks = vec![
            make_chunk(0, "The data shows improvement.", 0, blocks[0].text.len()),
            make_chunk(
                1,
                "Furthermore, the trend continues.",
                blocks[0].text.len(),
                blocks[0].text.len() + blocks[1].text.len(),
            ),
        ];

        let (healed, result) = heal_proposition_breaks(chunks, &blocks, &signals, 1000);
        assert_eq!(result.merges, 1, "Should merge discourse continuation");
        assert_eq!(healed.len(), 1);
    }
}
