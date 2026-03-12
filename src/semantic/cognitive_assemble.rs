//! Cognitive chunk assembly.
//!
//! Assembles enriched blocks into chunks using boundary signals.
//! Uses local-valley detection on the join score curve, similar to the
//! existing semantic pipeline but with cognitive scoring.

use super::cognitive_types::{
    BlockEnvelope, BoundaryReason, BoundarySignal, CognitiveChunk, CognitiveConfig, CognitiveResult,
};

/// Assemble blocks into cognitive chunks using boundary signals.
///
/// The algorithm:
/// 1. Smooth the join score curve.
/// 2. Find local minima (valleys) — these are candidate break points.
/// 3. Apply hard budget ceiling as a forced break.
/// 4. Assemble blocks into chunks between break points.
pub fn assemble_chunks(
    blocks: &[BlockEnvelope],
    mut signals: Vec<BoundarySignal>,
    config: &CognitiveConfig,
) -> CognitiveResult {
    if blocks.is_empty() {
        return CognitiveResult {
            chunks: vec![],
            signals: vec![],
            block_count: 0,
            evaluation: super::evaluation::EvaluationMetrics::default(),
            shared_entities: std::collections::HashMap::new(),
        };
    }

    if blocks.len() == 1 {
        let chunk = single_block_chunk(&blocks[0]);
        let shared_entities = build_shared_entities(std::slice::from_ref(&chunk));
        return CognitiveResult {
            chunks: vec![chunk],
            signals: vec![],
            block_count: 1,
            evaluation: super::evaluation::EvaluationMetrics::default(),
            shared_entities,
        };
    }

    // Step 1: Find break points using valley detection on join scores
    let join_scores: Vec<f64> = signals.iter().map(|s| s.join_score).collect();
    let break_indices = detect_valleys(&join_scores);

    // Mark breaks in signals
    for &idx in &break_indices {
        if idx < signals.len() {
            signals[idx].is_break = true;
        }
    }

    // Step 2: Apply hard budget ceiling as forced breaks
    apply_budget_breaks(blocks, &mut signals, config.hard_budget);

    // Step 3: Assemble chunks
    let chunks = build_chunks(blocks, &signals);

    // Step 3b: Proposition-aware healing — merge chunks with broken propositions
    let (mut chunks, _heal_result) = super::proposition_heal::heal_proposition_breaks(
        chunks,
        blocks,
        &signals,
        config.hard_budget,
    );

    // Step 4: Assign chunk indices and adjacency links
    let chunk_count = chunks.len();
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.chunk_index = i;
        chunk.prev_chunk = if i > 0 { Some(i - 1) } else { None };
        chunk.next_chunk = if i + 1 < chunk_count {
            Some(i + 1)
        } else {
            None
        };
    }

    // Step 5: Build cross-chunk entity tracking
    let shared_entities = build_shared_entities(&chunks);

    // Always return signals — caller decides whether to keep them for output
    CognitiveResult {
        chunks,
        signals,
        block_count: blocks.len(),
        evaluation: super::evaluation::EvaluationMetrics::default(),
        shared_entities,
    }
}

/// Detect local valleys in the join score curve.
///
/// A valley is a point where the score is lower than both neighbors.
/// Uses a simple approach: mark boundaries where join_score is below
/// the mean minus 0.5 standard deviations, and is a local minimum.
fn detect_valleys(scores: &[f64]) -> Vec<usize> {
    if scores.is_empty() {
        return vec![];
    }

    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    let std_dev = variance.sqrt();

    // If variance is negligible, there are no meaningful valleys
    if std_dev < 0.01 {
        return vec![];
    }

    // Adaptive threshold: mean - 0.5σ
    let threshold = mean - 0.5 * std_dev;

    let mut valleys = Vec::new();

    for i in 0..scores.len() {
        if scores[i] >= threshold {
            continue;
        }

        // Check if it's a local minimum (lower than or equal to neighbors)
        let left_ok = i == 0 || scores[i] <= scores[i - 1];
        let right_ok = i == scores.len() - 1 || scores[i] <= scores[i + 1];

        if left_ok && right_ok {
            valleys.push(i);
        }
    }

    valleys
}

/// Apply forced breaks when accumulated tokens exceed hard budget.
fn apply_budget_breaks(
    blocks: &[BlockEnvelope],
    signals: &mut [BoundarySignal],
    hard_budget: usize,
) {
    if hard_budget == 0 {
        return;
    }

    let mut accumulated = blocks[0].token_estimate;

    for (i, signal) in signals.iter_mut().enumerate() {
        if signal.is_break {
            // Reset accumulator after a break
            accumulated = blocks[i + 1].token_estimate;
            continue;
        }

        accumulated += blocks[i + 1].token_estimate;

        if accumulated > hard_budget {
            signal.is_break = true;
            signal.reasons.push(format!(
                "hard budget ceiling ({accumulated} > {hard_budget} tokens)"
            ));
            accumulated = blocks[i + 1].token_estimate;
        }
    }
}

/// Build chunks from blocks using break decisions in signals.
fn build_chunks(blocks: &[BlockEnvelope], signals: &[BoundarySignal]) -> Vec<CognitiveChunk> {
    let mut chunks = Vec::new();
    let mut chunk_start = 0;

    for (i, signal) in signals.iter().enumerate() {
        if signal.is_break {
            let chunk_end = i + 1; // Include block i, break before block i+1
            chunks.push(create_chunk(blocks, chunk_start, chunk_end, signals));
            chunk_start = chunk_end;
        }
    }

    // Remaining blocks form the last chunk
    if chunk_start < blocks.len() {
        chunks.push(create_chunk(blocks, chunk_start, blocks.len(), signals));
    }

    chunks
}

fn create_chunk(
    blocks: &[BlockEnvelope],
    start: usize,
    end: usize,
    signals: &[BoundarySignal],
) -> CognitiveChunk {
    let chunk_blocks = &blocks[start..end];

    let text = chunk_blocks
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join("");

    let offset_start = chunk_blocks.first().map(|b| b.offset_start).unwrap_or(0);
    let offset_end = chunk_blocks.last().map(|b| b.offset_end).unwrap_or(0);

    // Use the heading path of the first content block (skip headings)
    let heading_path = chunk_blocks
        .iter()
        .find(|b| b.block_type != super::blocks::BlockKind::Heading)
        .map(|b| b.heading_path.clone())
        .unwrap_or_else(|| {
            chunk_blocks
                .first()
                .map(|b| b.heading_path.clone())
                .unwrap_or_default()
        });

    // Collect dominant entities across the chunk
    let mut entity_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for block in chunk_blocks {
        for entity in &block.entities {
            *entity_counts.entry(entity.normalized.clone()).or_default() += 1;
        }
    }
    let mut dominant_entities: Vec<(String, usize)> = entity_counts.into_iter().collect();
    dominant_entities.sort_by(|a, b| b.1.cmp(&a.1));
    let dominant_entities: Vec<String> = dominant_entities
        .into_iter()
        .take(5)
        .map(|(name, _)| name)
        .collect();
    let mut all_entities: Vec<String> = chunk_blocks
        .iter()
        .flat_map(|block| block.entities.iter())
        .map(|entity| entity.normalized.clone())
        .collect();
    all_entities.sort();
    all_entities.dedup();

    let token_estimate: usize = chunk_blocks.iter().map(|b| b.token_estimate).sum();

    // Average join score within the chunk (continuity confidence)
    let internal_scores: Vec<f64> = (start..end.saturating_sub(1))
        .filter_map(|i| signals.get(i).map(|s| s.join_score))
        .collect();
    let continuity_confidence = if internal_scores.is_empty() {
        1.0
    } else {
        internal_scores.iter().sum::<f64>() / internal_scores.len() as f64
    };

    // Boundary reasons
    let boundary_reasons_start = if start > 0 {
        let from_heading = blocks[start - 1].heading_path.as_slice();
        let to_heading = blocks[start].heading_path.as_slice();
        signals
            .get(start - 1)
            .map(|s| derive_boundary_reasons(s, from_heading, to_heading))
            .unwrap_or_default()
    } else {
        vec![]
    };

    let boundary_reasons_end = if end < signals.len() + 1 {
        let from_heading = blocks[end - 1].heading_path.as_slice();
        let to_heading = if end < blocks.len() {
            blocks[end].heading_path.as_slice()
        } else {
            &[]
        };
        signals
            .get(end - 1)
            .map(|s| derive_boundary_reasons(s, from_heading, to_heading))
            .unwrap_or_default()
    } else {
        vec![]
    };

    CognitiveChunk {
        text,
        chunk_index: 0, // assigned after assembly
        offset_start,
        offset_end,
        heading_path,
        dominant_entities,
        all_entities,
        dominant_relations: vec![],
        token_estimate,
        continuity_confidence,
        boundary_reasons_start,
        boundary_reasons_end,
        synopsis: None,
        prev_chunk: None,
        next_chunk: None,
    }
}

fn single_block_chunk(block: &BlockEnvelope) -> CognitiveChunk {
    let mut all_entities: Vec<String> = block
        .entities
        .iter()
        .map(|entity| entity.normalized.clone())
        .collect();
    all_entities.sort();
    all_entities.dedup();

    CognitiveChunk {
        text: block.text.clone(),
        chunk_index: 0,
        offset_start: block.offset_start,
        offset_end: block.offset_end,
        heading_path: block.heading_path.clone(),
        dominant_entities: block
            .entities
            .iter()
            .map(|e| e.normalized.clone())
            .collect(),
        all_entities,
        dominant_relations: vec![],
        token_estimate: block.token_estimate,
        continuity_confidence: 1.0,
        boundary_reasons_start: vec![],
        boundary_reasons_end: vec![],
        synopsis: None,
        prev_chunk: None,
        next_chunk: None,
    }
}

fn derive_boundary_reasons(
    signal: &BoundarySignal,
    from_heading: &[String],
    to_heading: &[String],
) -> Vec<BoundaryReason> {
    let mut reasons = Vec::new();

    for reason in &signal.reasons {
        if reason.starts_with("high orphan risk") {
            reasons.push(BoundaryReason::ContinuationGlue {
                flags: reason.clone(),
            });
            continue;
        }

        if reason.starts_with("discourse continuation") {
            reasons.push(BoundaryReason::DiscourseBreak);
            continue;
        }

        if reason.starts_with("entity continuity") {
            reasons.push(BoundaryReason::EntityDiscontinuity { orphaned: vec![] });
            continue;
        }

        if reason.starts_with("heading change") {
            reasons.push(BoundaryReason::HeadingChange {
                from: from_heading.join(" > "),
                to: to_heading.join(" > "),
            });
            continue;
        }

        if reason.starts_with("topic shift") {
            reasons.push(BoundaryReason::TopicShift {
                similarity_drop: signal.topic_shift_penalty,
            });
            continue;
        }

        if reason.starts_with("relation continuity") {
            reasons.push(BoundaryReason::PropositionComplete);
            continue;
        }

        if reason.starts_with("budget pressure") || reason.starts_with("hard budget ceiling") {
            let tokens = reason
                .split(|c: char| !c.is_ascii_digit())
                .find_map(|part| part.parse::<usize>().ok())
                .unwrap_or(0);
            reasons.push(BoundaryReason::BudgetCeiling { tokens });
            continue;
        }
    }

    if reasons.is_empty() {
        reasons.push(BoundaryReason::TopicShift {
            similarity_drop: signal.topic_shift_penalty,
        });
    }

    reasons
}

/// Build cross-chunk entity tracking: entity name → list of chunk indices.
/// Only includes entities that appear in 2+ chunks (truly "shared").
fn build_shared_entities(
    chunks: &[CognitiveChunk],
) -> std::collections::HashMap<String, Vec<usize>> {
    let mut entity_chunks: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    for (i, chunk) in chunks.iter().enumerate() {
        for entity in &chunk.all_entities {
            entity_chunks.entry(entity.clone()).or_default().push(i);
        }
    }

    // Keep only entities appearing in 2+ chunks
    entity_chunks.retain(|_, indices| indices.len() >= 2);
    entity_chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_valleys() {
        // Clear valley at index 2
        let scores = vec![0.8, 0.7, 0.2, 0.6, 0.9];
        let valleys = detect_valleys(&scores);
        assert!(valleys.contains(&2), "Should detect valley at index 2");
    }

    #[test]
    fn test_detect_valleys_no_breaks() {
        // All similar high scores — no meaningful valleys
        let scores = vec![0.90, 0.89, 0.90, 0.91];
        let valleys = detect_valleys(&scores);
        assert!(
            valleys.is_empty(),
            "Should not detect valleys in uniform high scores"
        );
    }

    #[test]
    fn test_budget_breaks() {
        let blocks: Vec<BlockEnvelope> = (0..5)
            .map(|i| BlockEnvelope {
                text: format!("Block {i}. "),
                offset_start: i * 10,
                offset_end: i * 10 + 9,
                block_type: super::super::blocks::BlockKind::Sentence,
                heading_path: vec![],
                embedding: None,
                entities: vec![],
                noun_phrases: vec![],
                discourse_markers: vec![],
                continuation_flags: Default::default(),
                token_estimate: 200, // Each block ~200 tokens
            })
            .collect();

        let mut signals: Vec<BoundarySignal> = (0..4)
            .map(|i| BoundarySignal {
                index: i,
                semantic_similarity: 0.8,
                entity_continuity: 0.0,
                relation_continuity: 0.0,
                discourse_continuation: 0.0,
                heading_continuity: 1.0,
                structural_affinity: 0.0,
                topic_shift_penalty: 0.2,
                orphan_risk: 0.0,
                budget_pressure: 0.0,
                join_score: 0.7,
                is_break: false,
                reasons: vec![],
            })
            .collect();

        // Hard budget of 500 — should force breaks after ~2 blocks
        apply_budget_breaks(&blocks, &mut signals, 500);
        let break_count = signals.iter().filter(|s| s.is_break).count();
        assert!(
            break_count >= 1,
            "Should have at least one forced break, got {break_count}"
        );
    }
}
