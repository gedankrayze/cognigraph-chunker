//! Staged reranking for ambiguous boundaries.
//!
//! After initial heuristic scoring, boundaries with join scores in an
//! uncertainty band are sent to a cross-encoder for refinement.
//! This avoids O(n) expensive inference calls — typically only 10–20%
//! of boundaries are ambiguous.

use super::cognitive_types::{BlockEnvelope, BoundarySignal};
use crate::embeddings::reranker::RerankerProvider;

/// Identify ambiguous boundary indices.
///
/// A boundary is ambiguous if its join score falls within `[mean - band, mean + band]`.
/// Returns the indices into the signals array that need reranking.
pub fn find_ambiguous_boundaries(signals: &[BoundarySignal], band_width: f64) -> Vec<usize> {
    if signals.is_empty() {
        return vec![];
    }

    let scores: Vec<f64> = signals.iter().map(|s| s.join_score).collect();
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    let std_dev = variance.sqrt();

    // If variance is negligible, all boundaries are equally uncertain
    if std_dev < 0.01 {
        return (0..signals.len()).collect();
    }

    let lower = mean - band_width * std_dev;
    let upper = mean + band_width * std_dev;

    signals
        .iter()
        .enumerate()
        .filter(|(_, s)| s.join_score >= lower && s.join_score <= upper)
        .map(|(i, _)| i)
        .collect()
}

/// Refine ambiguous boundaries using a cross-encoder reranker.
///
/// For each ambiguous boundary between block[i] and block[i+1],
/// the reranker scores the text pair. The reranker score replaces
/// the semantic_similarity component and the join_score is recomputed.
///
/// `reranker_weight` controls how much the reranker overrides the
/// original score (0.0 = no effect, 1.0 = full replacement).
pub async fn refine_boundaries<R: RerankerProvider>(
    blocks: &[BlockEnvelope],
    signals: &mut [BoundarySignal],
    ambiguous_indices: &[usize],
    reranker: &R,
    reranker_weight: f64,
) -> anyhow::Result<usize> {
    if ambiguous_indices.is_empty() {
        return Ok(0);
    }

    let mut refined_count = 0;

    for &idx in ambiguous_indices {
        if idx >= signals.len() || idx + 1 >= blocks.len() {
            continue;
        }

        let text_a = blocks[idx].text.as_str();
        let text_b = blocks[idx + 1].text.as_str();

        // Get reranker score for this pair
        let scores = reranker.rerank(text_a, &[text_b]).await?;
        let reranker_score = scores.first().copied().unwrap_or(0.5);

        // Blend reranker score with original semantic similarity
        let original_sim = signals[idx].semantic_similarity;
        let blended_sim = original_sim * (1.0 - reranker_weight) + reranker_score * reranker_weight;

        // Update the signal with refined scores
        let old_join = signals[idx].join_score;
        let sim_delta = blended_sim - original_sim;
        signals[idx].semantic_similarity = blended_sim;
        signals[idx].join_score += sim_delta;
        signals[idx].reasons.push(format!(
            "reranked: {old_join:.3} → {:.3}",
            signals[idx].join_score
        ));

        refined_count += 1;
    }

    Ok(refined_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(index: usize, join_score: f64) -> BoundarySignal {
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
            join_score,
            is_break: false,
            reasons: vec![],
        }
    }

    #[test]
    fn test_find_ambiguous_boundaries() {
        let signals = vec![
            make_signal(0, 0.9),  // clearly join
            make_signal(1, 0.55), // ambiguous
            make_signal(2, 0.1),  // clearly break
            make_signal(3, 0.5),  // ambiguous
            make_signal(4, 0.85), // clearly join
        ];

        let ambiguous = find_ambiguous_boundaries(&signals, 0.5);
        // Mean ≈ 0.58, std ≈ 0.29, band = [0.44, 0.72]
        // Indices 1 (0.55) and 3 (0.5) should be ambiguous
        assert!(
            ambiguous.contains(&1),
            "Index 1 (score 0.55) should be ambiguous, got {ambiguous:?}"
        );
        assert!(
            ambiguous.contains(&3),
            "Index 3 (score 0.5) should be ambiguous, got {ambiguous:?}"
        );
        assert!(
            !ambiguous.contains(&0),
            "Index 0 (score 0.9) should not be ambiguous"
        );
        assert!(
            !ambiguous.contains(&2),
            "Index 2 (score 0.1) should not be ambiguous"
        );
    }

    #[test]
    fn test_find_ambiguous_uniform_scores() {
        // All scores nearly identical → all are ambiguous
        let signals = vec![
            make_signal(0, 0.50),
            make_signal(1, 0.51),
            make_signal(2, 0.50),
        ];

        let ambiguous = find_ambiguous_boundaries(&signals, 0.5);
        assert_eq!(ambiguous.len(), 3, "Uniform scores → all ambiguous");
    }

    #[test]
    fn test_find_ambiguous_empty() {
        let ambiguous = find_ambiguous_boundaries(&[], 0.5);
        assert!(ambiguous.is_empty());
    }
}
