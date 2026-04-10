//! Intrinsic quality metrics for evaluating chunking output.
//!
//! Five metrics are provided:
//! - **Size Compliance (SC):** fraction of chunks within the soft/hard token budget.
//! - **Block Integrity (BI):** fraction of structural elements fully within one chunk.
//! - **Reference Completeness (RC):** fraction of chunks not starting with orphan pronouns.
//! - **Intrachunk Cohesion (ICC):** mean cosine similarity of sentences to their chunk centroid.
//! - **Contextual Coherence (DCC):** mean cosine similarity between adjacent chunk embeddings.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::embeddings::EmbeddingProvider;
use super::blocks::{BlockKind, split_blocks};
use super::sentence::split_sentences;

// ── Types ────────────────────────────────────────────────────────────────────

/// All five intrinsic quality metrics plus a weighted composite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Fraction of chunks whose token count falls within [soft_budget/2, hard_budget].
    pub size_compliance: f64,
    /// Mean cosine similarity of sentence embeddings to their chunk centroid.
    pub intrachunk_cohesion: f64,
    /// Mean cosine similarity between adjacent chunk embeddings.
    pub contextual_coherence: f64,
    /// Fraction of structural elements (tables, code, lists, quotes) fully within one chunk.
    pub block_integrity: f64,
    /// Fraction of chunks not starting with an orphan pronoun/demonstrative.
    pub reference_completeness: f64,
    /// Weighted composite of all five metrics.
    pub composite: f64,
}

/// Weights for the composite score (should sum to 1.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricWeights {
    pub size_compliance: f64,
    pub intrachunk_cohesion: f64,
    pub contextual_coherence: f64,
    pub block_integrity: f64,
    pub reference_completeness: f64,
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self {
            size_compliance: 0.20,
            intrachunk_cohesion: 0.20,
            contextual_coherence: 0.20,
            block_integrity: 0.20,
            reference_completeness: 0.20,
        }
    }
}

/// A chunk ready for evaluation, with its text and byte offsets in the original document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkForEval {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
}

/// Configuration for quality metric evaluation.
#[derive(Debug, Clone)]
pub struct MetricConfig {
    /// Soft token budget (lower bound = soft_budget / 2).
    pub soft_budget: usize,
    /// Hard token budget (upper bound).
    pub hard_budget: usize,
    /// Weights for the composite score.
    pub weights: MetricWeights,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            soft_budget: 512,
            hard_budget: 768,
            weights: MetricWeights::default(),
        }
    }
}

// ── Metric implementations ────────────────────────────────────────────────────

/// Estimate token count by whitespace-splitting the text.
fn token_estimate(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Compute cosine similarity between two equal-length vectors.
///
/// Returns 0.0 if either vector is zero.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "cosine_similarity: vectors must have the same length");

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Compute the element-wise mean of a slice of vectors.
fn mean_vector(vecs: &[Vec<f64>]) -> Vec<f64> {
    if vecs.is_empty() {
        return vec![];
    }
    let dim = vecs[0].len();
    let mut sum = vec![0.0f64; dim];
    for v in vecs {
        for (s, x) in sum.iter_mut().zip(v.iter()) {
            *s += x;
        }
    }
    let n = vecs.len() as f64;
    sum.iter().map(|s| s / n).collect()
}

// ── SC: Size Compliance ───────────────────────────────────────────────────────

/// Fraction of chunks whose token count falls within `[soft_budget/2, hard_budget]`.
///
/// An empty chunk list returns 1.0 (vacuously true).
pub fn size_compliance(chunks: &[ChunkForEval], soft_budget: usize, hard_budget: usize) -> f64 {
    if chunks.is_empty() {
        return 1.0;
    }
    let lower = soft_budget / 2;
    let compliant = chunks
        .iter()
        .filter(|c| {
            let tokens = token_estimate(&c.text);
            tokens >= lower && tokens <= hard_budget
        })
        .count();
    compliant as f64 / chunks.len() as f64
}

// ── BI: Block Integrity ───────────────────────────────────────────────────────

/// Fraction of structural markdown elements fully contained within a single chunk's offset range.
///
/// Structural kinds: `Table`, `CodeBlock`, `List`, `BlockQuote`.
/// Returns 1.0 if no structural elements are found.
pub fn block_integrity(original_text: &str, chunks: &[ChunkForEval]) -> f64 {
    let blocks = split_blocks(original_text);

    let structural: Vec<_> = blocks
        .iter()
        .filter(|b| {
            matches!(
                b.kind,
                BlockKind::Table | BlockKind::CodeBlock | BlockKind::List | BlockKind::BlockQuote
            )
        })
        .collect();

    if structural.is_empty() {
        return 1.0;
    }

    let intact = structural
        .iter()
        .filter(|block| {
            let block_start = block.offset;
            let block_end = block.offset + block.text.len();
            // A block is "intact" if it is fully within any single chunk
            chunks
                .iter()
                .any(|c| c.offset_start <= block_start && block_end <= c.offset_end)
        })
        .count();

    intact as f64 / structural.len() as f64
}

// ── RC: Reference Completeness ────────────────────────────────────────────────

/// Orphan pronouns and demonstratives that should not start a chunk.
const ORPHAN_PREFIXES: &[&str] = &[
    "it ", "it's ", "its ",
    "this ", "these ", "those ", "that ",
    "they ", "them ", "their ", "they're ",
    "he ", "he's ", "him ", "his ",
    "she ", "she's ", "her ",
    "we ", "we've ", "us ", "our ",
    "there ", "here ",
];

/// Fraction of chunks that do NOT start with an orphan pronoun/demonstrative.
///
/// `orphan_count / boundary_count` is subtracted from 1.0.
/// A single-chunk document returns 1.0 (no cross-chunk boundaries).
pub fn reference_completeness(chunks: &[ChunkForEval]) -> f64 {
    if chunks.len() <= 1 {
        return 1.0;
    }
    // Only chunks after the first form cross-chunk boundaries
    let boundary_count = chunks.len() - 1;
    let orphan_count = chunks[1..]
        .iter()
        .filter(|c| {
            let lower = c.text.to_lowercase();
            ORPHAN_PREFIXES.iter().any(|p| lower.starts_with(p))
        })
        .count();
    1.0 - (orphan_count as f64 / boundary_count as f64)
}

// ── ICC: Intrachunk Cohesion ──────────────────────────────────────────────────

/// Mean cosine similarity of sentence embeddings to their chunk centroid, averaged across chunks.
///
/// Chunks with fewer than 2 sentences are skipped (trivially cohesive).
/// Returns 1.0 for empty input.
pub async fn intrachunk_cohesion<P: EmbeddingProvider>(
    chunks: &[ChunkForEval],
    provider: &P,
) -> Result<f64> {
    if chunks.is_empty() {
        return Ok(1.0);
    }

    let mut scores = Vec::new();

    for chunk in chunks {
        let sentences: Vec<String> = split_sentences(&chunk.text)
            .into_iter()
            .map(|s| s.text.to_string())
            .collect();

        if sentences.len() < 2 {
            // Single-sentence chunk is trivially cohesive
            scores.push(1.0);
            continue;
        }

        let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
        let mut texts_to_embed = sentence_refs.clone();
        texts_to_embed.push(&chunk.text);

        let all_embeddings = provider.embed(&texts_to_embed).await?;
        let (sentence_embeddings, chunk_embedding) = all_embeddings.split_at(sentences.len());

        // Compute chunk centroid (mean of sentence embeddings)
        let centroid = mean_vector(sentence_embeddings);
        if centroid.is_empty() {
            continue;
        }

        // Mean cosine similarity of sentences to centroid
        let mean_sim = sentence_embeddings
            .iter()
            .map(|e| cosine_similarity(e, &centroid))
            .sum::<f64>()
            / sentences.len() as f64;

        // Also factor in similarity of each sentence to the full chunk embedding
        let chunk_emb = &chunk_embedding[0];
        let mean_to_chunk = sentence_embeddings
            .iter()
            .map(|e| cosine_similarity(e, chunk_emb))
            .sum::<f64>()
            / sentences.len() as f64;

        scores.push((mean_sim + mean_to_chunk) / 2.0);
    }

    if scores.is_empty() {
        return Ok(1.0);
    }
    Ok(scores.iter().sum::<f64>() / scores.len() as f64)
}

// ── DCC: Contextual Coherence ────────────────────────────────────────────────

/// Mean cosine similarity between adjacent chunk embeddings.
///
/// High values indicate smooth topic flow; low values indicate abrupt transitions.
/// Returns 1.0 for fewer than 2 chunks.
pub async fn contextual_coherence<P: EmbeddingProvider>(
    chunks: &[ChunkForEval],
    provider: &P,
) -> Result<f64> {
    if chunks.len() < 2 {
        return Ok(1.0);
    }

    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let embeddings = provider.embed(&texts).await?;

    let mut total = 0.0f64;
    let pairs = embeddings.len() - 1;
    for i in 0..pairs {
        total += cosine_similarity(&embeddings[i], &embeddings[i + 1]);
    }

    Ok(total / pairs as f64)
}

// ── Composite ─────────────────────────────────────────────────────────────────

/// Weighted sum of all five metric scores.
pub fn composite_score(metrics: &QualityMetrics, weights: &MetricWeights) -> f64 {
    weights.size_compliance * metrics.size_compliance
        + weights.intrachunk_cohesion * metrics.intrachunk_cohesion
        + weights.contextual_coherence * metrics.contextual_coherence
        + weights.block_integrity * metrics.block_integrity
        + weights.reference_completeness * metrics.reference_completeness
}

// ── Main evaluation function ──────────────────────────────────────────────────

/// Evaluate all five intrinsic quality metrics for a set of chunks.
///
/// The `original_text` is used for block integrity analysis.
/// The embedding `provider` is used for cohesion and coherence metrics.
pub async fn evaluate_chunks<P: EmbeddingProvider>(
    original_text: &str,
    chunks: &[ChunkForEval],
    provider: &P,
    config: &MetricConfig,
) -> Result<QualityMetrics> {
    let sc = size_compliance(chunks, config.soft_budget, config.hard_budget);
    let bi = block_integrity(original_text, chunks);
    let rc = reference_completeness(chunks);
    let icc = intrachunk_cohesion(chunks, provider).await?;
    let dcc = contextual_coherence(chunks, provider).await?;

    let mut metrics = QualityMetrics {
        size_compliance: sc,
        intrachunk_cohesion: icc,
        contextual_coherence: dcc,
        block_integrity: bi,
        reference_completeness: rc,
        composite: 0.0,
    };
    metrics.composite = composite_score(&metrics, &config.weights);

    Ok(metrics)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(text: &str, start: usize) -> ChunkForEval {
        ChunkForEval {
            text: text.to_string(),
            offset_start: start,
            offset_end: start + text.len(),
        }
    }

    // ── cosine_similarity ─────────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10, "Identical vectors should have similarity 1.0, got {sim}");
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10, "Orthogonal vectors should have similarity 0.0, got {sim}");
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Zero vector should yield 0.0");
    }

    // ── Size Compliance ───────────────────────────────────────────────────────

    #[test]
    fn test_sc_all_compliant() {
        // 300 tokens is within [256, 768] (lower = 512/2 = 256, upper = 768)
        let text = "word ".repeat(300);
        let chunks = vec![chunk(&text, 0)];
        let sc = size_compliance(&chunks, 512, 768);
        assert_eq!(sc, 1.0, "All compliant → should return 1.0");
    }

    #[test]
    fn test_sc_too_small() {
        // 5 tokens < 256 (512/2)
        let text = "one two three four five";
        let chunks = vec![chunk(text, 0)];
        let sc = size_compliance(&chunks, 512, 768);
        assert_eq!(sc, 0.0, "Too-small chunk → should return 0.0");
    }

    #[test]
    fn test_sc_empty_chunks() {
        let sc = size_compliance(&[], 512, 768);
        assert_eq!(sc, 1.0, "Empty list → vacuously 1.0");
    }

    #[test]
    fn test_sc_mixed() {
        // 400 tokens: within [256, 768] ✓
        let big = "word ".repeat(400);
        // 3 tokens: below lower bound of 256 ✗
        let small = "tiny text here";
        let chunks = vec![chunk(&big, 0), chunk(&small, 2000)];
        let sc = size_compliance(&chunks, 512, 768);
        // big is in range [256, 768], small is not
        assert_eq!(sc, 0.5);
    }

    // ── Block Integrity ───────────────────────────────────────────────────────

    #[test]
    fn test_bi_no_structural_elements() {
        let text = "Hello world. This is plain text. No tables or code here.";
        let chunks = vec![chunk(text, 0)];
        let bi = block_integrity(text, &chunks);
        assert_eq!(bi, 1.0, "No structural elements → should return 1.0");
    }

    #[test]
    fn test_bi_table_fully_contained() {
        let text = "Intro.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nOutro.\n";
        let chunks = vec![chunk(text, 0)];
        let bi = block_integrity(text, &chunks);
        assert_eq!(bi, 1.0, "Table within single chunk → 1.0");
    }

    // ── Reference Completeness ────────────────────────────────────────────────

    #[test]
    fn test_rc_no_orphans() {
        let chunks = vec![
            chunk("The system processes data efficiently.", 0),
            chunk("Performance is measured in throughput.", 50),
            chunk("Results indicate high accuracy.", 100),
        ];
        let rc = reference_completeness(&chunks);
        assert_eq!(rc, 1.0, "No orphans → should return 1.0");
    }

    #[test]
    fn test_rc_with_orphan() {
        let chunks = vec![
            chunk("The system processes data.", 0),
            chunk("it handles errors too.", 30),
        ];
        let rc = reference_completeness(&chunks);
        assert_eq!(rc, 0.0, "All boundaries are orphans → should return 0.0");
    }

    #[test]
    fn test_rc_single_chunk() {
        let chunks = vec![chunk("Just one chunk.", 0)];
        let rc = reference_completeness(&chunks);
        assert_eq!(rc, 1.0, "Single chunk → no boundaries → 1.0");
    }

    #[test]
    fn test_rc_partial_orphans() {
        let chunks = vec![
            chunk("The system processes data.", 0),
            chunk("This is fine.", 30),      // "this " is an orphan
            chunk("Performance is good.", 50), // clean start
        ];
        let rc = reference_completeness(&chunks);
        // 1 orphan out of 2 boundaries
        assert!((rc - 0.5).abs() < 1e-10, "Half orphans → 0.5, got {rc}");
    }
}
