//! Intent-driven chunking pipeline with dynamic programming alignment.
//!
//! Pipeline: blocks → LLM intent generation → embed blocks + intents
//!   → DP alignment → optimal chunk partition

use anyhow::{Result, bail};

use crate::embeddings::EmbeddingProvider;
use crate::llm::CompletionClient;

use super::blocks::{Block, BlockKind, split_blocks};
use super::enrichment::heading_context::compute_heading_paths;
use super::intent_types::{IntentChunk, IntentResult};
use super::sentence::split_sentences;

/// Configuration for intent-driven chunking.
#[derive(Debug, Clone)]
pub struct IntentConfig {
    /// Maximum number of intents to generate via LLM.
    pub max_intents: usize,
    /// Soft token budget per chunk (DP prefers chunks near this size).
    pub soft_budget: usize,
    /// Hard token ceiling per chunk (never exceed unless single block is larger).
    pub hard_budget: usize,
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            max_intents: 20,
            soft_budget: 512,
            hard_budget: 768,
        }
    }
}

/// Run intent-driven chunking with markdown-aware block splitting.
pub async fn intent_chunk<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    llm_client: &CompletionClient,
    config: &IntentConfig,
) -> Result<IntentResult> {
    let blocks = split_blocks(text);
    run_intent_pipeline(blocks, provider, llm_client, config).await
}

/// Run intent-driven chunking with plain text (no markdown parsing).
pub async fn intent_chunk_plain<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    llm_client: &CompletionClient,
    config: &IntentConfig,
) -> Result<IntentResult> {
    let sentences = split_sentences(text);
    let blocks: Vec<Block<'_>> = sentences
        .into_iter()
        .map(|s| Block {
            text: s.text,
            offset: s.offset,
            kind: BlockKind::Sentence,
        })
        .collect();
    run_intent_pipeline(blocks, provider, llm_client, config).await
}

async fn run_intent_pipeline<P: EmbeddingProvider>(
    blocks: Vec<Block<'_>>,
    provider: &P,
    llm_client: &CompletionClient,
    config: &IntentConfig,
) -> Result<IntentResult> {
    let block_count = blocks.len();

    if blocks.is_empty() {
        return Ok(IntentResult {
            chunks: vec![],
            intents: vec![],
            partition_score: 0.0,
            block_count: 0,
        });
    }

    // Step 1: Compute heading paths
    let (heading_paths, _heading_terms) = compute_heading_paths(&blocks);

    // Step 2: Generate intents via LLM
    let full_text: String = blocks.iter().map(|b| b.text).collect::<Vec<_>>().join(" ");
    let intents = crate::llm::intents::generate_intents(llm_client, &full_text, config.max_intents)
        .await?;

    if intents.is_empty() {
        // If LLM returns no intents, produce a single chunk
        let text_joined: String = blocks.iter().map(|b| b.text).collect::<Vec<_>>().join("");
        let offset_start = blocks[0].offset;
        let offset_end = blocks.last().map(|b| b.offset + b.text.len()).unwrap_or(offset_start);
        return Ok(IntentResult {
            chunks: vec![IntentChunk {
                text: text_joined.clone(),
                offset_start,
                offset_end,
                token_estimate: estimate_tokens(&text_joined),
                best_intent: 0,
                alignment_score: 0.0,
                heading_path: heading_paths.first().cloned().unwrap_or_default(),
            }],
            intents: vec![],
            partition_score: 0.0,
            block_count,
        });
    }

    // Step 3: Embed all blocks
    let block_texts: Vec<&str> = blocks.iter().map(|b| b.text).collect();
    let block_embeddings = provider.embed(&block_texts).await?;

    if block_embeddings.len() != blocks.len() {
        bail!(
            "Provider returned {} embeddings for {} blocks",
            block_embeddings.len(),
            blocks.len()
        );
    }

    if block_embeddings.is_empty() {
        bail!("No block embeddings returned");
    }

    let dim = block_embeddings[0].len();
    if dim == 0 {
        bail!("Embedding dimension is 0");
    }

    // Step 4: Embed all intents
    let intent_queries: Vec<&str> = intents.iter().map(|i| i.query.as_str()).collect();
    let intent_embeddings = provider.embed(&intent_queries).await?;

    if intent_embeddings.len() != intents.len() {
        bail!(
            "Provider returned {} embeddings for {} intents",
            intent_embeddings.len(),
            intents.len()
        );
    }

    // Step 5: Compute token estimates per block
    let block_tokens: Vec<usize> = blocks.iter().map(|b| estimate_tokens(b.text)).collect();

    // Step 6: DP alignment
    // dp[i] = (best_score, backtrack_idx) for optimal partition of blocks 0..i
    let n = blocks.len();
    let min_blocks = 1;
    // Max blocks per chunk: enough to fill hard_budget
    let max_blocks_per_chunk = n.min(
        if config.hard_budget > 0 {
            // Estimate: average ~4 tokens per block minimum, so hard_budget/1 is upper bound
            config.hard_budget.max(1)
        } else {
            n
        },
    );

    // dp[i] represents the best partition score for blocks 0..i
    // dp[0] = 0.0 (empty prefix)
    let mut dp_score: Vec<f64> = vec![f64::NEG_INFINITY; n + 1];
    let mut dp_back: Vec<usize> = vec![0; n + 1];
    dp_score[0] = 0.0;

    for i in 1..=n {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_j = 0;

        // Try all chunk sizes ending at block i-1
        for size in 1..=i.min(max_blocks_per_chunk) {
            let j = i - size; // chunk spans blocks j..i
            let chunk_tokens_sum: usize = block_tokens[j..i].iter().sum();

            // Skip if previous prefix is unreachable
            if dp_score[j] == f64::NEG_INFINITY {
                continue;
            }

            // Skip if over hard budget (unless this is a single block)
            if chunk_tokens_sum > config.hard_budget && size > 1 {
                break; // Larger chunks will only be bigger
            }

            if size < min_blocks {
                continue;
            }

            // Compute chunk centroid embedding
            let chunk_centroid = centroid(&block_embeddings[j..i]);

            // Find best intent alignment
            let (_best_intent_idx, alignment) =
                best_intent_match(&chunk_centroid, &intent_embeddings);

            // Budget penalty: penalize chunks far from soft_budget
            let budget_ratio = chunk_tokens_sum as f64 / config.soft_budget as f64;
            let budget_penalty = (budget_ratio - 1.0).abs() * 0.1;

            let chunk_score = alignment - budget_penalty;
            let total = dp_score[j] + chunk_score;

            if total > best_score {
                best_score = total;
                best_j = j;
            }
        }

        dp_score[i] = best_score;
        dp_back[i] = best_j;
    }

    // Backtrack to recover partition
    let mut boundaries = Vec::new();
    let mut pos = n;
    while pos > 0 {
        let start = dp_back[pos];
        boundaries.push((start, pos));
        pos = start;
    }
    boundaries.reverse();

    // Step 7: Build IntentChunk vec
    let partition_score = if n > 0 && !boundaries.is_empty() {
        dp_score[n] / boundaries.len() as f64
    } else {
        0.0
    };

    let mut result_intents = intents;
    let mut chunks = Vec::with_capacity(boundaries.len());

    for (chunk_idx, &(start, end)) in boundaries.iter().enumerate() {
        let chunk_text: String = blocks[start..end]
            .iter()
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");
        let offset_start = blocks[start].offset;
        // Offsets are byte-based (from pulldown_cmark AST), matching split_blocks() convention.
        let offset_end = blocks[end - 1].offset + blocks[end - 1].text.len();
        let token_est = estimate_tokens(&chunk_text);
        let chunk_centroid = centroid(&block_embeddings[start..end]);
        let (best_intent_idx, alignment_score) =
            best_intent_match(&chunk_centroid, &intent_embeddings);

        // Track which chunks matched which intent
        if best_intent_idx < result_intents.len() {
            result_intents[best_intent_idx]
                .matched_chunks
                .push(chunk_idx);
        }

        let heading_path = heading_paths
            .get(start)
            .cloned()
            .unwrap_or_default();

        chunks.push(IntentChunk {
            text: chunk_text,
            offset_start,
            offset_end,
            token_estimate: token_est,
            best_intent: best_intent_idx,
            alignment_score,
            heading_path,
        });
    }

    Ok(IntentResult {
        chunks,
        intents: result_intents,
        partition_score,
        block_count,
    })
}

/// Compute the centroid (element-wise mean) of a set of embeddings.
fn centroid(embeddings: &[Vec<f64>]) -> Vec<f64> {
    if embeddings.is_empty() {
        return vec![];
    }
    let dim = embeddings[0].len();
    let n = embeddings.len() as f64;
    let mut result = vec![0.0; dim];
    for emb in embeddings {
        for (i, &val) in emb.iter().enumerate() {
            result[i] += val;
        }
    }
    for val in &mut result {
        *val /= n;
    }
    result
}

/// Find the intent with highest cosine similarity to the given embedding.
///
/// Returns (intent_index, similarity_score).
fn best_intent_match(embedding: &[f64], intent_embeddings: &[Vec<f64>]) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_sim = f64::NEG_INFINITY;

    for (i, intent_emb) in intent_embeddings.iter().enumerate() {
        let sim = cosine_similarity(embedding, intent_emb);
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }

    (best_idx, best_sim.max(0.0))
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }

    dot / denom
}

/// Estimate token count using whitespace splitting (fast approximation).
fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_centroid_single() {
        let embeddings = vec![vec![1.0, 2.0, 3.0]];
        let c = centroid(&embeddings);
        assert_eq!(c, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_centroid_multiple() {
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let c = centroid(&embeddings);
        let expected = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        for (a, b) in c.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_centroid_empty() {
        let embeddings: Vec<Vec<f64>> = vec![];
        let c = centroid(&embeddings);
        assert!(c.is_empty());
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world"), 2);
        assert_eq!(estimate_tokens("  one  two  three  "), 3);
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("single"), 1);
    }

    #[test]
    fn test_best_intent_match() {
        let embedding = vec![1.0, 0.0, 0.0];
        let intent_embeddings = vec![
            vec![0.0, 1.0, 0.0], // orthogonal
            vec![1.0, 0.0, 0.0], // identical
            vec![0.0, 0.0, 1.0], // orthogonal
        ];
        let (idx, score) = best_intent_match(&embedding, &intent_embeddings);
        assert_eq!(idx, 1);
        assert!((score - 1.0).abs() < 1e-10);
    }
}
