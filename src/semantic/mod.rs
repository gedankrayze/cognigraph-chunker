//! Semantic chunking pipeline.
//!
//! Pipeline: markdown blocks → embeddings → cosine similarity → S-G smooth → minima → filter → chunks
//!
//! Tables, code blocks, lists, and block quotes are kept as atomic units.
//! Paragraphs are sentence-split for fine-grained boundary detection.

pub mod blocks;
pub mod cognitive_assemble;
pub mod cognitive_rerank;
pub mod cognitive_score;
pub mod cognitive_types;
pub mod diagnostics;
pub mod enriched_chunk;
pub mod enriched_types;
pub mod enrichment;
pub mod evaluation;
pub mod graph_export;
pub mod intent_chunk;
pub mod intent_types;
pub mod proposition_heal;
pub mod quality_metrics;
pub mod sentence;
pub mod sir;
pub mod topo_chunk;
pub mod topo_types;

use anyhow::{Result, bail};

use crate::core::savgol::{
    FilteredIndices, filter_split_indices, find_local_minima_interpolated, savgol_filter,
    windowed_cross_similarity,
};
use crate::embeddings::EmbeddingProvider;

use blocks::{Block, BlockKind, split_blocks};
use sentence::split_sentences;

/// Configuration for semantic chunking.
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    /// Window size for windowed cross-similarity (must be odd, >= 3).
    pub sim_window: usize,
    /// Savitzky-Golay filter window size (must be odd).
    pub sg_window: usize,
    /// Savitzky-Golay polynomial order.
    pub poly_order: usize,
    /// Percentile threshold for filtering split points (0.0–1.0).
    pub threshold: f64,
    /// Minimum block gap between split points.
    pub min_distance: usize,
    /// Maximum number of blocks before bailing (O(n²) protection).
    pub max_blocks: usize,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            sim_window: 3,
            sg_window: 11,
            poly_order: 3,
            threshold: 0.5,
            min_distance: 2,
            max_blocks: 10_000,
        }
    }
}

/// Result of semantic chunking, including optional debug signals.
pub struct SemanticResult {
    /// Chunks as (text, byte_offset) pairs.
    pub chunks: Vec<(String, usize)>,
    /// Raw similarity curve (n-1 values for n blocks).
    pub similarities: Vec<f64>,
    /// Smoothed similarity curve.
    pub smoothed: Vec<f64>,
    /// Detected split indices (block boundary indices).
    pub split_indices: FilteredIndices,
    /// Number of blocks by kind.
    pub block_stats: BlockStats,
}

/// Statistics about the blocks extracted from the document.
#[derive(Debug, Default)]
pub struct BlockStats {
    pub sentences: usize,
    pub tables: usize,
    pub code_blocks: usize,
    pub headings: usize,
    pub lists: usize,
    pub block_quotes: usize,
}

impl BlockStats {
    fn from_blocks(blocks: &[Block<'_>]) -> Self {
        let mut stats = Self::default();
        for b in blocks {
            match b.kind {
                BlockKind::Sentence => stats.sentences += 1,
                BlockKind::Table => stats.tables += 1,
                BlockKind::CodeBlock => stats.code_blocks += 1,
                BlockKind::Heading => stats.headings += 1,
                BlockKind::List => stats.lists += 1,
                BlockKind::BlockQuote => stats.block_quotes += 1,
            }
        }
        stats
    }

    pub fn total(&self) -> usize {
        self.sentences
            + self.tables
            + self.code_blocks
            + self.headings
            + self.lists
            + self.block_quotes
    }
}

/// Run the full semantic chunking pipeline with markdown-aware block splitting.
///
/// 1. Parse markdown AST and extract blocks (tables/code kept atomic, paragraphs sentence-split)
/// 2. Embed all blocks via the provider
/// 3. Compute windowed cross-similarity
/// 4. Smooth with Savitzky-Golay filter
/// 5. Find local minima (topic boundaries)
/// 6. Filter by percentile threshold and minimum distance
/// 7. Group blocks into chunks at boundaries
pub async fn semantic_chunk<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    config: &SemanticConfig,
) -> Result<SemanticResult> {
    let blocks = split_blocks(text);
    run_pipeline(blocks, provider, config).await
}

/// Run the semantic chunking pipeline with plain sentence splitting (no markdown parsing).
pub async fn semantic_chunk_plain<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    config: &SemanticConfig,
) -> Result<SemanticResult> {
    let sentences = split_sentences(text);
    let blocks: Vec<Block<'_>> = sentences
        .into_iter()
        .map(|s| Block {
            text: s.text,
            offset: s.offset,
            kind: BlockKind::Sentence,
        })
        .collect();
    run_pipeline(blocks, provider, config).await
}

async fn run_pipeline<P: EmbeddingProvider>(
    blocks: Vec<Block<'_>>,
    provider: &P,
    config: &SemanticConfig,
) -> Result<SemanticResult> {
    let block_stats = BlockStats::from_blocks(&blocks);

    let empty_result = || SemanticResult {
        chunks: vec![],
        similarities: vec![],
        smoothed: vec![],
        split_indices: FilteredIndices {
            indices: vec![],
            values: vec![],
        },
        block_stats: BlockStats::default(),
    };

    if blocks.is_empty() {
        return Ok(empty_result());
    }

    if blocks.len() > config.max_blocks {
        bail!(
            "Input exceeds maximum block count ({} blocks, limit {}). \
             Reduce input size or increase max_blocks.",
            blocks.len(),
            config.max_blocks
        );
    }

    if blocks.len() == 1 {
        return Ok(SemanticResult {
            chunks: vec![(blocks[0].text.to_string(), blocks[0].offset)],
            similarities: vec![],
            smoothed: vec![],
            split_indices: FilteredIndices {
                indices: vec![],
                values: vec![],
            },
            block_stats,
        });
    }

    // Embed all blocks
    let block_texts: Vec<&str> = blocks.iter().map(|b| b.text).collect();
    let embeddings = provider.embed(&block_texts).await?;

    if embeddings.len() != blocks.len() {
        bail!(
            "Provider returned {} embeddings for {} blocks",
            embeddings.len(),
            blocks.len()
        );
    }

    let dim = embeddings[0].len();
    if dim == 0 {
        bail!("Embedding dimension is 0");
    }

    // Flatten embeddings into contiguous array for windowed_cross_similarity
    let flat_embeddings: Vec<f64> = embeddings.iter().flat_map(|e| e.iter().copied()).collect();

    // Compute windowed cross-similarity
    let similarities =
        windowed_cross_similarity(&flat_embeddings, blocks.len(), dim, config.sim_window)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Failed to compute cross-similarity (sim_window={}, blocks={}, dim={})",
                    config.sim_window,
                    blocks.len(),
                    dim
                )
            })?;

    // Smooth the similarity curve with Savitzky-Golay filter
    let effective_sg_window = clamp_odd_window(config.sg_window, similarities.len());
    let effective_sg_window = if effective_sg_window <= config.poly_order {
        0
    } else {
        effective_sg_window
    };

    let smoothed = if effective_sg_window >= 3 {
        savgol_filter(&similarities, effective_sg_window, config.poly_order, 0)
            .unwrap_or_else(|| similarities.clone())
    } else {
        similarities.clone()
    };

    // Find local minima in smoothed similarity (low similarity = topic change)
    let minima_window = clamp_odd_window(effective_sg_window.max(5), smoothed.len());

    let minima = if minima_window >= 3 && minima_window > config.poly_order {
        find_local_minima_interpolated(&smoothed, minima_window, config.poly_order, 0.1)
            .unwrap_or_else(|| crate::core::savgol::MinimaResult {
                indices: vec![],
                values: vec![],
            })
    } else {
        crate::core::savgol::MinimaResult {
            indices: vec![],
            values: vec![],
        }
    };

    // Filter split points by threshold and minimum distance
    let split_indices = filter_split_indices(
        &minima.indices,
        &minima.values,
        config.threshold,
        config.min_distance,
    );

    // Group blocks into chunks at split boundaries
    let chunks = group_blocks_at_boundaries(&blocks, &split_indices.indices);

    Ok(SemanticResult {
        chunks,
        similarities,
        smoothed,
        split_indices,
        block_stats,
    })
}

/// Clamp a window size to be odd and <= data_len, minimum 3.
fn clamp_odd_window(window: usize, data_len: usize) -> usize {
    let w = window.min(data_len);
    let w = if w.is_multiple_of(2) {
        w.saturating_sub(1)
    } else {
        w
    };
    w.max(3).min(data_len)
}

/// Group blocks into chunks, splitting at the given boundary indices.
///
/// A split index `k` means we split between block `k` and `k+1`.
fn group_blocks_at_boundaries(
    blocks: &[Block<'_>],
    split_indices: &[usize],
) -> Vec<(String, usize)> {
    if blocks.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut chunk_start = 0;

    for &split_idx in split_indices {
        let chunk_end = split_idx + 1;
        if chunk_end > chunk_start && chunk_end <= blocks.len() {
            let chunk_text = join_blocks(&blocks[chunk_start..chunk_end]);
            let offset = blocks[chunk_start].offset;
            chunks.push((chunk_text, offset));
            chunk_start = chunk_end;
        }
    }

    // Remaining blocks form the last chunk
    if chunk_start < blocks.len() {
        let chunk_text = join_blocks(&blocks[chunk_start..]);
        let offset = blocks[chunk_start].offset;
        chunks.push((chunk_text, offset));
    }

    chunks
}

fn join_blocks(blocks: &[Block<'_>]) -> String {
    blocks.iter().map(|b| b.text).collect::<Vec<_>>().join("")
}

// ── Cognitive chunking pipeline ─────────────────────────────────────

use cognitive_assemble::assemble_chunks;
use cognitive_rerank::{find_ambiguous_boundaries, refine_boundaries};
use cognitive_score::score_boundaries;
use cognitive_types::{CognitiveConfig, CognitiveResult};
use enrichment::enrich_blocks;

use crate::embeddings::reranker::RerankerProvider;

/// Run the cognition-aware chunking pipeline with markdown-aware parsing.
///
/// Pipeline: parse → blocks → enrich → embed → score boundaries → assemble chunks
pub async fn cognitive_chunk<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    config: &CognitiveConfig,
) -> Result<CognitiveResult> {
    let blocks = split_blocks(text);
    run_cognitive_pipeline(blocks, provider, config, None::<&NoReranker>).await
}

/// Run the cognition-aware chunking pipeline with plain text (no markdown).
pub async fn cognitive_chunk_plain<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    config: &CognitiveConfig,
) -> Result<CognitiveResult> {
    let sentences = split_sentences(text);
    let blocks: Vec<Block<'_>> = sentences
        .into_iter()
        .map(|s| Block {
            text: s.text,
            offset: s.offset,
            kind: BlockKind::Sentence,
        })
        .collect();
    run_cognitive_pipeline(blocks, provider, config, None::<&NoReranker>).await
}

/// Run cognitive chunking with reranker for ambiguous boundary refinement.
pub async fn cognitive_chunk_with_reranker<P: EmbeddingProvider, R: RerankerProvider>(
    text: &str,
    provider: &P,
    config: &CognitiveConfig,
    reranker: &R,
) -> Result<CognitiveResult> {
    let blocks = split_blocks(text);
    run_cognitive_pipeline(blocks, provider, config, Some(reranker)).await
}

/// Run cognitive chunking (plain text) with reranker.
pub async fn cognitive_chunk_plain_with_reranker<P: EmbeddingProvider, R: RerankerProvider>(
    text: &str,
    provider: &P,
    config: &CognitiveConfig,
    reranker: &R,
) -> Result<CognitiveResult> {
    let sentences = split_sentences(text);
    let blocks: Vec<Block<'_>> = sentences
        .into_iter()
        .map(|s| Block {
            text: s.text,
            offset: s.offset,
            kind: BlockKind::Sentence,
        })
        .collect();
    run_cognitive_pipeline(blocks, provider, config, Some(reranker)).await
}

/// No-op reranker used as a type witness when reranking is disabled.
struct NoReranker;
impl RerankerProvider for NoReranker {
    async fn rerank(&self, _query: &str, _documents: &[&str]) -> Result<Vec<f64>> {
        Ok(vec![])
    }
    fn model_name(&self) -> &str {
        "none"
    }
}

async fn run_cognitive_pipeline<P: EmbeddingProvider, R: RerankerProvider>(
    blocks: Vec<Block<'_>>,
    provider: &P,
    config: &CognitiveConfig,
    reranker: Option<&R>,
) -> Result<CognitiveResult> {
    if blocks.is_empty() {
        return Ok(CognitiveResult {
            chunks: vec![],
            signals: vec![],
            block_count: 0,
            evaluation: evaluation::EvaluationMetrics::default(),
            shared_entities: std::collections::HashMap::new(),
        });
    }

    if blocks.len() > config.max_blocks {
        bail!(
            "Input exceeds maximum block count ({} blocks, limit {}). \
             Reduce input size or increase max_blocks.",
            blocks.len(),
            config.max_blocks
        );
    }

    // Step 1: Enrich blocks with cognitive signals
    let mut enriched = if let Some(lang) = config.language {
        enrichment::enrich_blocks_with_language(&blocks, lang)
    } else {
        enrich_blocks(&blocks)
    };

    if enriched.len() == 1 {
        return Ok(assemble_chunks(&enriched, vec![], config));
    }

    // Step 2: Embed all blocks
    let block_texts: Vec<&str> = enriched.iter().map(|b| b.text.as_str()).collect();
    let embeddings = provider.embed(&block_texts).await?;

    if embeddings.len() != enriched.len() {
        bail!(
            "Provider returned {} embeddings for {} blocks",
            embeddings.len(),
            enriched.len()
        );
    }

    let dim = embeddings[0].len();
    if dim == 0 {
        bail!("Embedding dimension is 0");
    }

    // Attach embeddings to enriched blocks
    for (block, emb) in enriched.iter_mut().zip(embeddings.iter()) {
        block.embedding = Some(emb.clone());
    }

    // Step 3: Compute semantic similarity curve
    let flat_embeddings: Vec<f64> = embeddings.iter().flat_map(|e| e.iter().copied()).collect();

    let similarities =
        windowed_cross_similarity(&flat_embeddings, enriched.len(), dim, config.sim_window)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Failed to compute cross-similarity (sim_window={}, blocks={}, dim={})",
                    config.sim_window,
                    enriched.len(),
                    dim
                )
            })?;

    // Step 4: Smooth similarity curve
    let effective_sg_window = clamp_odd_window(config.sg_window, similarities.len());
    let effective_sg_window = if effective_sg_window <= config.poly_order {
        0
    } else {
        effective_sg_window
    };

    let smoothed = if effective_sg_window >= 3 {
        savgol_filter(&similarities, effective_sg_window, config.poly_order, 0)
            .unwrap_or_else(|| similarities.clone())
    } else {
        similarities.clone()
    };

    // Step 5: Score boundaries using cognitive signals + smoothed similarity
    let mut signals = score_boundaries(&enriched, &smoothed, &config.weights, config.soft_budget);

    // Step 5b: Rerank ambiguous boundaries (if reranker provided)
    if let Some(reranker) = reranker {
        let ambiguous = find_ambiguous_boundaries(&signals, 0.5);
        if !ambiguous.is_empty() {
            refine_boundaries(&enriched, &mut signals, &ambiguous, reranker, 0.7).await?;
        }
    }

    // Step 6: Assemble chunks (signals always included for evaluation)
    let mut result = assemble_chunks(&enriched, signals, config);

    // Step 7: Evaluate quality metrics
    result.evaluation = evaluation::evaluate(&enriched, &result.signals, &result.chunks);

    // Drop signals if not requested for output
    if !config.emit_signals {
        result.signals = vec![];
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::blocks::Block;
    use super::*;

    #[test]
    fn test_group_blocks_no_splits() {
        let blocks = vec![
            Block {
                text: "Hello. ",
                offset: 0,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "World. ",
                offset: 7,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "Test.",
                offset: 14,
                kind: BlockKind::Sentence,
            },
        ];
        let chunks = group_blocks_at_boundaries(&blocks, &[]);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, "Hello. World. Test.");
        assert_eq!(chunks[0].1, 0);
    }

    #[test]
    fn test_group_blocks_with_table() {
        let blocks = vec![
            Block {
                text: "Intro. ",
                offset: 0,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "| A | B |\n|---|---|\n| 1 | 2 |",
                offset: 7,
                kind: BlockKind::Table,
            },
            Block {
                text: "After.",
                offset: 40,
                kind: BlockKind::Sentence,
            },
        ];
        // Split after the table
        let chunks = group_blocks_at_boundaries(&blocks, &[1]);
        assert_eq!(chunks.len(), 2);
        assert!(
            chunks[0].0.contains("| A | B |"),
            "First chunk should contain table"
        );
        assert_eq!(chunks[1].0, "After.");
    }

    #[test]
    fn test_block_stats() {
        let blocks = vec![
            Block {
                text: "heading",
                offset: 0,
                kind: BlockKind::Heading,
            },
            Block {
                text: "sent1",
                offset: 10,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "sent2",
                offset: 20,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "table",
                offset: 30,
                kind: BlockKind::Table,
            },
            Block {
                text: "code",
                offset: 40,
                kind: BlockKind::CodeBlock,
            },
        ];
        let stats = BlockStats::from_blocks(&blocks);
        assert_eq!(stats.headings, 1);
        assert_eq!(stats.sentences, 2);
        assert_eq!(stats.tables, 1);
        assert_eq!(stats.code_blocks, 1);
        assert_eq!(stats.total(), 5);
    }
}
