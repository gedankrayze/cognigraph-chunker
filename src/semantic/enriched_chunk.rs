//! Enriched chunking pipeline.
//!
//! Structure-preserving chunking + single-call LLM enrichment with
//! 7 metadata fields + semantic-key recombination.
//!
//! Pipeline:
//! 1. Parse blocks (markdown or plain)
//! 2. Compute heading paths
//! 3. Greedy initial grouping respecting budgets and atomic blocks
//! 4. LLM enrichment per chunk (rolling key dictionary)
//! 5. Semantic-key recombination (merge chunks sharing identical keys)
//! 6. Optional re-enrichment of merged chunks

use std::collections::HashMap;

use anyhow::Result;

use crate::llm::CompletionClient;
use crate::llm::enrichment::{self, EnrichmentResponse};

use super::blocks::{Block, BlockKind, split_blocks};
use super::enriched_types::{EnrichedChunk, EnrichedResult, MergeRecord, TypedEntity};
use super::enrichment::heading_context::compute_heading_paths;
use super::sentence::split_sentences;

/// Configuration for the enriched chunking pipeline.
#[derive(Debug, Clone)]
pub struct EnrichedConfig {
    /// Soft token budget: prefer to stay under this per chunk.
    pub soft_budget: usize,
    /// Hard token ceiling: never exceed unless a single block is larger.
    pub hard_budget: usize,
    /// Whether to perform semantic-key recombination.
    pub recombine: bool,
    /// Whether to re-enrich merged chunks (update title + summary).
    pub re_enrich: bool,
}

impl Default for EnrichedConfig {
    fn default() -> Self {
        Self {
            soft_budget: 512,
            hard_budget: 768,
            recombine: true,
            re_enrich: true,
        }
    }
}

/// Run the enriched chunking pipeline with markdown-aware parsing.
pub async fn enriched_chunk(
    text: &str,
    llm_client: &CompletionClient,
    config: &EnrichedConfig,
) -> Result<EnrichedResult> {
    let blocks = split_blocks(text);
    run_enriched_pipeline(blocks, llm_client, config).await
}

/// Run the enriched chunking pipeline with plain text (no markdown).
pub async fn enriched_chunk_plain(
    text: &str,
    llm_client: &CompletionClient,
    config: &EnrichedConfig,
) -> Result<EnrichedResult> {
    let sentences = split_sentences(text);
    let blocks: Vec<Block<'_>> = sentences
        .into_iter()
        .map(|s| Block {
            text: s.text,
            offset: s.offset,
            kind: BlockKind::Sentence,
        })
        .collect();
    run_enriched_pipeline(blocks, llm_client, config).await
}

/// Estimate token count using whitespace splitting (fast approximation).
pub fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

/// A group of blocks forming an initial chunk before LLM enrichment.
struct InitialGroup {
    text: String,
    offset_start: usize,
    offset_end: usize,
    token_estimate: usize,
    heading_path: Vec<String>,
}

/// Greedy initial grouping of blocks into chunks.
///
/// Rules:
/// - Headings start a new chunk.
/// - Atomic blocks (Table, CodeBlock, List, BlockQuote) are kept whole.
/// - Accumulate blocks until the soft budget is reached.
fn initial_grouping(
    blocks: &[Block<'_>],
    heading_paths: &[Vec<String>],
    soft_budget: usize,
) -> Vec<InitialGroup> {
    if blocks.is_empty() {
        return vec![];
    }

    let mut groups: Vec<InitialGroup> = Vec::new();
    let mut current_text = String::new();
    let mut current_start = blocks[0].offset;
    let mut current_tokens: usize = 0;
    let mut current_heading = if !heading_paths.is_empty() {
        heading_paths[0].clone()
    } else {
        vec![]
    };

    for (i, block) in blocks.iter().enumerate() {
        let block_tokens = estimate_tokens(block.text);
        let is_heading = block.kind == BlockKind::Heading;
        let is_atomic = matches!(
            block.kind,
            BlockKind::Table | BlockKind::CodeBlock | BlockKind::List | BlockKind::BlockQuote
        );
        let heading_path = if i < heading_paths.len() {
            &heading_paths[i]
        } else {
            &current_heading
        };

        // Start new chunk on heading
        if is_heading && !current_text.is_empty() {
            let offset_end = current_start + current_text.len();
            groups.push(InitialGroup {
                text: std::mem::take(&mut current_text),
                offset_start: current_start,
                offset_end,
                token_estimate: current_tokens,
                heading_path: current_heading.clone(),
            });
            current_start = block.offset;
            current_tokens = 0;
        }

        // Start new chunk if adding this block would exceed soft budget
        // (unless current chunk is empty — always take at least one block)
        if !current_text.is_empty()
            && current_tokens + block_tokens > soft_budget
            && (is_atomic || block_tokens > 0)
        {
            let offset_end = current_start + current_text.len();
            groups.push(InitialGroup {
                text: std::mem::take(&mut current_text),
                offset_start: current_start,
                offset_end,
                token_estimate: current_tokens,
                heading_path: current_heading.clone(),
            });
            current_start = block.offset;
            current_tokens = 0;
        }

        current_text.push_str(block.text);
        current_tokens += block_tokens;
        current_heading = heading_path.clone();
    }

    // Flush remaining
    if !current_text.is_empty() {
        let offset_end = current_start + current_text.len();
        groups.push(InitialGroup {
            text: current_text,
            offset_start: current_start,
            offset_end,
            token_estimate: current_tokens,
            heading_path: current_heading,
        });
    }

    groups
}

async fn run_enriched_pipeline(
    blocks: Vec<Block<'_>>,
    llm_client: &CompletionClient,
    config: &EnrichedConfig,
) -> Result<EnrichedResult> {
    let block_count = blocks.len();

    if blocks.is_empty() {
        return Ok(EnrichedResult {
            chunks: vec![],
            key_dictionary: HashMap::new(),
            merge_history: vec![],
            block_count: 0,
        });
    }

    // Step 1-2: Compute heading paths
    let (heading_paths, _heading_terms) = compute_heading_paths(&blocks);

    // Step 3: Initial grouping
    let groups = initial_grouping(&blocks, &heading_paths, config.soft_budget);

    // Step 4: Enrich each group via LLM
    let mut key_dictionary: HashMap<String, Vec<usize>> = HashMap::new();
    let mut chunks: Vec<EnrichedChunk> = Vec::with_capacity(groups.len());

    for (idx, group) in groups.iter().enumerate() {
        let enrichment = enrichment::enrich_chunk(llm_client, &group.text, &key_dictionary).await?;

        // Update key dictionary
        for key in &enrichment.semantic_keys {
            let normalized = key.to_lowercase();
            key_dictionary
                .entry(normalized.clone())
                .or_default()
                .push(idx);
        }

        chunks.push(enrichment_to_chunk(group, enrichment));
    }

    // Step 5: Semantic-key recombination
    let mut merge_history: Vec<MergeRecord> = Vec::new();

    if config.recombine {
        let (merged_chunks, merges) =
            recombine_by_keys(&chunks, &key_dictionary, config.hard_budget);
        chunks = merged_chunks;
        merge_history = merges;

        // Rebuild key dictionary after merges
        key_dictionary.clear();
        for (idx, chunk) in chunks.iter().enumerate() {
            for key in &chunk.semantic_keys {
                key_dictionary.entry(key.clone()).or_default().push(idx);
            }
        }

        // Step 6: Re-enrich merged chunks
        if config.re_enrich && !merge_history.is_empty() {
            for record in &merge_history {
                let chunk_idx = record.result_chunk;
                if chunk_idx < chunks.len() {
                    match enrichment::re_enrich_merged(llm_client, &chunks[chunk_idx].text).await {
                        Ok((title, summary)) => {
                            chunks[chunk_idx].title = title;
                            chunks[chunk_idx].summary = summary;
                        }
                        Err(_) => {
                            // Keep original title/summary on failure
                        }
                    }
                }
            }
        }
    }

    Ok(EnrichedResult {
        chunks,
        key_dictionary,
        merge_history,
        block_count,
    })
}

/// Convert an LLM enrichment response into an EnrichedChunk.
fn enrichment_to_chunk(group: &InitialGroup, resp: EnrichmentResponse) -> EnrichedChunk {
    EnrichedChunk {
        text: group.text.clone(),
        offset_start: group.offset_start,
        offset_end: group.offset_end,
        token_estimate: group.token_estimate,
        title: resp.title,
        summary: resp.summary,
        keywords: resp.keywords,
        typed_entities: resp
            .typed_entities
            .into_iter()
            .map(|e| TypedEntity {
                name: e.name,
                entity_type: e.entity_type,
            })
            .collect(),
        hypothetical_questions: resp.hypothetical_questions,
        semantic_keys: resp
            .semantic_keys
            .into_iter()
            .map(|k| k.to_lowercase())
            .collect(),
        category: resp.category,
        heading_path: group.heading_path.clone(),
    }
}

/// Recombine chunks that share identical semantic keys using bin-packing.
///
/// Prioritizes adjacent chunks. Respects hard_budget limit.
fn recombine_by_keys(
    chunks: &[EnrichedChunk],
    key_dictionary: &HashMap<String, Vec<usize>>,
    hard_budget: usize,
) -> (Vec<EnrichedChunk>, Vec<MergeRecord>) {
    let mut merged_into: Vec<Option<usize>> = vec![None; chunks.len()];
    let mut merge_records: Vec<MergeRecord> = Vec::new();

    // Find merge candidates: keys that appear in exactly 2 chunks
    let mut merge_pairs: Vec<(String, usize, usize)> = Vec::new();
    for (key, indices) in key_dictionary {
        if indices.len() == 2 {
            let a = indices[0];
            let b = indices[1];
            // Prefer adjacent pairs
            merge_pairs.push((key.clone(), a, b));
        }
    }

    // Sort by adjacency (adjacent pairs first) then by index
    merge_pairs.sort_by_key(|&(_, a, b)| {
        let distance = if b > a { b - a } else { a - b };
        (distance, a)
    });

    for (key, a, b) in &merge_pairs {
        let a = *a;
        let b = *b;

        // Skip if either chunk is already merged
        if merged_into[a].is_some() || merged_into[b].is_some() {
            continue;
        }

        // Check budget
        let combined_tokens = chunks[a].token_estimate + chunks[b].token_estimate;
        if combined_tokens > hard_budget {
            continue;
        }

        // Merge b into a
        merged_into[b] = Some(a);
        merge_records.push(MergeRecord {
            result_chunk: a, // Will be remapped after building result
            source_chunks: vec![a, b],
            shared_key: key.clone(),
        });
    }

    // Build result
    let mut result: Vec<EnrichedChunk> = Vec::new();
    let mut old_to_new: Vec<Option<usize>> = vec![None; chunks.len()];

    for (i, chunk) in chunks.iter().enumerate() {
        if merged_into[i].is_some() {
            continue; // This chunk was merged into another
        }

        let new_idx = result.len();
        old_to_new[i] = Some(new_idx);

        // Collect all chunks merged into this one
        let merge_sources: Vec<usize> = (0..chunks.len())
            .filter(|&j| merged_into[j] == Some(i))
            .collect();

        if merge_sources.is_empty() {
            result.push(chunk.clone());
        } else {
            // Merge texts and metadata
            let mut merged = chunk.clone();
            for &src_idx in &merge_sources {
                let src = &chunks[src_idx];
                merged.text.push('\n');
                merged.text.push_str(&src.text);
                merged.token_estimate += src.token_estimate;
                merged.offset_end = merged.offset_end.max(src.offset_end);

                // Union keywords
                for kw in &src.keywords {
                    if !merged.keywords.contains(kw) {
                        merged.keywords.push(kw.clone());
                    }
                }
                // Union entities
                for ent in &src.typed_entities {
                    if !merged.typed_entities.iter().any(|e| e.name == ent.name) {
                        merged.typed_entities.push(ent.clone());
                    }
                }
                // Union questions
                for q in &src.hypothetical_questions {
                    if !merged.hypothetical_questions.contains(q) {
                        merged.hypothetical_questions.push(q.clone());
                    }
                }
                // Union semantic keys
                for sk in &src.semantic_keys {
                    if !merged.semantic_keys.contains(sk) {
                        merged.semantic_keys.push(sk.clone());
                    }
                }
            }
            result.push(merged);
        }
    }

    // Remap merge record indices
    for record in &mut merge_records {
        if let Some(new_idx) = old_to_new[record.result_chunk] {
            record.result_chunk = new_idx;
        }
    }

    (result, merge_records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world"), 2);
        assert_eq!(estimate_tokens("one two three four"), 4);
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("  spaces  between  "), 2);
    }

    #[test]
    fn test_initial_grouping_respects_budget() {
        // Create blocks that should be split across multiple groups
        let blocks = vec![
            Block {
                text: "Word ".repeat(100).leak(),
                offset: 0,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "More ".repeat(100).leak(),
                offset: 500,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "Extra ".repeat(100).leak(),
                offset: 1000,
                kind: BlockKind::Sentence,
            },
        ];
        let heading_paths = vec![vec![]; 3];

        let groups = initial_grouping(&blocks, &heading_paths, 150);

        // Each block has 100 tokens, budget is 150, so:
        // - Block 0 (100 tokens) fits alone
        // - Block 1 (100 tokens) would exceed 150 with block 0, so starts new group
        // - Block 2 (100 tokens) would exceed 150 with block 1, so starts new group
        assert!(
            groups.len() >= 2,
            "Expected at least 2 groups, got {}",
            groups.len()
        );
        for group in &groups {
            assert!(
                group.token_estimate <= 200,
                "Group token estimate {} exceeds expected limit",
                group.token_estimate
            );
        }
    }

    #[test]
    fn test_initial_grouping_heading_starts_new() {
        let blocks = vec![
            Block {
                text: "Intro text.",
                offset: 0,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "# Section A\n",
                offset: 12,
                kind: BlockKind::Heading,
            },
            Block {
                text: "Section content.",
                offset: 25,
                kind: BlockKind::Sentence,
            },
        ];
        let heading_paths = vec![
            vec![],
            vec!["Section A".to_string()],
            vec!["Section A".to_string()],
        ];

        let groups = initial_grouping(&blocks, &heading_paths, 1000);

        // The heading should force a split even though budget allows all together
        assert_eq!(groups.len(), 2, "Heading should start a new group");
        assert!(groups[0].text.contains("Intro"));
        assert!(groups[1].text.contains("# Section A"));
    }
}
