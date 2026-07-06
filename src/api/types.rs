//! Shared request/response types for the REST API.

use serde::{Deserialize, Serialize};

/// A chunk in the response.
#[derive(Debug, Serialize)]
pub struct ChunkEntry {
    pub index: usize,
    pub text: String,
    pub offset: usize,
    pub length: usize,
}

/// Standard chunked response.
#[derive(Debug, Serialize)]
pub struct ChunksResponse {
    pub chunks: Vec<ChunkEntry>,
    pub count: usize,
}

/// Merge response with token counts.
#[derive(Debug, Serialize)]
pub struct MergeResponse {
    pub chunks: Vec<ChunkEntry>,
    pub count: usize,
    pub token_counts: Vec<usize>,
}

/// Shared merge post-processing options.
#[derive(Debug, Deserialize, Default)]
pub struct MergeParams {
    /// Apply merge post-processing.
    #[serde(default)]
    pub merge: bool,
    /// Target token count per merged chunk.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
}

fn default_chunk_size() -> usize {
    512
}

/// Build a ChunksResponse from (text, offset) pairs.
pub fn chunks_response(chunks: Vec<(String, usize)>) -> ChunksResponse {
    let count = chunks.len();
    let entries = chunks
        .into_iter()
        .enumerate()
        .map(|(i, (text, offset))| {
            let length = text.len();
            ChunkEntry {
                index: i,
                text,
                offset,
                length,
            }
        })
        .collect();
    ChunksResponse {
        chunks: entries,
        count,
    }
}

/// Apply merge post-processing if enabled.
pub fn maybe_merge_api(chunks: Vec<(String, usize)>, params: &MergeParams) -> Vec<(String, usize)> {
    if !params.merge || chunks.len() <= 1 {
        return chunks;
    }

    use crate::core::merge::merge_splits;

    let texts: Vec<&str> = chunks.iter().map(|(t, _)| t.as_str()).collect();
    let token_counts: Vec<usize> = texts.iter().map(|t| t.split_whitespace().count()).collect();
    let result = merge_splits(&texts, &token_counts, params.chunk_size);

    let mut merged_chunks = Vec::with_capacity(result.merged.len());
    let mut orig_idx = 0;

    for merged_text in &result.merged {
        let offset = chunks[orig_idx].1;
        merged_chunks.push((merged_text.clone(), offset));

        let mut consumed_len = 0;
        while orig_idx < chunks.len() && consumed_len < merged_text.len() {
            consumed_len += chunks[orig_idx].0.len();
            orig_idx += 1;
        }
    }

    merged_chunks
}
