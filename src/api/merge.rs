//! POST /api/v1/merge handler.

use axum::Json;
use serde::Deserialize;

use crate::core::merge::merge_splits;

use super::errors::ApiError;
use super::types::{ChunkEntry, MergeResponse};

fn default_chunk_size() -> usize {
    512
}

#[derive(Debug, Deserialize)]
pub struct MergeRequest {
    pub chunks: Vec<String>,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
}

pub async fn merge_handler(
    Json(req): Json<MergeRequest>,
) -> Result<Json<MergeResponse>, ApiError> {
    if req.chunks.is_empty() {
        return Ok(Json(MergeResponse {
            chunks: vec![],
            count: 0,
            token_counts: vec![],
        }));
    }

    let texts: Vec<&str> = req.chunks.iter().map(|s| s.as_str()).collect();
    let token_counts: Vec<usize> = texts.iter().map(|t| t.split_whitespace().count()).collect();
    let result = merge_splits(&texts, &token_counts, req.chunk_size);

    let mut offset = 0;
    let entries: Vec<ChunkEntry> = result
        .merged
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let length = text.len();
            let entry = ChunkEntry {
                index: i,
                text: text.clone(),
                offset,
                length,
            };
            offset += length;
            entry
        })
        .collect();

    let count = entries.len();
    Ok(Json(MergeResponse {
        chunks: entries,
        count,
        token_counts: result.token_counts,
    }))
}
