//! POST /api/v1/chunk handler.

use axum::Json;
use serde::Deserialize;

use crate::core::chunk::chunk;

use super::errors::ApiError;
use super::types::{
    ChunksResponse, MergeParams, chunks_response, maybe_merge_api, parse_delimiters,
};

fn default_size() -> usize {
    4096
}

#[derive(Debug, Deserialize)]
pub struct ChunkRequest {
    pub text: String,
    #[serde(default = "default_size")]
    pub size: usize,
    pub delimiters: Option<String>,
    pub pattern: Option<String>,
    #[serde(default)]
    pub prefix: bool,
    #[serde(default)]
    pub consecutive: bool,
    #[serde(default)]
    pub forward_fallback: bool,
    #[serde(default, flatten)]
    pub merge_params: MergeParams,
}

pub async fn chunk_handler(
    Json(req): Json<ChunkRequest>,
) -> Result<Json<ChunksResponse>, ApiError> {
    let text_bytes = req.text.as_bytes();
    let mut chunker = chunk(text_bytes).size(req.size);

    let delim_bytes;
    let pattern_bytes;

    if let Some(ref pat) = req.pattern {
        pattern_bytes = pat.as_bytes().to_vec();
        chunker = chunker.pattern(&pattern_bytes);
    } else if let Some(ref delims) = req.delimiters {
        delim_bytes = parse_delimiters(delims);
        chunker = chunker.delimiters(&delim_bytes);
    }

    if req.prefix {
        chunker = chunker.prefix();
    }
    if req.consecutive {
        chunker = chunker.consecutive();
    }
    if req.forward_fallback {
        chunker = chunker.forward_fallback();
    }

    let mut offset = 0;
    let mut results: Vec<(String, usize)> = Vec::new();
    for chunk_bytes in chunker {
        let text = String::from_utf8_lossy(chunk_bytes).into_owned();
        results.push((text, offset));
        offset += chunk_bytes.len();
    }

    let results = maybe_merge_api(results, &req.merge_params);
    Ok(Json(chunks_response(results)))
}
