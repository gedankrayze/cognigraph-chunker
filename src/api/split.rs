//! POST /api/v1/split handler.

use axum::Json;
use serde::Deserialize;

use crate::core::split::{IncludeDelim, split_at_delimiters, split_at_patterns};

use super::errors::ApiError;
use super::types::{
    ChunksResponse, MergeParams, chunks_response, maybe_merge_api, parse_delimiters, parse_patterns,
};

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncludeDelimParam {
    #[default]
    Prev,
    Next,
    None,
}

impl From<IncludeDelimParam> for IncludeDelim {
    fn from(p: IncludeDelimParam) -> Self {
        match p {
            IncludeDelimParam::Prev => IncludeDelim::Prev,
            IncludeDelimParam::Next => IncludeDelim::Next,
            IncludeDelimParam::None => IncludeDelim::None,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SplitRequest {
    pub text: String,
    pub delimiters: Option<String>,
    pub patterns: Option<String>,
    #[serde(default)]
    pub include_delim: IncludeDelimParam,
    #[serde(default)]
    pub min_chars: usize,
    #[serde(default, flatten)]
    pub merge_params: MergeParams,
}

pub async fn split_handler(
    Json(req): Json<SplitRequest>,
) -> Result<Json<ChunksResponse>, ApiError> {
    let text_bytes = req.text.as_bytes();
    let include_delim: IncludeDelim = req.include_delim.into();

    let offsets = if let Some(ref patterns_str) = req.patterns {
        let pattern_strings = parse_patterns(patterns_str);
        let pattern_refs: Vec<&[u8]> = pattern_strings.iter().map(|s| s.as_bytes()).collect();
        split_at_patterns(text_bytes, &pattern_refs, include_delim, req.min_chars)
    } else {
        let delim_bytes = if let Some(ref d) = req.delimiters {
            parse_delimiters(d)
        } else {
            b"\n.?".to_vec()
        };
        split_at_delimiters(text_bytes, &delim_bytes, include_delim, req.min_chars)
    };

    let results: Vec<(String, usize)> = offsets
        .iter()
        .map(|&(start, end)| {
            let segment = String::from_utf8_lossy(&text_bytes[start..end]).into_owned();
            (segment, start)
        })
        .collect();

    let results = maybe_merge_api(results, &req.merge_params);
    Ok(Json(chunks_response(results)))
}
