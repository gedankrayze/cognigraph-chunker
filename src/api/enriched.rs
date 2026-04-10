//! POST /api/v1/enriched handler.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use crate::llm::{CompletionClient, LlmConfig};
use crate::semantic::enriched_chunk::{EnrichedConfig, enriched_chunk, enriched_chunk_plain};
use crate::semantic::enriched_types::TypedEntity;

use super::AppState;
use super::errors::ApiError;

fn default_soft_budget() -> usize {
    512
}
fn default_hard_budget() -> usize {
    768
}
fn default_recombine() -> bool {
    true
}
fn default_re_enrich() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct EnrichedRequest {
    pub text: String,
    pub enrichment_model: Option<String>,
    pub api_key: Option<String>,
    pub llm_base_url: Option<String>,
    #[serde(default = "default_soft_budget")]
    pub soft_budget: usize,
    #[serde(default = "default_hard_budget")]
    pub hard_budget: usize,
    #[serde(default = "default_recombine")]
    pub recombine: bool,
    #[serde(default = "default_re_enrich")]
    pub re_enrich: bool,
    #[serde(default)]
    pub no_markdown: bool,
}

#[derive(Serialize)]
pub struct EnrichedResponse {
    pub chunks: Vec<EnrichedChunkEntry>,
    pub count: usize,
    pub block_count: usize,
    pub key_dictionary: std::collections::HashMap<String, Vec<usize>>,
}

#[derive(Serialize)]
pub struct EnrichedChunkEntry {
    pub index: usize,
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub length: usize,
    pub token_estimate: usize,
    pub title: String,
    pub summary: String,
    pub keywords: Vec<String>,
    pub typed_entities: Vec<TypedEntity>,
    pub hypothetical_questions: Vec<String>,
    pub semantic_keys: Vec<String>,
    pub category: String,
    pub heading_path: Vec<String>,
}

pub async fn enriched_handler(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<EnrichedRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let llm_config = LlmConfig::resolve(
        &req.api_key,
        &req.llm_base_url,
        &req.enrichment_model,
    )?;
    let llm_client = CompletionClient::new(llm_config)?;

    let config = EnrichedConfig {
        soft_budget: req.soft_budget,
        hard_budget: req.hard_budget,
        recombine: req.recombine,
        re_enrich: req.re_enrich,
    };

    let result = if req.no_markdown {
        enriched_chunk_plain(&req.text, &llm_client, &config).await?
    } else {
        enriched_chunk(&req.text, &llm_client, &config).await?
    };

    let chunks: Vec<EnrichedChunkEntry> = result
        .chunks
        .iter()
        .enumerate()
        .map(|(i, c)| EnrichedChunkEntry {
            index: i,
            text: c.text.clone(),
            offset_start: c.offset_start,
            offset_end: c.offset_end,
            length: c.text.len(),
            token_estimate: c.token_estimate,
            title: c.title.clone(),
            summary: c.summary.clone(),
            keywords: c.keywords.clone(),
            typed_entities: c.typed_entities.clone(),
            hypothetical_questions: c.hypothetical_questions.clone(),
            semantic_keys: c.semantic_keys.clone(),
            category: c.category.clone(),
            heading_path: c.heading_path.clone(),
        })
        .collect();

    let count = chunks.len();
    let response = EnrichedResponse {
        chunks,
        count,
        block_count: result.block_count,
        key_dictionary: result.key_dictionary,
    };

    Ok(Json(serde_json::to_value(response).unwrap()))
}
