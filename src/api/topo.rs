//! POST /api/v1/topo handler.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use crate::llm::{CompletionClient, LlmConfig};
use crate::semantic::topo_chunk::{TopoConfig, topo_chunk};
use crate::semantic::topo_types::TopoResult;

use super::AppState;
use super::errors::ApiError;

fn default_soft_budget() -> usize {
    512
}
fn default_hard_budget() -> usize {
    768
}

#[derive(Debug, Deserialize)]
pub struct TopoRequest {
    pub text: String,
    /// LLM model for topology agents (default: gpt-4.1-mini).
    pub topo_model: Option<String>,
    /// API key for the LLM.
    pub api_key: Option<String>,
    /// Base URL for the LLM API.
    pub llm_base_url: Option<String>,
    /// Soft token budget per chunk.
    #[serde(default = "default_soft_budget")]
    pub soft_budget: usize,
    /// Hard token ceiling per chunk.
    #[serde(default = "default_hard_budget")]
    pub hard_budget: usize,
    /// Whether to emit the SIR in the response.
    #[serde(default)]
    pub emit_sir: bool,
}

#[derive(Serialize)]
pub struct TopoResponse {
    pub chunks: Vec<TopoChunkEntry>,
    pub count: usize,
    pub block_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sir: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct TopoChunkEntry {
    pub index: usize,
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub length: usize,
    pub token_estimate: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub heading_path: Vec<String>,
    pub section_classification: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cross_references: Vec<usize>,
}

pub async fn topo_handler(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<TopoRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Resolve LLM config
    let llm_config = LlmConfig::resolve(&req.api_key, &req.llm_base_url, &req.topo_model)?;
    let llm_client = CompletionClient::new(llm_config)?;

    let config = TopoConfig {
        soft_budget: req.soft_budget,
        hard_budget: req.hard_budget,
        emit_sir: req.emit_sir,
    };

    let result = topo_chunk(&req.text, &llm_client, &config).await?;

    let response = build_response(result, req.emit_sir);
    Ok(Json(serde_json::to_value(response).unwrap()))
}

fn build_response(result: TopoResult, emit_sir: bool) -> TopoResponse {
    let chunks: Vec<TopoChunkEntry> = result
        .chunks
        .iter()
        .enumerate()
        .map(|(i, c)| TopoChunkEntry {
            index: i,
            text: c.text.clone(),
            offset_start: c.offset_start,
            offset_end: c.offset_end,
            length: c.text.len(),
            token_estimate: c.token_estimate,
            heading_path: c.heading_path.clone(),
            section_classification: c.section_classification.clone(),
            cross_references: c.cross_references.clone(),
        })
        .collect();

    let count = chunks.len();
    let sir = if emit_sir {
        Some(serde_json::to_value(&result.sir).unwrap())
    } else {
        None
    };

    TopoResponse {
        chunks,
        count,
        block_count: result.block_count,
        sir,
    }
}
