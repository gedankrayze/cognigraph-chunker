//! POST /api/v1/intent handler.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use crate::embeddings::EmbeddingProvider;
use crate::llm::{CompletionClient, LlmConfig};
use crate::semantic::intent_chunk::{IntentConfig, intent_chunk, intent_chunk_plain};
use crate::semantic::intent_types::{IntentResult, PredictedIntent};

use super::AppState;
use super::errors::ApiError;

fn default_soft_budget() -> usize {
    512
}
fn default_hard_budget() -> usize {
    768
}
fn default_max_intents() -> usize {
    20
}

#[derive(Debug, Deserialize)]
pub struct IntentRequest {
    pub text: String,
    #[serde(flatten)]
    pub provider_opts: crate::embeddings::EmbedProviderOpts,
    #[serde(default = "default_soft_budget")]
    pub soft_budget: usize,
    #[serde(default = "default_hard_budget")]
    pub hard_budget: usize,
    /// LLM model for intent generation (default: gpt-4.1-mini).
    pub intent_model: Option<String>,
    /// Maximum number of intents to generate.
    #[serde(default = "default_max_intents")]
    pub max_intents: usize,
    /// Base URL for the LLM API (defaults to OpenAI).
    pub llm_base_url: Option<String>,
    #[serde(default)]
    pub no_markdown: bool,
}

#[derive(Serialize)]
pub struct IntentResponse {
    pub chunks: Vec<IntentChunkEntry>,
    pub intents: Vec<PredictedIntent>,
    pub partition_score: f64,
    pub count: usize,
    pub block_count: usize,
}

#[derive(Serialize)]
pub struct IntentChunkEntry {
    pub index: usize,
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub length: usize,
    pub token_estimate: usize,
    pub best_intent: usize,
    pub alignment_score: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub heading_path: Vec<String>,
}

pub async fn intent_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IntentRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // SSRF validation for every user-supplied outbound URL
    super::validation::validate_outbound_urls(
        state.allow_private_urls,
        &[
            ("base_url", req.provider_opts.base_url.as_deref()),
            ("llm_base_url", req.llm_base_url.as_deref()),
            (
                "oauth_token_url",
                req.provider_opts.oauth_token_url.as_deref(),
            ),
            (
                "oauth_base_url",
                req.provider_opts.oauth_base_url.as_deref(),
            ),
        ],
    )?;

    // Resolve LLM config for intent generation
    let llm_config = LlmConfig::resolve(
        &req.provider_opts.api_key,
        &req.llm_base_url,
        &req.intent_model,
    )?;
    let llm_client = CompletionClient::new(llm_config)?;

    let config = IntentConfig {
        max_intents: req.max_intents,
        soft_budget: req.soft_budget,
        hard_budget: req.hard_budget,
    };

    let provider = super::provider::build_api_provider(&req.provider_opts, &state).await?;
    let result = run_intent(&req.text, &provider, &llm_client, &config, req.no_markdown).await?;

    let response = build_response(result);
    Ok(Json(serde_json::to_value(response).unwrap()))
}

async fn run_intent<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    llm_client: &CompletionClient,
    config: &IntentConfig,
    no_markdown: bool,
) -> anyhow::Result<IntentResult> {
    if no_markdown {
        intent_chunk_plain(text, provider, llm_client, config).await
    } else {
        intent_chunk(text, provider, llm_client, config).await
    }
}

fn build_response(result: IntentResult) -> IntentResponse {
    let chunks: Vec<IntentChunkEntry> = result
        .chunks
        .iter()
        .enumerate()
        .map(|(i, c)| IntentChunkEntry {
            index: i,
            text: c.text.clone(),
            offset_start: c.offset_start,
            offset_end: c.offset_end,
            length: c.text.len(),
            token_estimate: c.token_estimate,
            best_intent: c.best_intent,
            alignment_score: c.alignment_score,
            heading_path: c.heading_path.clone(),
        })
        .collect();

    let count = chunks.len();
    IntentResponse {
        chunks,
        intents: result.intents,
        partition_score: result.partition_score,
        count,
        block_count: result.block_count,
    }
}
