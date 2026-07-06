//! POST /api/v1/adaptive handler.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use crate::embeddings::EmbeddingProvider;
use crate::llm::{CompletionClient, LlmConfig};
use crate::semantic::adaptive_chunk::{AdaptiveConfig, adaptive_chunk};
use crate::semantic::adaptive_types::AdaptiveResult;
use crate::semantic::quality_metrics::MetricWeights;

use super::AppState;
use super::errors::ApiError;

fn default_soft_budget() -> usize {
    512
}
fn default_hard_budget() -> usize {
    768
}
fn default_sim_window() -> usize {
    3
}
fn default_sg_window() -> usize {
    11
}
fn default_poly_order() -> usize {
    3
}

#[derive(Debug, Deserialize)]
pub struct AdaptiveRequest {
    pub text: String,
    #[serde(flatten)]
    pub provider_opts: crate::embeddings::EmbedProviderOpts,
    #[serde(default = "default_soft_budget")]
    pub soft_budget: usize,
    #[serde(default = "default_hard_budget")]
    pub hard_budget: usize,
    #[serde(default = "default_sim_window")]
    pub sim_window: usize,
    #[serde(default = "default_sg_window")]
    pub sg_window: usize,
    #[serde(default = "default_poly_order")]
    pub poly_order: usize,
    /// Comma-separated candidate methods (default: "semantic,cognitive").
    pub candidates: Option<String>,
    /// Bypass pre-screening heuristics.
    #[serde(default)]
    pub force_candidates: bool,
    /// Custom metric weights for composite scoring.
    pub metric_weights: Option<MetricWeights>,
    /// Include full quality report in response.
    #[serde(default)]
    pub include_report: bool,
    /// LLM model for intent/enriched/topo methods.
    pub llm_model: Option<String>,
    /// LLM base URL override.
    pub llm_base_url: Option<String>,
}

pub async fn adaptive_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AdaptiveRequest>,
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

    let candidates: Vec<String> = if let Some(ref c) = req.candidates {
        c.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        vec!["semantic".to_string(), "cognitive".to_string()]
    };

    let metric_weights = req.metric_weights.clone().unwrap_or_default();

    let config = AdaptiveConfig {
        candidates,
        force_candidates: req.force_candidates,
        soft_budget: req.soft_budget,
        hard_budget: req.hard_budget,
        metric_weights,
        sim_window: req.sim_window,
        sg_window: req.sg_window,
        poly_order: req.poly_order,
    };

    // Build optional LLM client
    let llm_client = match LlmConfig::resolve(
        &req.provider_opts.api_key,
        &req.llm_base_url,
        &req.llm_model,
    ) {
        Ok(llm_config) => CompletionClient::new(llm_config).ok(),
        Err(_) => None,
    };

    let provider = super::provider::build_api_provider(&req.provider_opts, &state).await?;
    let result = run_adaptive(&req.text, &provider, llm_client.as_ref(), &config).await?;

    if req.include_report {
        Ok(Json(serde_json::to_value(&result).unwrap()))
    } else {
        let output = serde_json::json!({
            "winner": result.winner,
            "chunks": result.chunks,
            "count": result.count,
        });
        Ok(Json(output))
    }
}

async fn run_adaptive<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    llm_client: Option<&CompletionClient>,
    config: &AdaptiveConfig,
) -> anyhow::Result<AdaptiveResult> {
    adaptive_chunk(text, provider, llm_client, config).await
}
