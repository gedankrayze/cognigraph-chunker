//! POST /api/v1/evaluate handler — intrinsic quality metrics for chunking output.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use crate::semantic::quality_metrics::{
    ChunkForEval, MetricConfig, MetricWeights, QualityMetrics, evaluate_chunks,
};

use super::AppState;
use super::errors::ApiError;

fn default_soft_budget() -> usize {
    512
}

fn default_hard_budget() -> usize {
    768
}

#[derive(Debug, Deserialize)]
pub struct EvaluateRequest {
    /// The original document text (used for block integrity analysis).
    pub text: String,
    /// The chunks to evaluate.
    pub chunks: Vec<ChunkForEval>,
    /// Embedding provider to use for cohesion and coherence metrics.
    #[serde(flatten)]
    pub provider_opts: crate::embeddings::EmbedProviderOpts,
    #[serde(default = "default_soft_budget")]
    pub soft_budget: usize,
    #[serde(default = "default_hard_budget")]
    pub hard_budget: usize,
    /// Optional custom metric weights. Defaults to 0.20 for each metric.
    pub metric_weights: Option<MetricWeights>,
}

#[derive(Serialize)]
pub struct EvaluateResponse {
    pub metrics: QualityMetrics,
    pub chunk_count: usize,
}

pub async fn evaluate_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EvaluateRequest>,
) -> Result<Json<EvaluateResponse>, ApiError> {
    // SSRF validation for every user-supplied outbound URL
    super::validation::validate_outbound_urls(
        state.allow_private_urls,
        &[
            ("base_url", req.provider_opts.base_url.as_deref()),
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

    let config = MetricConfig {
        soft_budget: req.soft_budget,
        hard_budget: req.hard_budget,
        weights: req.metric_weights.unwrap_or_default(),
    };

    let chunk_count = req.chunks.len();

    let provider = super::provider::build_api_provider(&req.provider_opts, &state).await?;
    let metrics = evaluate_chunks(&req.text, &req.chunks, &provider, &config).await?;

    Ok(Json(EvaluateResponse {
        metrics,
        chunk_count,
    }))
}
