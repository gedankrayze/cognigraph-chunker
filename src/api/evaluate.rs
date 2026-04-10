//! POST /api/v1/evaluate handler — intrinsic quality metrics for chunking output.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use crate::embeddings::EmbeddingProvider;
use crate::embeddings::cloudflare::{CloudflareProvider, resolve_cloudflare_credentials};
use crate::embeddings::oauth::{OAuthProvider, resolve_oauth_credentials};
use crate::embeddings::ollama::OllamaProvider;
use crate::embeddings::onnx::OnnxProvider;
use crate::embeddings::openai::OpenAiProvider;
use crate::semantic::quality_metrics::{
    ChunkForEval, MetricConfig, MetricWeights, QualityMetrics, evaluate_chunks,
};

use super::AppState;
use super::errors::ApiError;
use super::semantic::{ProviderParam, validate_base_url};

fn default_provider() -> ProviderParam {
    ProviderParam::Ollama
}

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
    #[serde(default = "default_provider")]
    pub provider: ProviderParam,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model_path: Option<String>,
    pub cf_auth_token: Option<String>,
    pub cf_account_id: Option<String>,
    pub cf_ai_gateway: Option<String>,
    pub oauth_token_url: Option<String>,
    pub oauth_client_id: Option<String>,
    pub oauth_client_secret: Option<String>,
    pub oauth_scope: Option<String>,
    pub oauth_base_url: Option<String>,
    #[serde(default)]
    pub danger_accept_invalid_certs: bool,
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
    if let Some(ref base_url) = req.base_url {
        validate_base_url(base_url, state.allow_private_urls)?;
    }

    let config = MetricConfig {
        soft_budget: req.soft_budget,
        hard_budget: req.hard_budget,
        weights: req.metric_weights.unwrap_or_default(),
    };

    let chunk_count = req.chunks.len();

    let metrics = match req.provider {
        ProviderParam::Ollama => {
            let provider = OllamaProvider::new(req.base_url.clone(), req.model.clone())?;
            run_evaluate(&req.text, &req.chunks, &provider, &config).await?
        }
        ProviderParam::Openai => {
            let api_key = resolve_openai_key(&req.api_key)?;
            let provider = OpenAiProvider::new(api_key, req.base_url.clone(), req.model.clone())?;
            run_evaluate(&req.text, &req.chunks, &provider, &config).await?
        }
        ProviderParam::Onnx => {
            let model_path = req
                .model_path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("model_path is required for onnx provider"))?;
            let provider = OnnxProvider::new(model_path)?;
            run_evaluate(&req.text, &req.chunks, &provider, &config).await?
        }
        ProviderParam::Cloudflare => {
            let (token, account_id, gateway) = resolve_cloudflare_credentials(
                &req.cf_auth_token,
                &req.cf_account_id,
                &req.cf_ai_gateway,
            )?;
            let provider = CloudflareProvider::new(token, account_id, req.model.clone(), gateway)?;
            provider.verify_token().await?;
            run_evaluate(&req.text, &req.chunks, &provider, &config).await?
        }
        ProviderParam::Oauth => {
            let creds = resolve_oauth_credentials(
                &req.oauth_token_url,
                &req.oauth_client_id,
                &req.oauth_client_secret,
                &req.oauth_scope,
                &req.oauth_base_url,
                &req.model,
            )?;
            let provider = OAuthProvider::new(
                creds.token_url,
                creds.client_id,
                creds.client_secret,
                creds.scope,
                creds.base_url,
                creds.model,
                req.danger_accept_invalid_certs,
            )?;
            provider.verify_credentials().await?;
            run_evaluate(&req.text, &req.chunks, &provider, &config).await?
        }
    };

    Ok(Json(EvaluateResponse {
        metrics,
        chunk_count,
    }))
}

async fn run_evaluate<P: EmbeddingProvider>(
    text: &str,
    chunks: &[ChunkForEval],
    provider: &P,
    config: &MetricConfig,
) -> anyhow::Result<QualityMetrics> {
    evaluate_chunks(text, chunks, provider, config).await
}

/// Resolve OpenAI API key from request field, env var, or .env.openai file.
fn resolve_openai_key(flag: &Option<String>) -> anyhow::Result<String> {
    if let Some(key) = flag {
        return Ok(key.clone());
    }
    if let Ok(key) = std::env::var("OPENAI_API_KEY")
        && !key.is_empty()
    {
        return Ok(key);
    }
    if let Ok(content) = std::fs::read_to_string(".env.openai") {
        for line in content.lines() {
            let line = line.trim();
            if let Some(val) = line.strip_prefix("OPENAI_API_KEY=") {
                let val = val.trim();
                if !val.is_empty() {
                    return Ok(val.to_string());
                }
            }
        }
    }
    anyhow::bail!("OpenAI API key not found.")
}
