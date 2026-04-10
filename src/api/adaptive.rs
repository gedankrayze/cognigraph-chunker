//! POST /api/v1/adaptive handler.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use crate::embeddings::EmbeddingProvider;
use crate::embeddings::cloudflare::{CloudflareProvider, resolve_cloudflare_credentials};
use crate::embeddings::oauth::{OAuthProvider, resolve_oauth_credentials};
use crate::embeddings::ollama::OllamaProvider;
use crate::embeddings::onnx::OnnxProvider;
use crate::embeddings::openai::OpenAiProvider;
use crate::llm::{CompletionClient, LlmConfig};
use crate::semantic::adaptive_chunk::{AdaptiveConfig, adaptive_chunk};
use crate::semantic::adaptive_types::AdaptiveResult;
use crate::semantic::quality_metrics::MetricWeights;

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
    if let Some(ref base_url) = req.base_url {
        validate_base_url(base_url, state.allow_private_urls)?;
    }

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
    let llm_client = match LlmConfig::resolve(&req.api_key, &req.llm_base_url, &req.llm_model) {
        Ok(llm_config) => CompletionClient::new(llm_config).ok(),
        Err(_) => None,
    };

    let result = match req.provider {
        ProviderParam::Ollama => {
            let provider = OllamaProvider::new(req.base_url.clone(), req.model.clone())?;
            run_adaptive(&req.text, &provider, llm_client.as_ref(), &config).await?
        }
        ProviderParam::Openai => {
            let api_key = resolve_openai_key(&req.api_key)?;
            let provider = OpenAiProvider::new(api_key, req.base_url.clone(), req.model.clone())?;
            run_adaptive(&req.text, &provider, llm_client.as_ref(), &config).await?
        }
        ProviderParam::Onnx => {
            let model_path = req
                .model_path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("model_path is required for onnx provider"))?;
            let provider = OnnxProvider::new(model_path)?;
            run_adaptive(&req.text, &provider, llm_client.as_ref(), &config).await?
        }
        ProviderParam::Cloudflare => {
            let (token, account_id, gateway) = resolve_cloudflare_credentials(
                &req.cf_auth_token,
                &req.cf_account_id,
                &req.cf_ai_gateway,
            )?;
            let provider = CloudflareProvider::new(token, account_id, req.model.clone(), gateway)?;
            provider.verify_token().await?;
            run_adaptive(&req.text, &provider, llm_client.as_ref(), &config).await?
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
            run_adaptive(&req.text, &provider, llm_client.as_ref(), &config).await?
        }
    };

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
