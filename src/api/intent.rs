//! POST /api/v1/intent handler.

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
use crate::llm::{CompletionClient, LlmConfig};
use crate::semantic::intent_chunk::{IntentConfig, intent_chunk, intent_chunk_plain};
use crate::semantic::intent_types::{IntentResult, PredictedIntent};

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
fn default_max_intents() -> usize {
    20
}

#[derive(Debug, Deserialize)]
pub struct IntentRequest {
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
    if let Some(ref base_url) = req.base_url {
        validate_base_url(base_url, state.allow_private_urls)?;
    }

    // Resolve LLM config for intent generation
    let llm_config = LlmConfig::resolve(&req.api_key, &req.llm_base_url, &req.intent_model)?;
    let llm_client = CompletionClient::new(llm_config)?;

    let config = IntentConfig {
        max_intents: req.max_intents,
        soft_budget: req.soft_budget,
        hard_budget: req.hard_budget,
    };

    let result = match req.provider {
        ProviderParam::Ollama => {
            let provider = OllamaProvider::new(req.base_url.clone(), req.model.clone())?;
            run_intent(&req.text, &provider, &llm_client, &config, req.no_markdown).await?
        }
        ProviderParam::Openai => {
            let api_key = resolve_openai_key(&req.api_key)?;
            let provider = OpenAiProvider::new(api_key, req.base_url.clone(), req.model.clone())?;
            run_intent(&req.text, &provider, &llm_client, &config, req.no_markdown).await?
        }
        ProviderParam::Onnx => {
            let model_path = req
                .model_path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("model_path is required for onnx provider"))?;
            let provider = OnnxProvider::new(model_path)?;
            run_intent(&req.text, &provider, &llm_client, &config, req.no_markdown).await?
        }
        ProviderParam::Cloudflare => {
            let (token, account_id, gateway) = resolve_cloudflare_credentials(
                &req.cf_auth_token,
                &req.cf_account_id,
                &req.cf_ai_gateway,
            )?;
            let provider = CloudflareProvider::new(token, account_id, req.model.clone(), gateway)?;
            provider.verify_token().await?;
            run_intent(&req.text, &provider, &llm_client, &config, req.no_markdown).await?
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
            run_intent(&req.text, &provider, &llm_client, &config, req.no_markdown).await?
        }
    };

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
