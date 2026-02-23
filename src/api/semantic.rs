//! POST /api/v1/semantic handler.

use axum::Json;
use serde::Deserialize;

use crate::embeddings::EmbeddingProvider;
use crate::embeddings::ollama::OllamaProvider;
use crate::embeddings::onnx::OnnxProvider;
use crate::embeddings::openai::OpenAiProvider;
use crate::semantic::{SemanticConfig, semantic_chunk, semantic_chunk_plain};

use super::errors::ApiError;
use super::types::{ChunksResponse, MergeParams, chunks_response, maybe_merge_api};

fn default_sim_window() -> usize {
    3
}
fn default_sg_window() -> usize {
    11
}
fn default_poly_order() -> usize {
    3
}
fn default_threshold() -> f64 {
    0.5
}
fn default_min_distance() -> usize {
    2
}

#[derive(Debug, Deserialize)]
pub struct SemanticRequest {
    pub text: String,
    #[serde(default = "default_provider")]
    pub provider: ProviderParam,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model_path: Option<String>,
    #[serde(default = "default_sim_window")]
    pub sim_window: usize,
    #[serde(default = "default_sg_window")]
    pub sg_window: usize,
    #[serde(default = "default_poly_order")]
    pub poly_order: usize,
    #[serde(default = "default_threshold")]
    pub threshold: f64,
    #[serde(default = "default_min_distance")]
    pub min_distance: usize,
    #[serde(default)]
    pub no_markdown: bool,
    #[serde(default, flatten)]
    pub merge_params: MergeParams,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderParam {
    Ollama,
    Openai,
    Onnx,
}

fn default_provider() -> ProviderParam {
    ProviderParam::Ollama
}

pub async fn semantic_handler(
    Json(req): Json<SemanticRequest>,
) -> Result<Json<ChunksResponse>, ApiError> {
    let config = SemanticConfig {
        sim_window: req.sim_window,
        sg_window: req.sg_window,
        poly_order: req.poly_order,
        threshold: req.threshold,
        min_distance: req.min_distance,
    };

    let result = match req.provider {
        ProviderParam::Ollama => {
            let provider = OllamaProvider::new(req.base_url.clone(), req.model.clone());
            run_semantic(&req.text, &provider, &config, req.no_markdown).await?
        }
        ProviderParam::Openai => {
            let api_key = resolve_openai_key(&req.api_key)?;
            let provider =
                OpenAiProvider::new(api_key, req.base_url.clone(), req.model.clone());
            run_semantic(&req.text, &provider, &config, req.no_markdown).await?
        }
        ProviderParam::Onnx => {
            let model_path = req.model_path.as_deref().ok_or_else(|| {
                anyhow::anyhow!("model_path is required for onnx provider")
            })?;
            let provider = OnnxProvider::new(model_path)?;
            run_semantic(&req.text, &provider, &config, req.no_markdown).await?
        }
    };

    let chunks = maybe_merge_api(result, &req.merge_params);
    Ok(Json(chunks_response(chunks)))
}

async fn run_semantic<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    config: &SemanticConfig,
    no_markdown: bool,
) -> anyhow::Result<Vec<(String, usize)>> {
    let result = if no_markdown {
        semantic_chunk_plain(text, provider, config).await?
    } else {
        semantic_chunk(text, provider, config).await?
    };
    Ok(result.chunks)
}

/// Resolve OpenAI API key from request, env var, or .env.openai file.
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

    anyhow::bail!("OpenAI API key not found. Provide it via the api_key field, OPENAI_API_KEY env var, or .env.openai file.")
}
