//! POST /api/v1/cognitive handler.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use crate::embeddings::EmbeddingProvider;
use crate::llm::{CompletionClient, LlmConfig, relations};
use crate::semantic::cognitive_types::{CognitiveConfig, CognitiveResult, CognitiveWeights};
use crate::semantic::diagnostics::signals_to_json;
use crate::semantic::{cognitive_chunk, cognitive_chunk_plain};

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
pub struct CognitiveRequest {
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
    #[serde(default)]
    pub no_markdown: bool,
    #[serde(default)]
    pub emit_signals: bool,
    /// Include relation triples in output metadata.
    #[serde(default)]
    pub relations: bool,
    /// Language override (e.g. "en", "de", "fr"). Auto-detect if omitted.
    pub language: Option<String>,
    /// Path to ONNX reranker model directory for ambiguous boundary refinement.
    pub reranker_path: Option<String>,
    /// If true, return graph-shaped output (nodes + edges) instead of flat chunks.
    #[serde(default)]
    pub graph: bool,
}

#[derive(Serialize)]
pub struct CognitiveResponse {
    pub chunks: Vec<CognitiveChunkEntry>,
    pub count: usize,
    pub block_count: usize,
    pub evaluation: EvaluationResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signals: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub shared_entities: std::collections::HashMap<String, Vec<usize>>,
}

#[derive(Serialize)]
pub struct EvaluationResponse {
    pub entity_orphan_rate: f64,
    pub pronoun_boundary_rate: f64,
    pub heading_attachment_rate: f64,
    pub discourse_break_rate: f64,
    pub triple_severance_rate: f64,
}

#[derive(Serialize)]
pub struct CognitiveChunkEntry {
    pub index: usize,
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub length: usize,
    pub heading_path: Vec<String>,
    pub dominant_entities: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dominant_relations: Vec<RelationEntryApi>,
    pub token_estimate: usize,
    pub continuity_confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synopsis: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prev_chunk: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_chunk: Option<usize>,
}

#[derive(Serialize)]
pub struct RelationEntryApi {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

pub async fn cognitive_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CognitiveRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
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

    let language = req
        .language
        .as_deref()
        .map(|s| {
            s.parse::<crate::semantic::enrichment::language::LanguageGroup>()
                .map_err(|e| anyhow::anyhow!(e))
        })
        .transpose()?;

    let config = CognitiveConfig {
        weights: CognitiveWeights::default(),
        soft_budget: req.soft_budget,
        hard_budget: req.hard_budget,
        sim_window: req.sim_window,
        sg_window: req.sg_window,
        poly_order: req.poly_order,
        max_blocks: 10_000,
        emit_signals: req.emit_signals,
        language,
    };

    let provider = super::provider::build_api_provider(&req.provider_opts, &state).await?;
    let mut result = run_cognitive(
        &req.text,
        &provider,
        &config,
        req.no_markdown,
        &req.reranker_path,
        &state.onnx_model_dir,
    )
    .await?;

    if req.relations {
        let llm_config = LlmConfig::resolve(
            &req.provider_opts.api_key,
            &req.provider_opts.base_url,
            &None,
        )?;
        let llm_client = CompletionClient::new(llm_config)?;
        enrich_with_relations(&mut result, &llm_client).await?;
    }

    if req.graph {
        let graph = crate::semantic::graph_export::to_chunk_graph(&result);
        Ok(Json(serde_json::to_value(graph).unwrap()))
    } else {
        let response = build_response(result, req.emit_signals, req.relations);
        Ok(Json(serde_json::to_value(response).unwrap()))
    }
}

async fn enrich_with_relations(
    result: &mut CognitiveResult,
    client: &CompletionClient,
) -> anyhow::Result<()> {
    for chunk in &mut result.chunks {
        if let Ok(relations) = relations::extract_relations(client, &chunk.text).await {
            chunk.dominant_relations = relations;
        }
    }
    Ok(())
}

async fn run_cognitive<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    config: &CognitiveConfig,
    no_markdown: bool,
    reranker_spec: &Option<String>,
    onnx_model_dir: &Option<std::path::PathBuf>,
) -> anyhow::Result<CognitiveResult> {
    if let Some(spec) = reranker_spec {
        let reranker = build_api_reranker(spec, onnx_model_dir)?;
        if no_markdown {
            crate::semantic::cognitive_chunk_plain_with_reranker(text, provider, config, &reranker)
                .await
        } else {
            crate::semantic::cognitive_chunk_with_reranker(text, provider, config, &reranker).await
        }
    } else if no_markdown {
        cognitive_chunk_plain(text, provider, config).await
    } else {
        cognitive_chunk(text, provider, config).await
    }
}

/// Parse a reranker spec string and construct the appropriate provider.
///
/// Accepted: `"nvidia"`, `"cohere"`, `"cloudflare"`, `"oauth"`,
/// `"onnx:<path>"`, or a bare path. ONNX paths are validated against the
/// server's `--onnx-model-dir` allowlist.
fn build_api_reranker(
    spec: &str,
    onnx_model_dir: &Option<std::path::PathBuf>,
) -> anyhow::Result<crate::embeddings::reranker::AnyReranker> {
    use crate::embeddings::reranker::AnyReranker;

    match spec.to_lowercase().as_str() {
        "nvidia" => {
            let reranker = crate::embeddings::reranker::NvidiaReranker::from_env()?;
            Ok(AnyReranker::Nvidia(reranker))
        }
        "cohere" => {
            let reranker = crate::embeddings::reranker::CohereReranker::from_env()?;
            Ok(AnyReranker::Cohere(reranker))
        }
        "cloudflare" => {
            let reranker = crate::embeddings::reranker::CloudflareReranker::from_env()?;
            Ok(AnyReranker::Cloudflare(reranker))
        }
        "oauth" => {
            let reranker = crate::embeddings::reranker::OAuthReranker::from_env(false)?;
            Ok(AnyReranker::OAuth(reranker))
        }
        other => {
            let path = other.strip_prefix("onnx:").unwrap_or(other);
            let path = super::validation::validate_model_path(path, onnx_model_dir)?;
            let reranker = crate::embeddings::reranker::OnnxReranker::new(&path.to_string_lossy())?;
            Ok(AnyReranker::Onnx(Box::new(reranker)))
        }
    }
}

fn build_response(
    result: CognitiveResult,
    include_signals: bool,
    include_relations: bool,
) -> CognitiveResponse {
    let signals = if include_signals && !result.signals.is_empty() {
        Some(serde_json::Value::Array(signals_to_json(&result.signals)))
    } else {
        None
    };

    let chunks: Vec<CognitiveChunkEntry> = result
        .chunks
        .iter()
        .enumerate()
        .map(|(i, c)| CognitiveChunkEntry {
            index: i,
            text: c.text.clone(),
            offset_start: c.offset_start,
            offset_end: c.offset_end,
            length: c.text.len(),
            heading_path: c.heading_path.clone(),
            dominant_entities: c.dominant_entities.clone(),
            dominant_relations: if include_relations {
                c.dominant_relations
                    .iter()
                    .map(|r| RelationEntryApi {
                        subject: r.subject.clone(),
                        predicate: r.predicate.clone(),
                        object: r.object.clone(),
                    })
                    .collect()
            } else {
                vec![]
            },
            token_estimate: c.token_estimate,
            continuity_confidence: c.continuity_confidence,
            synopsis: c.synopsis.clone(),
            prev_chunk: c.prev_chunk,
            next_chunk: c.next_chunk,
        })
        .collect();

    let count = chunks.len();
    CognitiveResponse {
        chunks,
        count,
        block_count: result.block_count,
        shared_entities: result.shared_entities,
        evaluation: EvaluationResponse {
            entity_orphan_rate: result.evaluation.entity_orphan_rate,
            pronoun_boundary_rate: result.evaluation.pronoun_boundary_rate,
            heading_attachment_rate: result.evaluation.heading_attachment_rate,
            discourse_break_rate: result.evaluation.discourse_break_rate,
            triple_severance_rate: result.evaluation.triple_severance_rate,
        },
        signals,
    }
}
