//! POST /api/v1/semantic handler.

use std::net::IpAddr;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use crate::embeddings::EmbeddingProvider;
use crate::embeddings::ollama::OllamaProvider;
use crate::embeddings::onnx::OnnxProvider;
use crate::embeddings::openai::OpenAiProvider;
use crate::semantic::{SemanticConfig, semantic_chunk, semantic_chunk_plain};

use super::AppState;
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

/// Check if an IP address is private/loopback/link-local/non-routable.
///
/// Handles IPv4-mapped IPv6 addresses (e.g. `::ffff:127.0.0.1`) by normalizing
/// to IPv4 before checking.
fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local(),
        IpAddr::V6(v6) => {
            // Check IPv4-mapped (::ffff:x.x.x.x) and IPv4-compatible (::x.x.x.x) forms
            if let Some(v4) = v6.to_ipv4_mapped().or_else(|| v6.to_ipv4()) {
                return v4.is_loopback() || v4.is_private() || v4.is_link_local();
            }

            if v6.is_loopback() || v6.is_multicast() {
                return true;
            }
            let segments = v6.segments();
            // Unique local (fc00::/7): first byte is fc or fd
            let is_unique_local = (segments[0] & 0xfe00) == 0xfc00;
            // Link-local (fe80::/10)
            let is_link_local = (segments[0] & 0xffc0) == 0xfe80;
            is_unique_local || is_link_local
        }
    }
}

/// Validate that a base_url does not point to private/loopback addresses.
///
/// Checks both the literal host and DNS-resolved addresses to prevent rebinding attacks.
fn validate_base_url(raw: &str, allow_private: bool) -> anyhow::Result<()> {
    if allow_private {
        return Ok(());
    }

    let parsed = url::Url::parse(raw).map_err(|e| anyhow::anyhow!("Invalid base_url: {e}"))?;

    let scheme = parsed.scheme();
    anyhow::ensure!(
        scheme == "http" || scheme == "https",
        "Invalid base_url scheme '{scheme}': must be http or https"
    );

    let host = parsed.host_str().unwrap_or("");

    // Reject "localhost"
    if host.eq_ignore_ascii_case("localhost") {
        anyhow::bail!(
            "Invalid base_url: private/loopback addresses are not allowed (use --allow-private-urls to override)"
        );
    }

    // If it's a literal IP, check directly
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_private_ip(ip) {
            anyhow::bail!(
                "Invalid base_url: private/loopback addresses are not allowed (use --allow-private-urls to override)"
            );
        }
        return Ok(());
    }

    // For hostnames, resolve DNS and check all resolved IPs.
    // NOTE: This is a TOCTOU check — reqwest resolves DNS independently at request time.
    // For strict enforcement, consider an outbound proxy or firewall policy.
    // This validation catches the common case and raises the bar significantly.
    let port = parsed.port_or_known_default().unwrap_or(443);
    let socket_addrs: Vec<std::net::SocketAddr> =
        std::net::ToSocketAddrs::to_socket_addrs(&(host, port))
            .map(|iter| iter.collect())
            .unwrap_or_default();

    if socket_addrs.is_empty() {
        anyhow::bail!("Invalid base_url: could not resolve hostname '{host}'");
    }

    for addr in &socket_addrs {
        if is_private_ip(addr.ip()) {
            anyhow::bail!(
                "Invalid base_url: hostname '{host}' resolves to private address {} (use --allow-private-urls to override)",
                addr.ip()
            );
        }
    }

    Ok(())
}

pub async fn semantic_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SemanticRequest>,
) -> Result<Json<ChunksResponse>, ApiError> {
    // SSRF validation for base_url
    if let Some(ref base_url) = req.base_url {
        validate_base_url(base_url, state.allow_private_urls)?;
    }

    let config = SemanticConfig {
        sim_window: req.sim_window,
        sg_window: req.sg_window,
        poly_order: req.poly_order,
        threshold: req.threshold,
        min_distance: req.min_distance,
        ..SemanticConfig::default()
    };

    let result = match req.provider {
        ProviderParam::Ollama => {
            let provider = OllamaProvider::new(req.base_url.clone(), req.model.clone())?;
            run_semantic(&req.text, &provider, &config, req.no_markdown).await?
        }
        ProviderParam::Openai => {
            let api_key = resolve_openai_key(&req.api_key)?;
            let provider = OpenAiProvider::new(api_key, req.base_url.clone(), req.model.clone())?;
            run_semantic(&req.text, &provider, &config, req.no_markdown).await?
        }
        ProviderParam::Onnx => {
            let model_path = req
                .model_path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("model_path is required for onnx provider"))?;
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

    anyhow::bail!(
        "OpenAI API key not found. Provide it via the api_key field, OPENAI_API_KEY env var, or .env.openai file."
    )
}
