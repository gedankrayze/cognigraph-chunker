//! Embedding providers for semantic chunking.

use std::{env, path::PathBuf};

pub mod cloudflare;
pub mod oauth;
pub mod oauth_token;
pub mod ollama;
pub mod onnx;
pub mod openai;
pub mod reranker;

use anyhow::Result;

/// Trait for embedding providers.
///
/// Each provider takes a batch of text strings and returns their embedding vectors.
#[allow(async_fn_in_trait)]
pub trait EmbeddingProvider {
    /// Embed a batch of text strings, returning one vector per input text.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>>;

    /// Return the embedding dimension (if known ahead of time).
    fn dimension(&self) -> Option<usize> {
        None
    }
}

/// Supported embedding provider types.
#[derive(Debug, Clone, Copy, clap::ValueEnum, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// OpenAI embeddings API
    Openai,
    /// Ollama local embeddings API
    Ollama,
    /// Local ONNX Runtime model
    Onnx,
    /// Cloudflare Workers AI embeddings
    Cloudflare,
    /// OAuth-authenticated OpenAI-compatible endpoint
    Oauth,
}

/// Type-erased embedding provider.
///
/// Lets the CLI and API build a provider once from shared parameters and
/// pass it to any pipeline without monomorphizing per concrete type.
pub enum AnyProvider {
    Ollama(ollama::OllamaProvider),
    OpenAi(openai::OpenAiProvider),
    Onnx(Box<onnx::OnnxProvider>),
    Cloudflare(cloudflare::CloudflareProvider),
    OAuth(oauth::OAuthProvider),
}

impl EmbeddingProvider for AnyProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        match self {
            Self::Ollama(p) => p.embed(texts).await,
            Self::OpenAi(p) => p.embed(texts).await,
            Self::Onnx(p) => p.embed(texts).await,
            Self::Cloudflare(p) => p.embed(texts).await,
            Self::OAuth(p) => p.embed(texts).await,
        }
    }

    fn dimension(&self) -> Option<usize> {
        match self {
            Self::Ollama(p) => p.dimension(),
            Self::OpenAi(p) => p.dimension(),
            Self::Onnx(p) => p.dimension(),
            Self::Cloudflare(p) => p.dimension(),
            Self::OAuth(p) => p.dimension(),
        }
    }
}

fn default_provider_type() -> ProviderType {
    ProviderType::Ollama
}

/// Provider selection and credentials shared by the CLI and the REST API.
///
/// Flatten into clap commands with `#[command(flatten)]` and into API
/// request types with `#[serde(flatten)]` — one definition, one builder,
/// identical behavior on both surfaces.
#[derive(Debug, Clone, clap::Args, serde::Deserialize)]
pub struct EmbedProviderOpts {
    /// Embedding provider
    #[arg(short = 'p', long, value_enum, default_value_t = ProviderType::Ollama)]
    #[serde(default = "default_provider_type")]
    pub provider: ProviderType,

    /// Model name (provider-specific default if omitted)
    #[arg(short = 'm', long)]
    #[serde(default)]
    pub model: Option<String>,

    /// API key (for OpenAI; also reads OPENAI_API_KEY env or .env.openai file)
    #[arg(long)]
    #[serde(default)]
    pub api_key: Option<String>,

    /// Base URL override for the embedding API
    #[arg(long)]
    #[serde(default)]
    pub base_url: Option<String>,

    /// Path to ONNX model directory (for onnx provider)
    #[arg(long)]
    #[serde(default)]
    pub model_path: Option<String>,

    /// Cloudflare auth token (also reads CLOUDFLARE_AUTH_TOKEN env or .env.cloudflare)
    #[arg(long)]
    #[serde(default)]
    pub cf_auth_token: Option<String>,

    /// Cloudflare account ID (also reads CLOUDFLARE_ACCOUNT_ID env or .env.cloudflare)
    #[arg(long)]
    #[serde(default)]
    pub cf_account_id: Option<String>,

    /// Cloudflare AI Gateway name (optional; also reads CLOUDFLARE_AI_GATEWAY env or .env.cloudflare)
    #[arg(long)]
    #[serde(default)]
    pub cf_ai_gateway: Option<String>,

    /// OAuth token endpoint URL (also reads OAUTH_TOKEN_URL env or .env.oauth)
    #[arg(long)]
    #[serde(default)]
    pub oauth_token_url: Option<String>,

    /// OAuth client ID (also reads OAUTH_CLIENT_ID env or .env.oauth)
    #[arg(long)]
    #[serde(default)]
    pub oauth_client_id: Option<String>,

    /// OAuth client secret (also reads OAUTH_CLIENT_SECRET env or .env.oauth)
    #[arg(long)]
    #[serde(default)]
    pub oauth_client_secret: Option<String>,

    /// OAuth scope (optional; also reads OAUTH_SCOPE env or .env.oauth)
    #[arg(long)]
    #[serde(default)]
    pub oauth_scope: Option<String>,

    /// OAuth base URL for the OpenAI-compatible API (also reads OAUTH_BASE_URL env or .env.oauth)
    #[arg(long)]
    #[serde(default)]
    pub oauth_base_url: Option<String>,

    /// Accept invalid TLS certificates (for corporate proxies with custom CAs)
    #[arg(long)]
    #[serde(default)]
    pub danger_accept_invalid_certs: bool,
}

impl EmbedProviderOpts {
    /// Build the selected provider.
    ///
    /// Cloudflare and OAuth credentials are verified over the network
    /// before returning, so misconfiguration fails fast.
    pub async fn build_provider(&self) -> Result<AnyProvider> {
        match self.provider {
            ProviderType::Ollama => Ok(AnyProvider::Ollama(ollama::OllamaProvider::new(
                self.base_url.clone(),
                self.model.clone(),
            )?)),
            ProviderType::Openai => {
                let api_key = crate::config::resolve_openai_key(&self.api_key)?;
                Ok(AnyProvider::OpenAi(openai::OpenAiProvider::new(
                    api_key,
                    self.base_url.clone(),
                    self.model.clone(),
                )?))
            }
            ProviderType::Onnx => {
                let model_path = self.model_path.as_deref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "model_path is required for the onnx provider. \
                         Provide a directory containing model.onnx and tokenizer.json."
                    )
                })?;
                Ok(AnyProvider::Onnx(Box::new(onnx::OnnxProvider::new(
                    model_path,
                )?)))
            }
            ProviderType::Cloudflare => {
                let (token, account_id, gateway) = cloudflare::resolve_cloudflare_credentials(
                    &self.cf_auth_token,
                    &self.cf_account_id,
                    &self.cf_ai_gateway,
                )?;
                let provider = cloudflare::CloudflareProvider::new(
                    token,
                    account_id,
                    self.model.clone(),
                    gateway,
                )?;
                provider.verify_token().await?;
                Ok(AnyProvider::Cloudflare(provider))
            }
            ProviderType::Oauth => {
                let creds = oauth::resolve_oauth_credentials(
                    &self.oauth_token_url,
                    &self.oauth_client_id,
                    &self.oauth_client_secret,
                    &self.oauth_scope,
                    &self.oauth_base_url,
                    &self.model,
                )?;
                let provider = oauth::OAuthProvider::new(
                    creds.token_url,
                    creds.client_id,
                    creds.client_secret,
                    creds.scope,
                    creds.base_url,
                    creds.model,
                    self.danger_accept_invalid_certs,
                )?;
                provider.verify_credentials().await?;
                Ok(AnyProvider::OAuth(provider))
            }
        }
    }
}

/// Ensure ONNX Runtime's shared library can be discovered before creating an ONNX session.
///
/// The crate is configured with `load-dynamic`, so missing runtimes do not fail at
/// compile time. This preflight check fails fast with a clear message instead of
/// hanging when `Session::builder().commit_from_file()` attempts to load the runtime.
pub fn ensure_onnx_runtime_available() -> Result<PathBuf> {
    let resolved = if let Ok(path) = env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(path.trim());
        resolve_explicit_ort_path(path).ok_or_else(|| {
            anyhow::anyhow!(
                "ORT_DYLIB_PATH is set, but the ONNX Runtime shared library was not found there."
            )
        })?
    } else {
        search_default_ort_locations().ok_or_else(|| {
            anyhow::anyhow!(
                "ONNX Runtime shared library not found. Install it (for example: `brew install onnxruntime`), \
                or set ORT_DYLIB_PATH to the library path (e.g. `.../libonnxruntime.dylib` or `.../onnxruntime.dll`)."
            )
        })?
    };

    // Make discovery deterministic for onnxruntime crate internals by setting the explicit path
    // before any session builder initializes the library.
    if env::var_os("ORT_DYLIB_PATH").is_none() {
        unsafe {
            env::set_var("ORT_DYLIB_PATH", &resolved);
        }
    }

    Ok(resolved)
}

fn resolve_explicit_ort_path(path: PathBuf) -> Option<PathBuf> {
    let library_names = onnx_runtime_library_names();
    if path.is_file() {
        return Some(path);
    }

    if path.is_dir() {
        for name in library_names {
            let candidate = path.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

fn search_default_ort_locations() -> Option<PathBuf> {
    let library_names = onnx_runtime_library_names();
    for dir in candidate_library_dirs() {
        for name in library_names {
            let path = dir.join(name);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn candidate_library_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    dirs.extend(env_search_dirs("LD_LIBRARY_PATH"));

    #[cfg(target_os = "macos")]
    dirs.extend(env_search_dirs("DYLD_LIBRARY_PATH"));

    dirs.extend(env_search_dirs("PATH"));

    #[cfg(target_os = "macos")]
    dirs.extend(["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib"].map(PathBuf::from));
    #[cfg(target_os = "linux")]
    dirs.extend(
        [
            "/usr/lib",
            "/usr/local/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/opt/conda/lib",
            "/usr/lib64",
        ]
        .map(PathBuf::from),
    );

    #[cfg(target_os = "windows")]
    dirs.push(std::env::current_dir().unwrap_or_default());

    dirs
}

fn env_search_dirs(name: &str) -> Vec<PathBuf> {
    env::var(name)
        .ok()
        .unwrap_or_default()
        .split(path_list_separator())
        .filter_map(|entry| {
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(PathBuf::from(trimmed))
            }
        })
        .collect()
}

const fn path_list_separator() -> &'static str {
    if cfg!(target_os = "windows") {
        ";"
    } else {
        ":"
    }
}

#[allow(clippy::needless_return)]
const fn onnx_runtime_library_names() -> &'static [&'static str] {
    #[cfg(target_os = "macos")]
    {
        return &["libonnxruntime.dylib"];
    }
    #[cfg(target_os = "windows")]
    {
        return &["onnxruntime.dll"];
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        return &[
            "libonnxruntime.so",
            "libonnxruntime.so.1",
            "libonnxruntime.so.1.16",
        ];
    }
}
