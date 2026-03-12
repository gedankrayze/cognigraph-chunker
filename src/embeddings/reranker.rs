//! Reranker provider trait and implementations for cross-encoder boundary refinement.
//!
//! Cross-encoders score text pairs more accurately than embedding similarity,
//! but are expensive. Used only on ambiguous boundaries (Phase 3).

use super::ensure_onnx_runtime_available;
use anyhow::Result;

/// Trait for reranking providers (cross-encoders).
///
/// Given a query text and candidate documents, returns relevance scores.
/// Higher scores indicate stronger semantic connection.
#[allow(async_fn_in_trait)]
pub trait RerankerProvider: Send + Sync {
    /// Score each document against the query.
    ///
    /// Returns one score per document in the same order as `documents`.
    /// Scores are in [0.0, 1.0] — higher means more relevant/connected.
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>>;

    /// Return the model name for diagnostics.
    fn model_name(&self) -> &str;
}

/// ONNX-based cross-encoder reranker.
///
/// Expects a sequence-classification model directory containing:
/// - `model.onnx` (cross-encoder ONNX model)
/// - `tokenizer.json` (HuggingFace tokenizer config)
///
/// Compatible with models like:
/// - `cross-encoder/ms-marco-MiniLM-L-6-v2`
/// - `BAAI/bge-reranker-base`
pub struct OnnxReranker {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
}

impl OnnxReranker {
    /// Create a new ONNX reranker from a model directory.
    pub fn new(model_path: &str) -> Result<Self> {
        ensure_onnx_runtime_available().context("ONNX Runtime library is not available")?;

        let model_dir = std::path::PathBuf::from(model_path);

        let onnx_path = model_dir.join("model.onnx");
        anyhow::ensure!(
            onnx_path.exists(),
            "ONNX reranker model not found at {}",
            onnx_path.display()
        );

        let tokenizer_path = model_dir.join("tokenizer.json");
        anyhow::ensure!(
            tokenizer_path.exists(),
            "Tokenizer not found at {}",
            tokenizer_path.display()
        );

        let session = ort::session::Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(&onnx_path)
            .context("Failed to load ONNX reranker model")?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let model_name = model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "onnx-reranker".to_string());

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            model_name,
        })
    }

    /// Score a single text pair. Returns a relevance score in [0.0, 1.0].
    fn score_pair(&self, text_a: &str, text_b: &str) -> Result<f64> {
        use anyhow::Context;

        // Cross-encoders encode both texts as a single input (text_a [SEP] text_b)
        let encoding = self
            .tokenizer
            .encode((text_a, text_b), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let seq_len = encoding.get_ids().len();
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

        let input_ids_val = ort::value::Value::from_array(([1, seq_len], input_ids))
            .context("Failed to create input_ids")?;
        let attention_mask_val = ort::value::Value::from_array(([1, seq_len], attention_mask))
            .context("Failed to create attention_mask")?;
        let token_type_ids_val = ort::value::Value::from_array(([1, seq_len], token_type_ids))
            .context("Failed to create token_type_ids")?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("Session lock poisoned: {e}"))?;

        let outputs = session
            .run(ort::inputs![
                input_ids_val,
                attention_mask_val,
                token_type_ids_val
            ])
            .context("ONNX reranker inference failed")?;

        // Output shape: [1, 1] or [1] — logit score
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract reranker output")?;

        let logit = if shape.iter().product::<i64>() >= 1 {
            data[0] as f64
        } else {
            anyhow::bail!("Unexpected reranker output shape: {:?}", shape);
        };

        // Sigmoid to normalize to [0, 1]
        Ok(sigmoid(logit))
    }
}

impl RerankerProvider for OnnxReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        let mut scores = Vec::with_capacity(documents.len());
        for doc in documents {
            scores.push(self.score_pair(query, doc)?);
        }
        Ok(scores)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// NVIDIA NIM reranker
// ---------------------------------------------------------------------------

/// NVIDIA NIM reranker using the `/ranking` endpoint.
///
/// Compatible with models like:
/// - `nv-rerank-qa-mistral-4b:1`
///
/// Requires `NVIDIA_API_KEY` and optionally `NVIDIA_RERANK_BASE_URL`.
pub struct NvidiaReranker {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl NvidiaReranker {
    pub fn new(api_key: String, base_url: Option<String>, model: Option<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .context("Failed to build HTTP client for NVIDIA reranker")?;

        let base = base_url
            .unwrap_or_else(|| "https://ai.api.nvidia.com/v1".to_string())
            .trim_end_matches('/')
            .to_string();

        Ok(Self {
            client,
            api_key,
            base_url: base,
            model: model.unwrap_or_else(|| "nv-rerank-qa-mistral-4b:1".to_string()),
        })
    }

    /// Create from environment variables.
    ///
    /// - `NVIDIA_API_KEY` (required)
    /// - `NVIDIA_RERANK_BASE_URL` (optional, defaults to `https://ai.api.nvidia.com/v1`)
    /// - `NVIDIA_RERANK_MODEL` (optional, defaults to `nv-rerank-qa-mistral-4b:1`)
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("NVIDIA_API_KEY")
            .context("NVIDIA_API_KEY environment variable not set")?;
        let base_url = std::env::var("NVIDIA_RERANK_BASE_URL").ok();
        let model = std::env::var("NVIDIA_RERANK_MODEL").ok();
        Self::new(api_key, base_url, model)
    }
}

#[derive(serde::Serialize)]
struct NvidiaRerankRequest<'a> {
    model: &'a str,
    query: NvidiaQuery<'a>,
    passages: Vec<NvidiaPassage<'a>>,
}

#[derive(serde::Serialize)]
struct NvidiaQuery<'a> {
    text: &'a str,
}

#[derive(serde::Serialize)]
struct NvidiaPassage<'a> {
    text: &'a str,
}

#[derive(serde::Deserialize)]
struct NvidiaRerankResponse {
    rankings: Vec<NvidiaRanking>,
}

#[derive(serde::Deserialize)]
struct NvidiaRanking {
    index: usize,
    logit: f64,
}

/// Derive the NVIDIA NIM reranking endpoint from the model name.
///
/// Legacy mistral model uses `/v1/retrieval/nvidia/reranking`.
/// Newer models embed the model name: `/v1/retrieval/nvidia/{model_slug}/reranking`.
fn nvidia_endpoint(base_url: &str, model: &str) -> String {
    // Legacy models without nvidia/ prefix or the original mistral model
    let slug = model.strip_prefix("nvidia/").unwrap_or(model);
    if slug.starts_with("nv-rerank-qa-mistral") || slug == "rerank-qa-mistral-4b" {
        format!("{}/retrieval/nvidia/reranking", base_url)
    } else {
        format!("{}/retrieval/nvidia/{}/reranking", base_url, slug)
    }
}

impl RerankerProvider for NvidiaReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let url = nvidia_endpoint(&self.base_url, &self.model);

        let request = NvidiaRerankRequest {
            model: &self.model,
            query: NvidiaQuery { text: query },
            passages: documents
                .iter()
                .map(|&d| NvidiaPassage { text: d })
                .collect(),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to NVIDIA rerank API")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read NVIDIA response body")?;

        if !status.is_success() {
            anyhow::bail!("NVIDIA rerank API error ({}): {}", status, body);
        }

        let parsed: NvidiaRerankResponse =
            serde_json::from_str(&body).context("Failed to parse NVIDIA rerank response")?;

        // Map rankings back to input order, applying sigmoid to logits
        let mut scores = vec![0.0f64; documents.len()];
        for ranking in parsed.rankings {
            if ranking.index < scores.len() {
                scores[ranking.index] = sigmoid(ranking.logit);
            }
        }

        Ok(scores)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// Cohere reranker
// ---------------------------------------------------------------------------

/// Cohere reranker using the `/rerank` endpoint.
///
/// Compatible with models like:
/// - `rerank-v3.5`
/// - `rerank-english-v3.0`
/// - `rerank-multilingual-v3.0`
///
/// Requires `COHERE_API_KEY` and optionally `COHERE_RERANK_MODEL`.
pub struct CohereReranker {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl CohereReranker {
    pub fn new(api_key: String, base_url: Option<String>, model: Option<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .context("Failed to build HTTP client for Cohere reranker")?;

        Ok(Self {
            client,
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.cohere.com/v2".to_string()),
            model: model.unwrap_or_else(|| "rerank-v3.5".to_string()),
        })
    }

    /// Create from environment variables.
    ///
    /// - `COHERE_API_KEY` (required)
    /// - `COHERE_RERANK_BASE_URL` (optional, defaults to `https://api.cohere.com/v2`)
    /// - `COHERE_RERANK_MODEL` (optional, defaults to `rerank-v3.5`)
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("COHERE_API_KEY")
            .context("COHERE_API_KEY environment variable not set")?;
        let base_url = std::env::var("COHERE_RERANK_BASE_URL").ok();
        let model = std::env::var("COHERE_RERANK_MODEL").ok();
        Self::new(api_key, base_url, model)
    }
}

#[derive(serde::Serialize)]
struct CohereRerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [&'a str],
    return_documents: bool,
}

#[derive(serde::Deserialize)]
struct CohereRerankResponse {
    results: Vec<CohereResult>,
}

#[derive(serde::Deserialize)]
struct CohereResult {
    index: usize,
    relevance_score: f64,
}

impl RerankerProvider for CohereReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let url = format!("{}/rerank", self.base_url);

        let request = CohereRerankRequest {
            model: &self.model,
            query,
            documents,
            return_documents: false,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Cohere rerank API")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read Cohere response body")?;

        if !status.is_success() {
            anyhow::bail!("Cohere rerank API error ({}): {}", status, body);
        }

        let parsed: CohereRerankResponse =
            serde_json::from_str(&body).context("Failed to parse Cohere rerank response")?;

        // Cohere returns scores already in [0, 1], map back to input order
        let mut scores = vec![0.0f64; documents.len()];
        for result in parsed.results {
            if result.index < scores.len() {
                scores[result.index] = result.relevance_score;
            }
        }

        Ok(scores)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// Cloudflare Workers AI reranker
// ---------------------------------------------------------------------------

/// Cloudflare Workers AI reranker using the `/ai/run/` endpoint.
///
/// Compatible with models like:
/// - `@cf/baai/bge-reranker-base`
///
/// Requires `CLOUDFLARE_AUTH_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`.
/// Optionally routes through AI Gateway when `CLOUDFLARE_AI_GATEWAY` is set.
pub struct CloudflareReranker {
    client: reqwest::Client,
    auth_token: String,
    account_id: String,
    model: String,
    ai_gateway: Option<String>,
}

impl CloudflareReranker {
    pub fn new(
        auth_token: String,
        account_id: String,
        model: Option<String>,
        ai_gateway: Option<String>,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .context("Failed to build HTTP client for Cloudflare reranker")?;

        Ok(Self {
            client,
            auth_token,
            account_id,
            model: model.unwrap_or_else(|| "@cf/baai/bge-reranker-base".to_string()),
            ai_gateway,
        })
    }

    /// Create from environment variables or `.env.cloudflare` file.
    ///
    /// Uses the same credentials as the Cloudflare embedding provider.
    pub fn from_env() -> Result<Self> {
        let (token, account_id, gateway) =
            super::cloudflare::resolve_cloudflare_credentials(&None, &None, &None)?;
        let model = std::env::var("CLOUDFLARE_RERANK_MODEL").ok();
        Self::new(token, account_id, model, gateway)
    }

    fn endpoint_url(&self) -> String {
        match &self.ai_gateway {
            Some(gateway) => {
                format!(
                    "https://gateway.ai.cloudflare.com/v1/{}/{}/workers-ai/{}",
                    self.account_id, gateway, self.model
                )
            }
            None => {
                format!(
                    "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
                    self.account_id, self.model
                )
            }
        }
    }
}

#[derive(serde::Serialize)]
struct CfRerankContext<'a> {
    text: &'a str,
}

#[derive(serde::Serialize)]
struct CfRerankRequest<'a> {
    query: &'a str,
    contexts: Vec<CfRerankContext<'a>>,
}

/// Cloudflare wraps AI responses in `{ result: ..., success: bool, errors: [...] }`.
/// The `result` field contains the model's actual output.
#[derive(serde::Deserialize)]
struct CfRerankResponse {
    result: Option<CfRerankResult>,
    success: bool,
    #[serde(default)]
    errors: Vec<CfRerankError>,
}

/// The model returns `{ response: [{ id, score }, ...] }`.
#[derive(serde::Deserialize)]
struct CfRerankResult {
    response: Vec<CfRerankScore>,
}

#[derive(serde::Deserialize)]
struct CfRerankScore {
    id: usize,
    score: f64,
}

#[derive(serde::Deserialize)]
struct CfRerankError {
    message: String,
}

impl RerankerProvider for CloudflareReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let url = self.endpoint_url();

        let request = CfRerankRequest {
            query,
            contexts: documents
                .iter()
                .map(|&d| CfRerankContext { text: d })
                .collect(),
        };

        let mut req_builder = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token));

        if self.ai_gateway.is_some() {
            req_builder = req_builder.header(
                "cf-aig-authorization",
                format!("Bearer {}", self.auth_token),
            );
        }

        let response = req_builder
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Cloudflare rerank API")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read Cloudflare rerank response body")?;

        if !status.is_success() {
            anyhow::bail!("Cloudflare rerank API error ({}): {}", status, body);
        }

        let parsed: CfRerankResponse =
            serde_json::from_str(&body).context("Failed to parse Cloudflare rerank response")?;

        if !parsed.success {
            let msgs: Vec<String> = parsed.errors.iter().map(|e| e.message.clone()).collect();
            anyhow::bail!("Cloudflare rerank error: {}", msgs.join("; "));
        }

        let result = parsed
            .result
            .ok_or_else(|| anyhow::anyhow!("Cloudflare rerank returned no result"))?;

        // Map scores back to input order using the `id` field (context index)
        let mut scores = vec![0.0f64; documents.len()];
        for entry in result.response {
            if entry.id < scores.len() {
                scores[entry.id] = entry.score;
            }
        }

        Ok(scores)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// OAuth-authenticated reranker
// ---------------------------------------------------------------------------

/// OAuth-authenticated reranker for corporate API gateways.
///
/// Uses the same OAuth2 client credentials flow as the embedding provider,
/// hitting a configurable rerank endpoint. Uses Cohere-compatible request/response
/// format (`{ model, query, documents }` → `{ results: [{ index, relevance_score }] }`).
///
/// The endpoint path is configurable via `OAUTH_RERANK_PATH` (default: `/rerank`).
pub struct OAuthReranker {
    client: reqwest::Client,
    token_url: String,
    client_id: String,
    client_secret: String,
    scope: Option<String>,
    base_url: String,
    rerank_path: String,
    model: String,
    token_cache: std::sync::Mutex<Option<OAuthCachedToken>>,
}

struct OAuthCachedToken {
    access_token: String,
    expires_at: std::time::Instant,
}

impl OAuthReranker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        token_url: String,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
        base_url: String,
        rerank_path: Option<String>,
        model: Option<String>,
        danger_accept_invalid_certs: bool,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(120))
            .danger_accept_invalid_certs(danger_accept_invalid_certs)
            .build()
            .context("Failed to build HTTP client for OAuth reranker")?;

        Ok(Self {
            client,
            token_url,
            client_id,
            client_secret,
            scope,
            base_url: base_url.trim_end_matches('/').to_string(),
            rerank_path: rerank_path.unwrap_or_else(|| "/rerank".to_string()),
            model: model.unwrap_or_else(|| "rerank-default".to_string()),
            token_cache: std::sync::Mutex::new(None),
        })
    }

    /// Create from environment variables or `.env.oauth` file.
    ///
    /// Reuses the same credentials as the OAuth embedding provider, plus:
    /// - `OAUTH_RERANK_PATH` (optional, default: `/rerank`)
    /// - `OAUTH_RERANK_MODEL` (optional)
    pub fn from_env(danger_accept_invalid_certs: bool) -> Result<Self> {
        let creds =
            super::oauth::resolve_oauth_credentials(&None, &None, &None, &None, &None, &None)?;
        let rerank_path = std::env::var("OAUTH_RERANK_PATH").ok();
        let model = std::env::var("OAUTH_RERANK_MODEL").ok();
        Self::new(
            creds.token_url,
            creds.client_id,
            creds.client_secret,
            creds.scope,
            creds.base_url,
            rerank_path,
            model,
            danger_accept_invalid_certs,
        )
    }

    /// Acquire a valid access token, using the cache if not expired.
    async fn get_token(&self) -> Result<String> {
        {
            let cache = self
                .token_cache
                .lock()
                .map_err(|e| anyhow::anyhow!("Token cache lock poisoned: {e}"))?;
            if let Some(ref cached) = *cache
                && std::time::Instant::now() + std::time::Duration::from_secs(60)
                    < cached.expires_at
            {
                return Ok(cached.access_token.clone());
            }
        }

        let mut form = vec![
            ("grant_type", "client_credentials"),
            ("client_id", &*self.client_id),
            ("client_secret", &*self.client_secret),
        ];
        let scope_val;
        if let Some(ref s) = self.scope {
            scope_val = s.clone();
            form.push(("scope", &scope_val));
        }

        let response = self
            .client
            .post(&self.token_url)
            .form(&form)
            .send()
            .await
            .context("Failed to request OAuth token for reranker")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read OAuth token response")?;

        if !status.is_success() {
            anyhow::bail!("OAuth token request failed ({}): {}", status, body);
        }

        let token_resp: OAuthTokenResponse =
            serde_json::from_str(&body).context("Failed to parse OAuth token response")?;

        let expires_at =
            std::time::Instant::now() + std::time::Duration::from_secs(token_resp.expires_in);
        let access_token = token_resp.access_token.clone();

        let mut cache = self
            .token_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("Token cache lock poisoned: {e}"))?;
        *cache = Some(OAuthCachedToken {
            access_token: token_resp.access_token,
            expires_at,
        });

        Ok(access_token)
    }

    /// Verify that we can acquire a token.
    pub async fn verify_credentials(&self) -> Result<()> {
        self.get_token().await?;
        Ok(())
    }
}

#[derive(serde::Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    expires_in: u64,
}

/// Request format: Cohere-compatible `{ model, query, documents }`.
#[derive(serde::Serialize)]
struct OAuthRerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [&'a str],
}

/// Response format: `{ results: [{ index, relevance_score }] }`.
#[derive(serde::Deserialize)]
struct OAuthRerankResponse {
    results: Vec<OAuthRerankResult>,
}

#[derive(serde::Deserialize)]
struct OAuthRerankResult {
    index: usize,
    relevance_score: f64,
}

impl RerankerProvider for OAuthReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let token = self.get_token().await?;
        let url = format!("{}{}", self.base_url, self.rerank_path);

        let request = OAuthRerankRequest {
            model: &self.model,
            query,
            documents,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to OAuth rerank API")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read OAuth rerank response body")?;

        if !status.is_success() {
            anyhow::bail!("OAuth rerank API error ({}): {}", status, body);
        }

        let parsed: OAuthRerankResponse =
            serde_json::from_str(&body).context("Failed to parse OAuth rerank response")?;

        let mut scores = vec![0.0f64; documents.len()];
        for result in parsed.results {
            if result.index < scores.len() {
                scores[result.index] = result.relevance_score;
            }
        }

        Ok(scores)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// Enum dispatch for runtime reranker selection
// ---------------------------------------------------------------------------

/// Runtime-selectable reranker that wraps any provider.
///
/// Needed because `async fn` in traits prevents `dyn` dispatch.
pub enum AnyReranker {
    Onnx(Box<OnnxReranker>),
    Nvidia(NvidiaReranker),
    Cohere(CohereReranker),
    Cloudflare(CloudflareReranker),
    OAuth(OAuthReranker),
}

impl RerankerProvider for AnyReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        match self {
            Self::Onnx(r) => r.rerank(query, documents).await,
            Self::Nvidia(r) => r.rerank(query, documents).await,
            Self::Cohere(r) => r.rerank(query, documents).await,
            Self::Cloudflare(r) => r.rerank(query, documents).await,
            Self::OAuth(r) => r.rerank(query, documents).await,
        }
    }

    fn model_name(&self) -> &str {
        match self {
            Self::Onnx(r) => r.model_name(),
            Self::Nvidia(r) => r.model_name(),
            Self::Cohere(r) => r.model_name(),
            Self::Cloudflare(r) => r.model_name(),
            Self::OAuth(r) => r.model_name(),
        }
    }
}

use anyhow::Context;
