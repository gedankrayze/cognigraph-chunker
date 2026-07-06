//! OAuth-authenticated OpenAI-compatible embeddings provider.
//!
//! Acquires a bearer token via OAuth2 client credentials grant, then uses it
//! to call an OpenAI-compatible `/embeddings` endpoint. The token is cached
//! and automatically refreshed before expiry.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::EmbeddingProvider;
use super::oauth_token::OAuthTokenSource;

/// OAuth-authenticated OpenAI-compatible embeddings provider.
pub struct OAuthProvider {
    client: reqwest::Client,
    token_source: OAuthTokenSource,
    base_url: String,
    model: String,
}

impl OAuthProvider {
    pub fn new(
        token_url: String,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
        base_url: String,
        model: Option<String>,
        danger_accept_invalid_certs: bool,
    ) -> Result<Self> {
        let client = crate::http_util::build_client(danger_accept_invalid_certs)?;
        let token_source =
            OAuthTokenSource::new(client.clone(), token_url, client_id, client_secret, scope);

        Ok(Self {
            client,
            token_source,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.unwrap_or_else(|| "text-embedding-3-small".to_string()),
        })
    }

    /// Verify that we can acquire a token.
    pub async fn verify_credentials(&self) -> Result<()> {
        self.token_source.get_token().await.map(|_| ())
    }
}

// -- OpenAI-compatible embedding types --

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f64>,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

/// Maximum inputs per request — OpenAI-compatible APIs cap at 2048 inputs.
const MAX_BATCH: usize = 2048;

impl EmbeddingProvider for OAuthProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_BATCH) {
            out.extend(self.embed_batch(batch).await?);
        }
        Ok(out)
    }
}

impl OAuthProvider {
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        let token = self.token_source.get_token().await?;
        let url = format!("{}/embeddings", self.base_url);

        let request = EmbeddingRequest {
            model: &self.model,
            input: texts,
        };

        let (status, body) = crate::http_util::send_with_retry(
            self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", token))
                .json(&request),
            "OAuth embeddings",
        )
        .await?;

        if !status.is_success() {
            if let Ok(err) = serde_json::from_str::<ErrorResponse>(&body) {
                bail!(
                    "OAuth embedding API error ({}): {}",
                    status,
                    err.error.message
                );
            }
            bail!("OAuth embedding API error ({}): {}", status, body);
        }

        let parsed: EmbeddingResponse =
            serde_json::from_str(&body).context("Failed to parse OAuth embedding response")?;

        let embeddings: Vec<Vec<f64>> = parsed.data.into_iter().map(|d| d.embedding).collect();

        if embeddings.len() != texts.len() {
            bail!(
                "OAuth embedding API returned {} embeddings for {} inputs",
                embeddings.len(),
                texts.len()
            );
        }

        Ok(embeddings)
    }
}

/// Resolved OAuth credentials.
pub struct OAuthCredentials {
    pub token_url: String,
    pub client_id: String,
    pub client_secret: String,
    pub scope: Option<String>,
    pub base_url: String,
    pub model: Option<String>,
}

/// Resolve OAuth credentials from args, env vars, or `.env.oauth` file.
pub fn resolve_oauth_credentials(
    token_url: &Option<String>,
    client_id: &Option<String>,
    client_secret: &Option<String>,
    scope: &Option<String>,
    base_url: &Option<String>,
    model: &Option<String>,
) -> Result<OAuthCredentials> {
    use crate::config::resolve_setting;

    const FILE: &str = ".env.oauth";
    let t_url = resolve_setting(token_url, "OAUTH_TOKEN_URL", FILE);
    let c_id = resolve_setting(client_id, "OAUTH_CLIENT_ID", FILE);
    let c_secret = resolve_setting(client_secret, "OAUTH_CLIENT_SECRET", FILE);
    let sc = resolve_setting(scope, "OAUTH_SCOPE", FILE);
    let b_url = resolve_setting(base_url, "OAUTH_BASE_URL", FILE);
    let mdl = resolve_setting(model, "OAUTH_MODEL", FILE);

    let t_url = t_url.ok_or_else(|| {
        anyhow::anyhow!(
            "OAuth token URL not found.\n\
             Provide it via --oauth-token-url, OAUTH_TOKEN_URL env var, or .env.oauth file."
        )
    })?;

    let c_id = c_id.ok_or_else(|| {
        anyhow::anyhow!(
            "OAuth client ID not found.\n\
             Provide it via --oauth-client-id, OAUTH_CLIENT_ID env var, or .env.oauth file."
        )
    })?;

    let c_secret = c_secret.ok_or_else(|| {
        anyhow::anyhow!(
            "OAuth client secret not found.\n\
             Provide it via --oauth-client-secret, OAUTH_CLIENT_SECRET env var, or .env.oauth file."
        )
    })?;

    let b_url = b_url.ok_or_else(|| {
        anyhow::anyhow!(
            "OAuth base URL not found.\n\
             Provide it via --oauth-base-url, OAUTH_BASE_URL env var, or .env.oauth file."
        )
    })?;

    Ok(OAuthCredentials {
        token_url: t_url,
        client_id: c_id,
        client_secret: c_secret,
        scope: sc,
        base_url: b_url,
        model: mdl,
    })
}
