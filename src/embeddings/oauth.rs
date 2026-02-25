//! OAuth-authenticated OpenAI-compatible embeddings provider.
//!
//! Acquires a bearer token via OAuth2 client credentials grant, then uses it
//! to call an OpenAI-compatible `/embeddings` endpoint. The token is cached
//! and automatically refreshed before expiry.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::EmbeddingProvider;

/// OAuth-authenticated OpenAI-compatible embeddings provider.
pub struct OAuthProvider {
    client: reqwest::Client,
    token_url: String,
    client_id: String,
    client_secret: String,
    scope: Option<String>,
    base_url: String,
    model: String,
    token_cache: Mutex<Option<CachedToken>>,
}

struct CachedToken {
    access_token: String,
    expires_at: Instant,
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
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(120))
            .danger_accept_invalid_certs(danger_accept_invalid_certs)
            .build()
            .context("Failed to build HTTP client for OAuth provider")?;

        Ok(Self {
            client,
            token_url,
            client_id,
            client_secret,
            scope,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.unwrap_or_else(|| "text-embedding-3-small".to_string()),
            token_cache: Mutex::new(None),
        })
    }

    /// Acquire a valid access token, using the cache if not expired.
    async fn get_token(&self) -> Result<String> {
        // Check cache (with 60s safety margin before expiry)
        {
            let cache = self
                .token_cache
                .lock()
                .map_err(|e| anyhow::anyhow!("Token cache lock poisoned: {e}"))?;
            if let Some(ref cached) = *cache
                && Instant::now() + Duration::from_secs(60) < cached.expires_at
            {
                return Ok(cached.access_token.clone());
            }
        }

        // Fetch new token
        let mut form = vec![
            ("grant_type", "client_credentials"),
            ("client_id", &self.client_id),
            ("client_secret", &self.client_secret),
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
            .context("Failed to request OAuth token")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read OAuth token response")?;

        if !status.is_success() {
            bail!("OAuth token request failed ({}): {}", status, body);
        }

        let token_resp: TokenResponse =
            serde_json::from_str(&body).context("Failed to parse OAuth token response")?;

        let expires_at = Instant::now() + Duration::from_secs(token_resp.expires_in);
        let access_token = token_resp.access_token.clone();

        // Update cache
        let mut cache = self
            .token_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("Token cache lock poisoned: {e}"))?;
        *cache = Some(CachedToken {
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

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
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

impl EmbeddingProvider for OAuthProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let token = self.get_token().await?;
        let url = format!("{}/embeddings", self.base_url);

        let request = EmbeddingRequest {
            model: &self.model,
            input: texts,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .json(&request)
            .send()
            .await
            .context("Failed to send request to OAuth embedding API")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read OAuth embedding response body")?;

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

/// Try to read a non-empty env var, returning `Some` if found.
fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
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
    let mut t_url = token_url
        .clone()
        .or_else(|| env_non_empty("OAUTH_TOKEN_URL"));
    let mut c_id = client_id
        .clone()
        .or_else(|| env_non_empty("OAUTH_CLIENT_ID"));
    let mut c_secret = client_secret
        .clone()
        .or_else(|| env_non_empty("OAUTH_CLIENT_SECRET"));
    let mut sc = scope.clone().or_else(|| env_non_empty("OAUTH_SCOPE"));
    let mut b_url = base_url.clone().or_else(|| env_non_empty("OAUTH_BASE_URL"));
    let mut mdl = model.clone().or_else(|| env_non_empty("OAUTH_MODEL"));

    // Try .env.oauth file for any still-missing values
    if (t_url.is_none() || c_id.is_none() || c_secret.is_none() || b_url.is_none())
        && let Ok(content) = std::fs::read_to_string(".env.oauth")
    {
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let Some((key, val)) = line.split_once('=') else {
                continue;
            };
            let val = val.trim();
            if val.is_empty() {
                continue;
            }
            match key.trim() {
                "OAUTH_TOKEN_URL" if t_url.is_none() => t_url = Some(val.to_string()),
                "OAUTH_CLIENT_ID" if c_id.is_none() => c_id = Some(val.to_string()),
                "OAUTH_CLIENT_SECRET" if c_secret.is_none() => c_secret = Some(val.to_string()),
                "OAUTH_SCOPE" if sc.is_none() => sc = Some(val.to_string()),
                "OAUTH_BASE_URL" if b_url.is_none() => b_url = Some(val.to_string()),
                "OAUTH_MODEL" if mdl.is_none() => mdl = Some(val.to_string()),
                _ => {}
            }
        }
    }

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
