//! Cloudflare Workers AI embeddings provider.
//!
//! Supports both direct Cloudflare AI API and AI Gateway routing.
//! Credentials are resolved from constructor args, environment variables,
//! or the `.env.cloudflare` file.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::EmbeddingProvider;

/// Cloudflare Workers AI embeddings provider.
///
/// When `ai_gateway` is `None`, requests go directly to the Cloudflare AI API:
///   `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}`
///
/// When `ai_gateway` is `Some(gateway)`, requests are routed through AI Gateway:
///   `https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway}/{model}`
pub struct CloudflareProvider {
    client: reqwest::Client,
    auth_token: String,
    account_id: String,
    model: String,
    ai_gateway: Option<String>,
}

impl CloudflareProvider {
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
            .context("Failed to build HTTP client for Cloudflare provider")?;

        Ok(Self {
            client,
            auth_token,
            account_id,
            model: model.unwrap_or_else(|| "@cf/baai/bge-m3".to_string()),
            ai_gateway,
        })
    }

    /// Verify the auth token against the Cloudflare API.
    ///
    /// Returns `Ok(())` if the token is valid and active, or an error otherwise.
    pub async fn verify_token(&self) -> Result<()> {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/tokens/verify",
            self.account_id
        );

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .send()
            .await
            .context("Failed to verify Cloudflare auth token")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read Cloudflare token verification response")?;

        if !status.is_success() {
            bail!(
                "Cloudflare token verification failed ({}): {}",
                status,
                body
            );
        }

        let parsed: VerifyResponse =
            serde_json::from_str(&body).context("Failed to parse Cloudflare verify response")?;

        if !parsed.success {
            let msgs: Vec<String> = parsed.errors.iter().map(|e| e.message.clone()).collect();
            bail!("Cloudflare token is not valid: {}", msgs.join("; "));
        }

        if parsed.result.status != "active" {
            bail!(
                "Cloudflare token status is '{}', expected 'active'",
                parsed.result.status
            );
        }

        Ok(())
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

// -- Token verification types --

#[derive(Deserialize)]
struct VerifyResponse {
    result: VerifyResult,
    success: bool,
    #[serde(default)]
    errors: Vec<CfMessage>,
}

#[derive(Deserialize)]
struct VerifyResult {
    status: String,
}

#[derive(Deserialize)]
struct CfMessage {
    message: String,
}

// -- Embedding request/response types --

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    text: &'a [&'a str],
}

#[derive(Deserialize)]
struct CfAiResponse {
    result: Option<CfEmbeddingResult>,
    success: bool,
    #[serde(default)]
    errors: Vec<CfMessage>,
}

#[derive(Deserialize)]
struct CfEmbeddingResult {
    data: Vec<Vec<f64>>,
}

impl EmbeddingProvider for CloudflareProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = self.endpoint_url();

        let request = EmbeddingRequest { text: texts };

        let mut req_builder = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token));

        // AI Gateway requires its own auth header in addition to the provider auth
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
            .context("Failed to send request to Cloudflare AI embeddings API")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read Cloudflare response body")?;

        if !status.is_success() {
            bail!("Cloudflare AI API error ({}): {}", status, body);
        }

        let parsed: CfAiResponse = serde_json::from_str(&body)
            .context("Failed to parse Cloudflare AI embeddings response")?;

        if !parsed.success {
            let msgs: Vec<String> = parsed.errors.iter().map(|e| e.message.clone()).collect();
            bail!("Cloudflare AI error: {}", msgs.join("; "));
        }

        let result = parsed
            .result
            .ok_or_else(|| anyhow::anyhow!("Cloudflare AI returned no result"))?;

        if result.data.len() != texts.len() {
            bail!(
                "Cloudflare AI returned {} embeddings for {} inputs",
                result.data.len(),
                texts.len()
            );
        }

        Ok(result.data)
    }
}

/// Resolve Cloudflare credentials from args, env vars, or `.env.cloudflare` file.
///
/// Returns `(auth_token, account_id, ai_gateway)`.
/// Try to read a non-empty env var, returning `Some` if found.
fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

pub fn resolve_cloudflare_credentials(
    auth_token: &Option<String>,
    account_id: &Option<String>,
    ai_gateway: &Option<String>,
) -> Result<(String, String, Option<String>)> {
    let mut token = auth_token
        .clone()
        .or_else(|| env_non_empty("CLOUDFLARE_AUTH_TOKEN"));
    let mut acct = account_id
        .clone()
        .or_else(|| env_non_empty("CLOUDFLARE_ACCOUNT_ID"));
    let mut gw = ai_gateway
        .clone()
        .or_else(|| env_non_empty("CLOUDFLARE_AI_GATEWAY"));

    // Try .env.cloudflare file for any still-missing values
    if (token.is_none() || acct.is_none() || gw.is_none())
        && let Ok(content) = std::fs::read_to_string(".env.cloudflare")
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
                "CLOUDFLARE_AUTH_TOKEN" if token.is_none() => token = Some(val.to_string()),
                "CLOUDFLARE_ACCOUNT_ID" if acct.is_none() => acct = Some(val.to_string()),
                "CLOUDFLARE_AI_GATEWAY" if gw.is_none() => gw = Some(val.to_string()),
                _ => {}
            }
        }
    }

    let token = token.ok_or_else(|| {
        anyhow::anyhow!(
            "Cloudflare auth token not found.\n\
             Provide it via --cf-auth-token, CLOUDFLARE_AUTH_TOKEN env var, or .env.cloudflare file."
        )
    })?;

    let acct = acct.ok_or_else(|| {
        anyhow::anyhow!(
            "Cloudflare account ID not found.\n\
             Provide it via --cf-account-id, CLOUDFLARE_ACCOUNT_ID env var, or .env.cloudflare file."
        )
    })?;

    Ok((token, acct, gw))
}
