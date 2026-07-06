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
        let client = crate::http_util::build_client(false)?;

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

        let (status, body) = crate::http_util::send_with_retry(
            self.client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.auth_token)),
            "Cloudflare token verification",
        )
        .await?;

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
        // Always the direct Workers AI endpoint. Gateway routing happens via
        // the `cf-aig-gateway-id` header instead of the
        // gateway.ai.cloudflare.com URL: the URL form forwards the
        // Authorization header through an internal hop that rejects
        // account-owned API tokens, while the header form authenticates
        // exactly like a direct call.
        format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
            self.account_id, self.model
        )
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

/// Maximum inputs per request — Cloudflare Workers AI caps `text` at 100 items.
const MAX_BATCH: usize = 100;

impl EmbeddingProvider for CloudflareProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_BATCH) {
            out.extend(self.embed_batch(batch).await?);
        }
        Ok(out)
    }
}

impl CloudflareProvider {
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        let url = self.endpoint_url();

        let request = EmbeddingRequest { text: texts };

        let mut req_builder = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token));

        // Route through the AI Gateway (logging, rate limiting) via header
        if let Some(ref gateway) = self.ai_gateway {
            req_builder = req_builder.header("cf-aig-gateway-id", gateway);
        }

        let (status, body) = crate::http_util::send_with_retry(
            req_builder.json(&request),
            "Cloudflare AI embeddings",
        )
        .await?;

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
pub fn resolve_cloudflare_credentials(
    auth_token: &Option<String>,
    account_id: &Option<String>,
    ai_gateway: &Option<String>,
) -> Result<(String, String, Option<String>)> {
    use crate::config::resolve_setting;

    const FILE: &str = ".env.cloudflare";
    let token = resolve_setting(auth_token, "CLOUDFLARE_AUTH_TOKEN", FILE);
    let acct = resolve_setting(account_id, "CLOUDFLARE_ACCOUNT_ID", FILE);
    let gw = resolve_setting(ai_gateway, "CLOUDFLARE_AI_GATEWAY", FILE);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gateway_routing_uses_direct_endpoint() {
        // Gateway routing must go via the cf-aig-gateway-id header on the
        // direct Workers AI endpoint. The gateway.ai.cloudflare.com URL form
        // forwards Authorization through an internal hop that rejects
        // account-owned API tokens (401 code 10000) even when the same token
        // works against the direct API.
        let with_gateway = CloudflareProvider::new(
            "token".into(),
            "acct-id".into(),
            None,
            Some("my-gateway".into()),
        )
        .unwrap();
        let without_gateway =
            CloudflareProvider::new("token".into(), "acct-id".into(), None, None).unwrap();

        assert_eq!(with_gateway.endpoint_url(), without_gateway.endpoint_url());
        assert!(
            with_gateway
                .endpoint_url()
                .starts_with("https://api.cloudflare.com/client/v4/accounts/acct-id/ai/run/")
        );
    }
}
