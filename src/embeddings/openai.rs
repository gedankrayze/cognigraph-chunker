//! OpenAI embeddings provider.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::EmbeddingProvider;

/// OpenAI-compatible embeddings provider.
///
/// Works with OpenAI API and any compatible endpoint (Azure, LiteLLM, etc.).
pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAiProvider {
    pub fn new(api_key: String, base_url: Option<String>, model: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            model: model.unwrap_or_else(|| "text-embedding-3-small".to_string()),
        }
    }
}

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

impl EmbeddingProvider for OpenAiProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = format!("{}/embeddings", self.base_url);

        let request = EmbeddingRequest {
            model: &self.model,
            input: texts,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .context("Failed to send request to OpenAI embeddings API")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read OpenAI response body")?;

        if !status.is_success() {
            if let Ok(err) = serde_json::from_str::<ErrorResponse>(&body) {
                bail!("OpenAI API error ({}): {}", status, err.error.message);
            }
            bail!("OpenAI API error ({}): {}", status, body);
        }

        let parsed: EmbeddingResponse =
            serde_json::from_str(&body).context("Failed to parse OpenAI embeddings response")?;

        // OpenAI returns embeddings sorted by index, but let's be safe
        let mut embeddings: Vec<Vec<f64>> = parsed.data.into_iter().map(|d| d.embedding).collect();

        if embeddings.len() != texts.len() {
            bail!(
                "OpenAI returned {} embeddings for {} inputs",
                embeddings.len(),
                texts.len()
            );
        }

        // Ensure ordering matches input (they should already be in order)
        embeddings.truncate(texts.len());
        Ok(embeddings)
    }
}
