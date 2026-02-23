//! Ollama local embeddings provider.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::EmbeddingProvider;

/// Ollama embeddings provider for local model inference.
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaProvider {
    pub fn new(base_url: Option<String>, model: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
            model: model.unwrap_or_else(|| "nomic-embed-text".to_string()),
        }
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct OllamaError {
    error: String,
}

impl EmbeddingProvider for OllamaProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = format!("{}/api/embed", self.base_url);

        let request = EmbedRequest {
            model: &self.model,
            input: texts,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to connect to Ollama. Is it running?")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read Ollama response body")?;

        if !status.is_success() {
            if let Ok(err) = serde_json::from_str::<OllamaError>(&body) {
                bail!("Ollama error ({}): {}", status, err.error);
            }
            bail!("Ollama error ({}): {}", status, body);
        }

        let parsed: EmbedResponse =
            serde_json::from_str(&body).context("Failed to parse Ollama embeddings response")?;

        if parsed.embeddings.len() != texts.len() {
            bail!(
                "Ollama returned {} embeddings for {} inputs",
                parsed.embeddings.len(),
                texts.len()
            );
        }

        Ok(parsed.embeddings)
    }
}
