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
    pub fn new(base_url: Option<String>, model: Option<String>) -> anyhow::Result<Self> {
        let client = crate::http_util::build_client(false)?;
        Ok(Self {
            client,
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
            model: model.unwrap_or_else(|| "nomic-embed-text".to_string()),
        })
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

        let (status, body) = crate::http_util::send_with_retry(
            self.client.post(&url).json(&request),
            "Ollama embeddings",
        )
        .await?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Spawn a local HTTP server that answers every request with a 302
    /// redirect back to itself and counts the requests it receives.
    async fn spawn_redirecting_server() -> (String, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let server_counter = counter.clone();

        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                server_counter.fetch_add(1, Ordering::SeqCst);
                let mut buf = [0u8; 4096];
                let _ = socket.read(&mut buf).await;
                let response = format!(
                    "HTTP/1.1 302 Found\r\nLocation: http://{addr}/elsewhere\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });

        (format!("http://{addr}"), counter)
    }

    #[tokio::test]
    async fn test_embed_does_not_follow_redirects() {
        // A redirect from a validated host would bypass the API server's SSRF
        // check (validate only sees the first hop). The client must surface
        // the 3xx as an error instead of following it.
        let (base_url, requests) = spawn_redirecting_server().await;
        let provider = OllamaProvider::new(Some(base_url), None).unwrap();

        let result = provider.embed(&["hello"]).await;

        assert!(result.is_err(), "a 302 response must be an error");
        assert_eq!(
            requests.load(Ordering::SeqCst),
            1,
            "client must not follow the redirect (each hop is a new request)"
        );
    }
}
