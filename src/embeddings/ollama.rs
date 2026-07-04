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

/// Maximum inputs per request — bounds request size and server memory.
const MAX_BATCH: usize = 256;

impl EmbeddingProvider for OllamaProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_BATCH) {
            out.extend(self.embed_batch(batch).await?);
        }
        Ok(out)
    }
}

impl OllamaProvider {
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
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

    /// Read a full HTTP request (headers + Content-Length body) from a socket.
    async fn read_request(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = socket.read(&mut tmp).await.unwrap_or(0);
            if n == 0 {
                return buf;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
                let content_length: usize = headers
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                let body_have = buf.len() - (pos + 4);
                if body_have >= content_length {
                    return buf;
                }
            }
        }
    }

    /// Ollama-compatible embed endpoint: echoes one embedding per input,
    /// where each embedding is [input_text_length]. Counts requests.
    async fn spawn_embed_server() -> (String, Arc<AtomicUsize>) {
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
                let raw = read_request(&mut socket).await;
                let body_start = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                let req: serde_json::Value = serde_json::from_slice(&raw[body_start..]).unwrap();
                let embeddings: Vec<Vec<f64>> = req["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|t| vec![t.as_str().unwrap().len() as f64])
                    .collect();
                let body = serde_json::json!({ "embeddings": embeddings }).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });

        (format!("http://{addr}"), counter)
    }

    #[tokio::test]
    async fn test_embed_splits_oversized_batches() {
        // 600 inputs must be split into ceil(600/256) = 3 requests, with
        // results concatenated in input order.
        let (base_url, requests) = spawn_embed_server().await;
        let provider = OllamaProvider::new(Some(base_url), None).unwrap();

        let texts: Vec<String> = (0..600).map(|i| "a".repeat(i % 50 + 1)).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        let embeddings = provider.embed(&refs).await.unwrap();

        assert_eq!(embeddings.len(), 600);
        for (i, emb) in embeddings.iter().enumerate() {
            assert_eq!(
                emb[0],
                (i % 50 + 1) as f64,
                "embedding {i} out of order across batch boundaries"
            );
        }
        assert_eq!(
            requests.load(Ordering::SeqCst),
            3,
            "600 inputs with a 256 batch limit must produce 3 requests"
        );
    }

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
