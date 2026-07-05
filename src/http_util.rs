//! Shared HTTP client construction and request sending with retry.
//!
//! Every provider previously built its own client (9 copies of the same
//! builder) and sent requests fire-once. This module is the single place
//! for client policy (timeouts, no redirects) and transient-failure retry.

use std::sync::Once;
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::StatusCode;

const MAX_ATTEMPTS: u32 = 3;

/// Install the rustls `ring` crypto provider as the process default.
///
/// reqwest is built with `rustls-no-provider` (to avoid aws-lc-sys, which
/// doesn't cross-compile for aarch64), so a provider must be installed
/// before the first TLS client is built. Every client goes through
/// `build_client`, making this the single, race-free init point.
fn ensure_crypto_provider() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // Ignore the error: a host application may have already installed a
        // provider, which is fine.
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Build the standard HTTP client used by all providers:
/// no redirects (SSRF defense), 30s connect / 120s total timeouts.
pub fn build_client(danger_accept_invalid_certs: bool) -> Result<reqwest::Client> {
    ensure_crypto_provider();
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(120))
        .danger_accept_invalid_certs(danger_accept_invalid_certs)
        .build()
        .context("Failed to build HTTP client")
}

/// Send a request, retrying transient failures with exponential backoff.
///
/// Retries connect/timeout errors and 429/500/502/503/504 responses (up to
/// 3 attempts total), honoring `Retry-After` seconds capped at 10s. Returns
/// the final status and body — callers keep their own success checks and
/// response parsing. Non-retryable statuses (including other 4xx) return
/// immediately.
pub async fn send_with_retry(
    request: reqwest::RequestBuilder,
    what: &str,
) -> Result<(StatusCode, String)> {
    for attempt in 1..MAX_ATTEMPTS {
        // Streaming bodies can't be cloned; fall through to a single attempt.
        let Some(req) = request.try_clone() else {
            break;
        };
        match req.send().await {
            Ok(response) => {
                let status = response.status();
                if !is_retryable_status(status) {
                    return read_response(response, what).await;
                }
                let delay = retry_after(&response).unwrap_or_else(|| backoff_delay(attempt));
                drop(response);
                tokio::time::sleep(delay).await;
            }
            Err(e) if e.is_connect() || e.is_timeout() => {
                tokio::time::sleep(backoff_delay(attempt)).await;
            }
            Err(e) => {
                return Err(anyhow::Error::new(e).context(format!("Failed to send {what} request")));
            }
        }
    }

    // Final attempt: whatever comes back is the answer.
    let response = request
        .send()
        .await
        .with_context(|| format!("Failed to send {what} request"))?;
    read_response(response, what).await
}

async fn read_response(response: reqwest::Response, what: &str) -> Result<(StatusCode, String)> {
    let status = response.status();
    let body = response
        .text()
        .await
        .with_context(|| format!("Failed to read {what} response body"))?;
    Ok((status, body))
}

fn is_retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

fn backoff_delay(attempt: u32) -> Duration {
    Duration::from_millis(300 * 2u64.pow(attempt - 1))
}

fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()
        .map(|secs| Duration::from_secs(secs.min(10)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Serve a fixed sequence of HTTP statuses; repeat the last one after
    /// the sequence is exhausted. Returns (base_url, request_counter).
    async fn spawn_status_server(statuses: Vec<u16>) -> (String, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let server_counter = counter.clone();

        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let n = server_counter.fetch_add(1, Ordering::SeqCst);
                let status = *statuses.get(n).unwrap_or(statuses.last().unwrap());
                let mut buf = [0u8; 4096];
                let _ = socket.read(&mut buf).await;
                let body = format!("response-{n}");
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });

        (format!("http://{addr}"), counter)
    }

    #[tokio::test]
    async fn test_retries_transient_5xx_until_success() {
        let (base, requests) = spawn_status_server(vec![503, 503, 200]).await;
        let client = build_client(false).unwrap();

        let (status, body) =
            send_with_retry(client.post(&base).json(&serde_json::json!({})), "test")
                .await
                .unwrap();

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "response-2");
        assert_eq!(
            requests.load(Ordering::SeqCst),
            3,
            "two retries then success"
        );
    }

    #[tokio::test]
    async fn test_does_not_retry_client_errors() {
        let (base, requests) = spawn_status_server(vec![400]).await;
        let client = build_client(false).unwrap();

        let (status, body) = send_with_retry(client.post(&base), "test").await.unwrap();

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body, "response-0");
        assert_eq!(
            requests.load(Ordering::SeqCst),
            1,
            "4xx must not be retried"
        );
    }

    #[tokio::test]
    async fn test_gives_up_after_max_attempts() {
        let (base, requests) = spawn_status_server(vec![503]).await;
        let client = build_client(false).unwrap();

        let (status, _) = send_with_retry(client.post(&base), "test").await.unwrap();

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            requests.load(Ordering::SeqCst),
            MAX_ATTEMPTS as usize,
            "exhausts attempts then returns the failing status"
        );
    }
}
