//! Shared OAuth2 client-credentials token source.
//!
//! Used by both the OAuth embedding provider and the OAuth reranker
//! (previously ~110 duplicated lines each). The cache lock is held across
//! the fetch, so concurrent callers with an expired token trigger exactly
//! one request to the token endpoint (single-flight).

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::http_util::send_with_retry;

/// Cached, auto-refreshing OAuth2 client-credentials token source.
pub struct OAuthTokenSource {
    client: reqwest::Client,
    token_url: String,
    client_id: String,
    client_secret: String,
    scope: Option<String>,
    cache: tokio::sync::Mutex<Option<CachedToken>>,
}

struct CachedToken {
    access_token: String,
    expires_at: Instant,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    /// RFC 6749 makes `expires_in` RECOMMENDED, not required — default to
    /// a conservative 5 minutes when the IdP omits it.
    #[serde(default = "default_expires_in")]
    expires_in: u64,
}

fn default_expires_in() -> u64 {
    300
}

impl OAuthTokenSource {
    pub fn new(
        client: reqwest::Client,
        token_url: String,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
    ) -> Self {
        Self {
            client,
            token_url,
            client_id,
            client_secret,
            scope,
            cache: tokio::sync::Mutex::new(None),
        }
    }

    /// Acquire a valid access token, using the cache if not near expiry.
    ///
    /// The cache lock is held across the network fetch: concurrent callers
    /// wait for the in-flight request instead of stampeding the endpoint.
    pub async fn get_token(&self) -> Result<String> {
        let mut cache = self.cache.lock().await;

        // 60s safety margin before expiry
        if let Some(ref cached) = *cache
            && Instant::now() + Duration::from_secs(60) < cached.expires_at
        {
            return Ok(cached.access_token.clone());
        }

        let mut form = vec![
            ("grant_type", "client_credentials"),
            ("client_id", &self.client_id),
            ("client_secret", &self.client_secret),
        ];
        if let Some(ref s) = self.scope {
            form.push(("scope", s));
        }

        let request = self.client.post(&self.token_url).form(&form);
        let (status, body) = send_with_retry(request, "OAuth token").await?;

        if !status.is_success() {
            bail!("OAuth token request failed ({}): {}", status, body);
        }

        let token_resp: TokenResponse =
            serde_json::from_str(&body).context("Failed to parse OAuth token response")?;

        let expires_at = Instant::now() + Duration::from_secs(token_resp.expires_in);
        let access_token = token_resp.access_token.clone();
        *cache = Some(CachedToken {
            access_token: token_resp.access_token,
            expires_at,
        });

        Ok(access_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn spawn_token_server() -> (String, Arc<AtomicUsize>) {
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
                // Note: no expires_in — RFC 6749 allows omitting it
                let body = r#"{"access_token":"tok-123"}"#;
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
    async fn test_concurrent_callers_single_flight() {
        let (token_url, requests) = spawn_token_server().await;
        let source = Arc::new(OAuthTokenSource::new(
            crate::http_util::build_client(false).unwrap(),
            token_url,
            "client".into(),
            "secret".into(),
            None,
        ));

        // 8 concurrent callers with a cold cache → exactly one token request
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let s = source.clone();
                tokio::spawn(async move { s.get_token().await })
            })
            .collect();
        for h in handles {
            assert_eq!(h.await.unwrap().unwrap(), "tok-123");
        }

        assert_eq!(
            requests.load(Ordering::SeqCst),
            1,
            "concurrent callers must share a single token fetch"
        );

        // Cache hit afterwards — still one request
        assert_eq!(source.get_token().await.unwrap(), "tok-123");
        assert_eq!(requests.load(Ordering::SeqCst), 1);
    }
}
