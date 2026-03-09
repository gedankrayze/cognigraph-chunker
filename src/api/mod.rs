//! REST API module — Axum router setup and shared state.

pub mod chunk;
pub mod cognitive;
pub mod errors;
pub mod health;
pub mod merge;
pub mod semantic;
pub mod split;
pub mod types;

use std::sync::Arc;
use std::time::Duration;

use axum::{
    Router,
    extract::State,
    http::{HeaderMap, Method, StatusCode, header},
    middleware::{self, Next},
    response::Response,
};
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;

/// Maximum API request body size (10 MiB).
///
/// Intentionally lower than the CLI's 50 MiB `--max-input-size` default:
/// API requests are network-bound and often multiplexed, so a tighter limit
/// protects against abuse. CLI users processing local files can use the
/// `--max-input-size` flag to raise their limit.
const API_BODY_LIMIT: usize = 10 * 1024 * 1024;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    /// Optional API key for bearer auth.
    pub api_key: Option<String>,
    /// Allow embedding provider base_urls pointing to private/loopback IPs.
    pub allow_private_urls: bool,
    /// Allowed CORS origins (empty = deny all browser cross-origin requests).
    pub cors_origins: Vec<String>,
}

/// Build the Axum router with all API routes.
pub fn router(state: AppState) -> Router {
    let cors = if state.cors_origins.is_empty() {
        // No origins configured: deny cross-origin (only same-origin or non-browser clients)
        CorsLayer::new()
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
    } else {
        // Origins are validated at startup in serve_cmd; unwrap is safe here.
        let origins: Vec<header::HeaderValue> = state
            .cors_origins
            .iter()
            .map(|o| o.parse().expect("CORS origin was validated at startup"))
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
    };

    let shared_state = Arc::new(state);

    Router::new()
        .route("/api/v1/health", axum::routing::get(health::health))
        .route("/api/v1/chunk", axum::routing::post(chunk::chunk_handler))
        .route("/api/v1/split", axum::routing::post(split::split_handler))
        .route(
            "/api/v1/semantic",
            axum::routing::post(semantic::semantic_handler),
        )
        .route(
            "/api/v1/cognitive",
            axum::routing::post(cognitive::cognitive_handler),
        )
        .route("/api/v1/merge", axum::routing::post(merge::merge_handler))
        .layer(middleware::from_fn_with_state(
            shared_state.clone(),
            auth_middleware,
        ))
        .layer(RequestBodyLimitLayer::new(API_BODY_LIMIT))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            Duration::from_secs(120),
        ))
        .layer(cors)
        .with_state(shared_state)
}

/// Bearer token auth middleware. If no API key is configured, all requests pass.
async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let Some(ref expected_key) = state.api_key else {
        return Ok(next.run(request).await);
    };

    // Health endpoint is always open
    if request.uri().path() == "/api/v1/health" {
        return Ok(next.run(request).await);
    }

    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match auth_header {
        Some(token) if token == expected_key => Ok(next.run(request).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
