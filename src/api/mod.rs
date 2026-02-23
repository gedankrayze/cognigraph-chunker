//! REST API module — Axum router setup and shared state.

pub mod chunk;
pub mod errors;
pub mod health;
pub mod merge;
pub mod semantic;
pub mod split;
pub mod types;

use std::sync::Arc;

use axum::{
    Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::Response,
};
use tower_http::cors::CorsLayer;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    /// Optional API key for bearer auth.
    pub api_key: Option<String>,
}

/// Build the Axum router with all API routes.
pub fn router(state: AppState) -> Router {
    let api = Router::new()
        .route("/api/v1/health", axum::routing::get(health::health))
        .route("/api/v1/chunk", axum::routing::post(chunk::chunk_handler))
        .route("/api/v1/split", axum::routing::post(split::split_handler))
        .route(
            "/api/v1/semantic",
            axum::routing::post(semantic::semantic_handler),
        )
        .route("/api/v1/merge", axum::routing::post(merge::merge_handler));

    let shared_state = Arc::new(state);

    let api = api.layer(middleware::from_fn_with_state(
        shared_state.clone(),
        auth_middleware,
    ));

    api.layer(CorsLayer::permissive())
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
