//! Error type → HTTP response mapping.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// API error type that maps to HTTP responses.
pub struct ApiError(pub anyhow::Error);

/// Categorize an error message into an appropriate HTTP status code.
fn categorize_error(msg: &str) -> StatusCode {
    let lower = msg.to_lowercase();

    // Validation errors → 400
    if lower.contains("required")
        || lower.contains("invalid")
        || lower.contains("exceeds")
        || lower.contains("must be")
        || lower.contains("could not resolve")
        || lower.contains("failed to resolve")
        || lower.contains("dns error")
        || lower.contains("no such host")
    {
        return StatusCode::BAD_REQUEST;
    }

    // Upstream provider/network errors → 502
    if lower.contains("ollama error")
        || lower.contains("openai api error")
        || lower.contains("failed to connect")
    {
        return StatusCode::BAD_GATEWAY;
    }

    // Timeouts → 504
    if lower.contains("timed out") || lower.contains("timeout") {
        return StatusCode::GATEWAY_TIMEOUT;
    }

    // Everything else → 500
    StatusCode::INTERNAL_SERVER_ERROR
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let msg = self.0.to_string();
        let status = categorize_error(&msg);
        let body = json!({ "error": msg });
        (status, axum::Json(body)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError(err)
    }
}
