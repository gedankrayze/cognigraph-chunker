//! Error type → HTTP response mapping.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// API error type that maps to HTTP responses.
pub struct ApiError(pub anyhow::Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({ "error": self.0.to_string() });
        (StatusCode::BAD_REQUEST, axum::Json(body)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError(err)
    }
}
