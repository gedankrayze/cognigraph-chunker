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
        // Internal errors can carry server-side detail (filesystem paths,
        // upstream response bodies, env hints) — log them, don't return them.
        let body = if status == StatusCode::INTERNAL_SERVER_ERROR {
            eprintln!("[api] internal error: {:#}", self.0);
            json!({ "error": "internal server error" })
        } else {
            json!({ "error": msg })
        };
        (status, axum::Json(body)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn body_of(resp: Response) -> String {
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn test_internal_errors_are_sanitized() {
        let err = ApiError(anyhow::anyhow!(
            "failed reading /home/deploy/.env.openai: permission denied"
        ));
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_of(resp).await;
        assert!(
            !body.contains(".env.openai") && !body.contains("/home/deploy"),
            "500 body must not leak internal detail: {body}"
        );
    }

    #[tokio::test]
    async fn test_validation_errors_keep_their_message() {
        let err = ApiError(anyhow::anyhow!("model_path is required for onnx provider"));
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_of(resp).await;
        assert!(body.contains("model_path is required"), "{body}");
    }
}
