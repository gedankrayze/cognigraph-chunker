//! Shared provider construction for API handlers.
use super::AppState;
use crate::embeddings::onnx::OnnxProvider;
use crate::embeddings::{AnyProvider, EmbedProviderOpts, ProviderType};

/// Build the embedding provider for an API request. ONNX model paths are
/// validated against the server's --onnx-model-dir allowlist first, and
/// loaded models are cached in AppState (re-parsing the model per request
/// dominated ONNX-backed request latency).
pub async fn build_api_provider(
    opts: &EmbedProviderOpts,
    state: &AppState,
) -> anyhow::Result<AnyProvider> {
    if matches!(opts.provider, ProviderType::Onnx) {
        let requested = opts
            .model_path
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("model_path is required for onnx provider"))?;
        let validated = super::validation::validate_model_path(requested, &state.onnx_model_dir)?;

        if let Some(cached) = state.onnx_providers.lock().unwrap().get(&validated) {
            return Ok(AnyProvider::Onnx(Box::new(cached.clone())));
        }

        // Load outside the lock (model parsing is slow); racing requests may
        // both load, the second insert wins harmlessly.
        let path_str = validated.to_string_lossy().into_owned();
        let provider = tokio::task::spawn_blocking(move || OnnxProvider::new(&path_str))
            .await
            .map_err(|e| anyhow::anyhow!("ONNX model load task panicked: {e}"))??;
        state
            .onnx_providers
            .lock()
            .unwrap()
            .insert(validated, provider.clone());
        return Ok(AnyProvider::Onnx(Box::new(provider)));
    }

    opts.build_provider().await
}

#[cfg(test)]
mod tests {
    use crate::api::semantic::SemanticRequest;
    use crate::embeddings::ProviderType;

    /// Locks the serde wire format: `#[serde(flatten)]` must inline the
    /// provider fields under the same snake_case names as before, and
    /// ProviderType must deserialize from lowercase strings.
    #[test]
    fn test_semantic_request_flatten_wire_compat() {
        let req: SemanticRequest = serde_json::from_str(
            r#"{"text":"x","provider":"openai","api_key":"k","base_url":"https://example.com"}"#,
        )
        .expect("flattened provider fields must deserialize");

        assert_eq!(req.text, "x");
        assert!(matches!(req.provider_opts.provider, ProviderType::Openai));
        assert_eq!(req.provider_opts.api_key.as_deref(), Some("k"));
        assert_eq!(
            req.provider_opts.base_url.as_deref(),
            Some("https://example.com")
        );
        assert!(!req.provider_opts.danger_accept_invalid_certs);
        assert!(req.provider_opts.model.is_none());
    }

    /// Omitting every provider field must fall back to the Ollama default.
    #[test]
    fn test_semantic_request_provider_defaults() {
        let req: SemanticRequest =
            serde_json::from_str(r#"{"text":"x"}"#).expect("defaults must apply");
        assert!(matches!(req.provider_opts.provider, ProviderType::Ollama));
    }
}
