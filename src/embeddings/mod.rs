//! Embedding providers for semantic chunking.

pub mod cloudflare;
pub mod oauth;
pub mod ollama;
pub mod onnx;
pub mod openai;

use anyhow::Result;

/// Trait for embedding providers.
///
/// Each provider takes a batch of text strings and returns their embedding vectors.
#[allow(async_fn_in_trait)]
pub trait EmbeddingProvider {
    /// Embed a batch of texts, returning one vector per input text.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>>;

    /// Return the embedding dimension (if known ahead of time).
    fn dimension(&self) -> Option<usize> {
        None
    }
}

/// Supported embedding provider types.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ProviderType {
    /// OpenAI embeddings API
    Openai,
    /// Ollama local embeddings API
    Ollama,
    /// Local ONNX Runtime model
    Onnx,
    /// Cloudflare Workers AI embeddings
    Cloudflare,
    /// OAuth-authenticated OpenAI-compatible endpoint
    Oauth,
}
