//! Integration tests for embedding providers.
//! Run with: cargo test --test embedding_providers -- --nocapture

mod common {
    // Re-import from main crate
    pub use cognigraph_chunker::embeddings::*;
}

// These tests require external services and are ignored by default.
// Run manually with: cargo test --test embedding_providers -- --ignored --nocapture

#[tokio::test]
#[ignore]
async fn test_ollama_nomic_embed() {
    use common::ollama::OllamaProvider;
    use common::EmbeddingProvider;

    let provider = OllamaProvider::new(None, Some("nomic-embed-text".to_string())).expect("Failed to build Ollama client");
    let texts = &["Hello world", "How are you?", "Rust is great"];

    let embeddings = provider.embed(texts).await.expect("Ollama embed failed");

    assert_eq!(embeddings.len(), 3);
    for (i, emb) in embeddings.iter().enumerate() {
        println!("  text[{}]: dim={}", i, emb.len());
        assert!(!emb.is_empty(), "Embedding should not be empty");
    }
    println!("nomic-embed-text: OK (dim={})", embeddings[0].len());
}

#[tokio::test]
#[ignore]
async fn test_ollama_qwen3_embedding() {
    use common::ollama::OllamaProvider;
    use common::EmbeddingProvider;

    let provider = OllamaProvider::new(None, Some("qwen3-embedding".to_string())).expect("Failed to build Ollama client");
    let texts = &["Hello world", "How are you?"];

    let embeddings = provider.embed(texts).await.expect("Ollama embed failed");

    assert_eq!(embeddings.len(), 2);
    for (i, emb) in embeddings.iter().enumerate() {
        println!("  text[{}]: dim={}", i, emb.len());
        assert!(!emb.is_empty(), "Embedding should not be empty");
    }
    println!("qwen3-embedding: OK (dim={})", embeddings[0].len());
}

#[tokio::test]
#[ignore]
async fn test_openai_embedding() {
    use common::openai::OpenAiProvider;
    use common::EmbeddingProvider;

    let api_key = std::fs::read_to_string(".env.openai")
        .expect("Missing .env.openai file")
        .lines()
        .find(|l| l.starts_with("OPENAI_API_KEY="))
        .expect("OPENAI_API_KEY not found in .env.openai")
        .strip_prefix("OPENAI_API_KEY=")
        .unwrap()
        .to_string();

    let provider = OpenAiProvider::new(api_key, None, None).expect("Failed to build OpenAI client");
    let texts = &["Hello world", "How are you?"];

    let embeddings = provider.embed(texts).await.expect("OpenAI embed failed");

    assert_eq!(embeddings.len(), 2);
    for (i, emb) in embeddings.iter().enumerate() {
        println!("  text[{}]: dim={}", i, emb.len());
        assert!(!emb.is_empty(), "Embedding should not be empty");
    }
    println!("OpenAI text-embedding-3-small: OK (dim={})", embeddings[0].len());
}

#[tokio::test]
#[ignore]
async fn test_onnx_minilm() {
    use common::onnx::OnnxProvider;
    use common::EmbeddingProvider;

    let provider = OnnxProvider::new("models/all-MiniLM-L6-v2")
        .expect("Failed to load ONNX model from models/all-MiniLM-L6-v2");

    let texts = &["Hello world", "How are you?", "Rust is great"];

    let embeddings = provider.embed(texts).await.expect("ONNX embed failed");

    assert_eq!(embeddings.len(), 3);
    for (i, emb) in embeddings.iter().enumerate() {
        println!("  text[{}]: dim={}", i, emb.len());
        assert!(!emb.is_empty(), "Embedding should not be empty");
        assert_eq!(emb.len(), 384, "all-MiniLM-L6-v2 should produce 384-dim embeddings");
    }
    println!("ONNX all-MiniLM-L6-v2: OK (dim={})", embeddings[0].len());
}
