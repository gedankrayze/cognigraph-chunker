//! Reranker provider trait and implementations for cross-encoder boundary refinement.
//!
//! Cross-encoders score text pairs more accurately than embedding similarity,
//! but are expensive. Used only on ambiguous boundaries (Phase 3).

use anyhow::Result;

/// Trait for reranking providers (cross-encoders).
///
/// Given a query text and candidate documents, returns relevance scores.
/// Higher scores indicate stronger semantic connection.
#[allow(async_fn_in_trait)]
pub trait RerankerProvider: Send + Sync {
    /// Score each document against the query.
    ///
    /// Returns one score per document in the same order as `documents`.
    /// Scores are in [0.0, 1.0] — higher means more relevant/connected.
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>>;

    /// Return the model name for diagnostics.
    fn model_name(&self) -> &str;
}

/// ONNX-based cross-encoder reranker.
///
/// Expects a sequence-classification model directory containing:
/// - `model.onnx` (cross-encoder ONNX model)
/// - `tokenizer.json` (HuggingFace tokenizer config)
///
/// Compatible with models like:
/// - `cross-encoder/ms-marco-MiniLM-L-6-v2`
/// - `BAAI/bge-reranker-base`
pub struct OnnxReranker {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
}

impl OnnxReranker {
    /// Create a new ONNX reranker from a model directory.
    pub fn new(model_path: &str) -> Result<Self> {
        let model_dir = std::path::PathBuf::from(model_path);

        let onnx_path = model_dir.join("model.onnx");
        anyhow::ensure!(
            onnx_path.exists(),
            "ONNX reranker model not found at {}",
            onnx_path.display()
        );

        let tokenizer_path = model_dir.join("tokenizer.json");
        anyhow::ensure!(
            tokenizer_path.exists(),
            "Tokenizer not found at {}",
            tokenizer_path.display()
        );

        let session = ort::session::Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(&onnx_path)
            .context("Failed to load ONNX reranker model")?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let model_name = model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "onnx-reranker".to_string());

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            model_name,
        })
    }

    /// Score a single text pair. Returns a relevance score in [0.0, 1.0].
    fn score_pair(&self, text_a: &str, text_b: &str) -> Result<f64> {
        use anyhow::Context;

        // Cross-encoders encode both texts as a single input (text_a [SEP] text_b)
        let encoding = self
            .tokenizer
            .encode((text_a, text_b), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let seq_len = encoding.get_ids().len();
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

        let input_ids_val = ort::value::Value::from_array(([1, seq_len], input_ids))
            .context("Failed to create input_ids")?;
        let attention_mask_val = ort::value::Value::from_array(([1, seq_len], attention_mask))
            .context("Failed to create attention_mask")?;
        let token_type_ids_val = ort::value::Value::from_array(([1, seq_len], token_type_ids))
            .context("Failed to create token_type_ids")?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("Session lock poisoned: {e}"))?;

        let outputs = session
            .run(ort::inputs![
                input_ids_val,
                attention_mask_val,
                token_type_ids_val
            ])
            .context("ONNX reranker inference failed")?;

        // Output shape: [1, 1] or [1] — logit score
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract reranker output")?;

        let logit = if shape.iter().product::<i64>() >= 1 {
            data[0] as f64
        } else {
            anyhow::bail!("Unexpected reranker output shape: {:?}", shape);
        };

        // Sigmoid to normalize to [0, 1]
        Ok(sigmoid(logit))
    }
}

impl RerankerProvider for OnnxReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f64>> {
        let mut scores = Vec::with_capacity(documents.len());
        for doc in documents {
            scores.push(self.score_pair(query, doc)?);
        }
        Ok(scores)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

use anyhow::Context;
