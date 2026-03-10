//! Local ONNX Runtime embeddings provider.
//!
//! Runs sentence-transformer models locally using ONNX Runtime.
//! Compatible with models exported from HuggingFace (e.g., all-MiniLM-L6-v2).

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;

use super::{EmbeddingProvider, ensure_onnx_runtime_available};

/// ONNX Runtime embeddings provider for local inference.
///
/// Expects a directory containing:
/// - `model.onnx` (the ONNX model)
/// - `tokenizer.json` (HuggingFace tokenizer config)
pub struct OnnxProvider {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl OnnxProvider {
    /// Create a new ONNX provider from a model directory.
    ///
    /// The directory should contain `model.onnx` and `tokenizer.json`.
    pub fn new(model_path: &str) -> Result<Self> {
        ensure_onnx_runtime_available().context("ONNX Runtime library is not available")?;

        let model_dir = PathBuf::from(model_path);

        let onnx_path = model_dir.join("model.onnx");
        if !onnx_path.exists() {
            bail!(
                "ONNX model not found at {}. Expected model.onnx in the model directory.",
                onnx_path.display()
            );
        }

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            bail!(
                "Tokenizer not found at {}. Expected tokenizer.json in the model directory.",
                tokenizer_path.display()
            );
        }

        let session = Session::builder()
            .context("Failed to create ONNX Runtime session builder")?
            .commit_from_file(&onnx_path)
            .context("Failed to load ONNX model")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }
}

/// Extract embeddings from ONNX output while session lock is held.
fn extract_embeddings(
    output: &ort::value::DynValue,
    encodings: &[tokenizers::Encoding],
    batch_size: usize,
) -> Result<Vec<Vec<f64>>> {
    let (shape, data) = output
        .try_extract_tensor::<f32>()
        .context("Failed to extract output tensor")?;
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

    if dims.len() == 3 {
        // [batch_size, seq_len, hidden_dim] — mean pool using attention mask
        let seq_len = dims[1];
        let hidden_dim = dims[2];
        let expected = batch_size * seq_len * hidden_dim;
        if data.len() < expected {
            bail!(
                "ONNX output tensor too small: expected at least {} elements for shape {:?}, got {}",
                expected,
                dims,
                data.len()
            );
        }
        let mut result = Vec::with_capacity(batch_size);

        for (i, encoding) in encodings.iter().enumerate().take(batch_size) {
            let mut embedding = vec![0.0f64; hidden_dim];
            let mask = encoding.get_attention_mask();
            let mask_sum: f64 = mask.iter().map(|&m| m as f64).sum();

            if mask_sum > 0.0 {
                for (j, &m) in mask.iter().enumerate() {
                    if m == 1 {
                        let base = i * seq_len * hidden_dim + j * hidden_dim;
                        for k in 0..hidden_dim {
                            embedding[k] += data[base + k] as f64;
                        }
                    }
                }
                for val in &mut embedding {
                    *val /= mask_sum;
                }
            }

            result.push(embedding);
        }
        Ok(result)
    } else if dims.len() == 2 {
        // [batch_size, hidden_dim] — already pooled
        let hidden_dim = dims[1];
        let expected = batch_size * hidden_dim;
        if data.len() < expected {
            bail!(
                "ONNX output tensor too small: expected at least {} elements for shape {:?}, got {}",
                expected,
                dims,
                data.len()
            );
        }
        let mut result = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let base = i * hidden_dim;
            let embedding: Vec<f64> = (0..hidden_dim).map(|k| data[base + k] as f64).collect();
            result.push(embedding);
        }
        Ok(result)
    } else {
        bail!("Unexpected output tensor shape: {:?}", dims);
    }
}

impl EmbeddingProvider for OnnxProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Build padded input_ids and attention_mask
        let mut input_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            input_ids.extend(ids.iter().map(|&id| id as i64));
            input_ids.extend(std::iter::repeat_n(0i64, max_len - ids.len()));

            attention_mask.extend(mask.iter().map(|&m| m as i64));
            attention_mask.extend(std::iter::repeat_n(0i64, max_len - mask.len()));
        }

        // token_type_ids: all zeros for single-segment inputs (required by BERT-based models)
        let token_type_ids: Vec<i64> = vec![0i64; batch_size * max_len];

        let input_ids_value = Value::from_array(([batch_size, max_len], input_ids))
            .context("Failed to create input_ids ORT value")?;
        let attention_mask_value = Value::from_array(([batch_size, max_len], attention_mask))
            .context("Failed to create attention_mask ORT value")?;
        let token_type_ids_value = Value::from_array(([batch_size, max_len], token_type_ids))
            .context("Failed to create token_type_ids ORT value")?;

        // Run inference and extract embeddings while session lock is held
        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("Session lock poisoned: {}", e))?;

        let outputs = session
            .run(ort::inputs![
                input_ids_value,
                attention_mask_value,
                token_type_ids_value
            ])
            .context("ONNX inference failed")?;

        let embeddings = extract_embeddings(&outputs[0], &encodings, batch_size)?;

        Ok(embeddings)
    }
}
