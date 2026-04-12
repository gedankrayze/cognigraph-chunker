//! Data types for intent-driven chunking.

use serde::{Deserialize, Serialize};

/// The type of user intent a query represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntentType {
    Factual,
    Procedural,
    Conceptual,
    Comparative,
}

/// A predicted user query with its intent type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedIntent {
    pub query: String,
    pub intent_type: IntentType,
    #[serde(default)]
    pub matched_chunks: Vec<usize>,
}

/// A chunk produced by intent-driven chunking.
#[derive(Debug, Clone, Serialize)]
pub struct IntentChunk {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub token_estimate: usize,
    /// Index into the intents vec for the best-matching intent.
    pub best_intent: usize,
    /// Cosine similarity between chunk centroid and best intent embedding.
    pub alignment_score: f64,
    pub heading_path: Vec<String>,
}

/// Result of intent-driven chunking.
#[derive(Debug)]
pub struct IntentResult {
    pub chunks: Vec<IntentChunk>,
    pub intents: Vec<PredictedIntent>,
    pub partition_score: f64,
    pub block_count: usize,
}
