//! Data types for enriched chunking.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// An entity with its type label (e.g. "OpenAI" / "Organization").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedEntity {
    pub name: String,
    pub entity_type: String,
}

/// Record of a merge operation during semantic-key recombination.
#[derive(Debug, Clone, Serialize)]
pub struct MergeRecord {
    pub result_chunk: usize,
    pub source_chunks: Vec<usize>,
    pub shared_key: String,
}

/// A single enriched chunk with full LLM-generated metadata.
#[derive(Debug, Clone, Serialize)]
pub struct EnrichedChunk {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub token_estimate: usize,
    pub title: String,
    pub summary: String,
    pub keywords: Vec<String>,
    pub typed_entities: Vec<TypedEntity>,
    pub hypothetical_questions: Vec<String>,
    pub semantic_keys: Vec<String>,
    pub category: String,
    pub heading_path: Vec<String>,
}

/// Result of the enriched chunking pipeline.
#[derive(Debug)]
pub struct EnrichedResult {
    pub chunks: Vec<EnrichedChunk>,
    pub key_dictionary: HashMap<String, Vec<usize>>,
    pub merge_history: Vec<MergeRecord>,
    pub block_count: usize,
}
