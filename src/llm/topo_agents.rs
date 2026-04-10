//! LLM agents for topology-aware chunking.
//!
//! Two agents work in sequence:
//! - **Inspector**: classifies SIR sections as atomic / splittable / merge_candidate
//! - **Refiner**: produces an optimal partition respecting the Inspector's classifications

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::CompletionClient;

// ── Inspector Agent ────────────────────────────────────────────────

const INSPECTOR_SYSTEM: &str = "\
You are a document structure analyst. Given a Structured Intermediate Representation (SIR) \
of a document, classify each section node by how it should be chunked for retrieval.

For each section, decide:
- \"atomic\": the section is small enough (roughly under the soft budget) to be kept as one chunk.
- \"splittable\": the section is too large and must be split into sub-chunks.
- \"merge_candidate\": the section is very small and should be merged with adjacent sections.

Also identify any cross-section dependencies where splitting would break important context \
(e.g., a definition in one section referenced in another).

Only classify nodes whose node_type is \"section\". Ignore content_block nodes.";

fn inspector_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section_id": {
                            "type": "integer",
                            "description": "SIR node ID of the section"
                        },
                        "class": {
                            "type": "string",
                            "enum": ["atomic", "splittable", "merge_candidate"],
                            "description": "How this section should be treated"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reasoning for the classification"
                        }
                    },
                    "required": ["section_id", "class", "reason"],
                    "additionalProperties": false
                }
            },
            "dependencies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {
                            "type": "integer",
                            "description": "Source section ID"
                        },
                        "to": {
                            "type": "integer",
                            "description": "Target section ID"
                        },
                        "dependency_type": {
                            "type": "string",
                            "description": "Type of dependency (e.g., definition_reference, continuation)"
                        }
                    },
                    "required": ["from", "to", "dependency_type"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["classifications", "dependencies"],
        "additionalProperties": false
    })
}

/// A single section classification from the Inspector.
#[derive(Debug, Serialize, Deserialize)]
pub struct InspectorClassification {
    pub section_id: usize,
    pub class: String,
    pub reason: String,
}

/// A cross-section dependency from the Inspector.
#[derive(Debug, Deserialize)]
pub struct InspectorDependency {
    pub from: usize,
    pub to: usize,
    pub dependency_type: String,
}

/// Complete Inspector response.
#[derive(Debug, Deserialize)]
pub struct InspectorResponse {
    pub classifications: Vec<InspectorClassification>,
    pub dependencies: Vec<InspectorDependency>,
}

/// Call the Inspector agent to classify SIR sections.
pub async fn inspect_sir(
    client: &CompletionClient,
    sir_json: &str,
) -> Result<InspectorResponse> {
    let user_prompt = format!(
        "Analyze this document SIR and classify each section:\n\n{}",
        sir_json
    );

    let response = client
        .complete_json(INSPECTOR_SYSTEM, &user_prompt, inspector_schema())
        .await?;

    let parsed: InspectorResponse = serde_json::from_str(&response).map_err(|e| {
        anyhow::anyhow!("Failed to parse Inspector response: {e}\nRaw: {response}")
    })?;

    Ok(parsed)
}

// ── Refiner Agent ──────────────────────────────────────────────────

const REFINER_SYSTEM: &str = "\
You are a document chunking optimizer. Given section classifications from an Inspector agent, \
a document SIR, and the full text of splittable sections, produce an optimal partition of \
the document into chunks.

Rules:
- Atomic sections become exactly one chunk (use their full block range).
- Merge candidates should be combined with adjacent atomic or merge_candidate sections.
- Splittable sections should be divided at natural breakpoints (paragraph boundaries, \
  sub-heading boundaries) respecting a soft token budget.
- Each chunk must reference the section IDs it covers and the block range (start, end) \
  indices from the SIR.
- Preserve cross-section dependencies: if two sections are linked, prefer keeping them \
  in the same or adjacent chunks.";

fn refiner_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "partition": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "integer",
                            "description": "Sequential chunk index"
                        },
                        "section_ids": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "description": "SIR section IDs included in this chunk"
                        },
                        "block_range": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "description": "Two-element array [start, end) of block indices"
                        }
                    },
                    "required": ["chunk_id", "section_ids", "block_range"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["partition"],
        "additionalProperties": false
    })
}

/// A single chunk in the Refiner's partition.
#[derive(Debug, Deserialize)]
pub struct RefinerChunk {
    pub chunk_id: usize,
    pub section_ids: Vec<usize>,
    pub block_range: Vec<usize>,
}

/// Complete Refiner response.
#[derive(Debug, Deserialize)]
pub struct RefinerResponse {
    pub partition: Vec<RefinerChunk>,
}

/// Call the Refiner agent to produce a chunk partition.
pub async fn refine_partition(
    client: &CompletionClient,
    inspector_json: &str,
    sir_json: &str,
    section_texts: &str,
) -> Result<RefinerResponse> {
    let user_prompt = format!(
        "Inspector classifications:\n{inspector_json}\n\n\
         Document SIR:\n{sir_json}\n\n\
         Splittable section texts:\n{section_texts}"
    );

    let response = client
        .complete_json(REFINER_SYSTEM, &user_prompt, refiner_schema())
        .await?;

    let parsed: RefinerResponse = serde_json::from_str(&response).map_err(|e| {
        anyhow::anyhow!("Failed to parse Refiner response: {e}\nRaw: {response}")
    })?;

    Ok(parsed)
}
