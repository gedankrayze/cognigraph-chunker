//! LLM-based relation triple extraction.
//!
//! Uses structured JSON output from an LLM to extract high-quality
//! Subject-Predicate-Object triples from chunk text.

use anyhow::Result;
use serde::Deserialize;

use super::CompletionClient;

/// A relation triple extracted by the LLM.
#[derive(Debug, Clone, Deserialize)]
pub struct RelationTriple {
    /// Subject entity or noun phrase.
    pub subject: String,
    /// Relation/verb phrase.
    pub predicate: String,
    /// Object entity or noun phrase.
    pub object: String,
}

#[derive(Deserialize)]
struct ExtractionResponse {
    relations: Vec<RelationTriple>,
}

const SYSTEM_PROMPT: &str = "\
You are a precise information extraction system. Extract factual relation triples \
(subject-predicate-object) from the given text. Each triple should represent a clear, \
self-contained factual statement.

Rules:
- Subject and object should be specific noun phrases, not pronouns
- Predicate should be a concise verb phrase (e.g. \"uses\", \"is\", \"depends on\", \"was administered to\")
- Only extract relations that are explicitly stated, not implied
- Normalize predicates to active voice present tense where possible
- Skip trivial relations (e.g. \"Section 3 contains text\")
- Maximum 10 triples per text";

fn json_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Subject entity or noun phrase"
                        },
                        "predicate": {
                            "type": "string",
                            "description": "Verb phrase describing the relation"
                        },
                        "object": {
                            "type": "string",
                            "description": "Object entity or noun phrase"
                        }
                    },
                    "required": ["subject", "predicate", "object"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["relations"],
        "additionalProperties": false
    })
}

/// Extract relation triples from a chunk of text using the LLM.
pub async fn extract_relations(
    client: &CompletionClient,
    text: &str,
) -> Result<Vec<RelationTriple>> {
    if text.trim().len() < 20 {
        return Ok(vec![]);
    }

    let response = client
        .complete_json(SYSTEM_PROMPT, text, json_schema())
        .await?;

    let parsed: ExtractionResponse = serde_json::from_str(&response).map_err(|e| {
        anyhow::anyhow!("Failed to parse LLM relation response: {e}\nRaw: {response}")
    })?;

    Ok(parsed.relations)
}

/// Extract relations for multiple chunks in sequence.
///
/// Returns one `Vec<RelationTriple>` per chunk, in the same order.
pub async fn extract_relations_batch(
    client: &CompletionClient,
    chunks: &[&str],
) -> Result<Vec<Vec<RelationTriple>>> {
    let mut results = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let rels = extract_relations(client, chunk).await?;
        results.push(rels);
    }

    Ok(results)
}
