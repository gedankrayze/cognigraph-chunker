//! LLM-based intent generation for intent-driven chunking.
//!
//! Uses structured JSON output from an LLM to predict user queries
//! that the document's content could answer.

use anyhow::{Context, Result};
use serde::Deserialize;

use super::CompletionClient;
use crate::semantic::intent_types::{IntentType, PredictedIntent};

#[derive(Deserialize)]
struct IntentsResponse {
    intents: Vec<RawIntent>,
}

#[derive(Deserialize)]
struct RawIntent {
    query: String,
    intent_type: IntentType,
}

const SYSTEM_PROMPT: &str = "\
You are an intent prediction system. Given a document, predict the most likely \
user queries that this document could answer. Each query should be a realistic \
question a user might type into a search engine or Q&A system.

Rules:
- Generate diverse queries covering different aspects of the document
- Each query should target a specific piece of information or concept
- Classify each query by intent type:
  - factual: seeks a specific fact, number, name, or date
  - procedural: asks how to do something, step-by-step instructions
  - conceptual: asks for explanation, definition, or understanding
  - comparative: asks to compare, contrast, or evaluate alternatives
- Prefer specific queries over vague ones
- Queries should be self-contained (not reference \"the document\" or \"the text\")";

fn json_schema(max_intents: usize) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "intents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A realistic user query this document could answer"
                        },
                        "intent_type": {
                            "type": "string",
                            "enum": ["factual", "procedural", "conceptual", "comparative"],
                            "description": "The type of user intent"
                        }
                    },
                    "required": ["query", "intent_type"],
                    "additionalProperties": false
                },
                "maxItems": max_intents
            }
        },
        "required": ["intents"],
        "additionalProperties": false
    })
}

/// Generate predicted user intents for a document using the LLM.
pub async fn generate_intents(
    client: &CompletionClient,
    text: &str,
    max_intents: usize,
) -> Result<Vec<PredictedIntent>> {
    if text.trim().len() < 20 {
        return Ok(vec![]);
    }

    let user_prompt = format!(
        "Generate up to {max_intents} predicted user queries for this document:\n\n{text}"
    );

    let response = client
        .complete_json(SYSTEM_PROMPT, &user_prompt, json_schema(max_intents))
        .await
        .context("Intent generation failed")?;

    let parsed: IntentsResponse = serde_json::from_str(&response).map_err(|e| {
        anyhow::anyhow!("Failed to parse LLM intent response: {e}\nRaw: {response}")
    })?;

    Ok(parsed
        .intents
        .into_iter()
        .map(|r| PredictedIntent {
            query: r.query,
            intent_type: r.intent_type,
            matched_chunks: vec![],
        })
        .collect())
}
