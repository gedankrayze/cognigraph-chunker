//! LLM-based chunk enrichment.
//!
//! Uses structured JSON output from an LLM to extract 7 metadata fields
//! per chunk: title, summary, keywords, typed_entities, hypothetical_questions,
//! semantic_keys, and category.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::CompletionClient;

/// Response from the enrichment LLM call.
#[derive(Debug, Deserialize)]
pub struct EnrichmentResponse {
    pub title: String,
    pub summary: String,
    pub keywords: Vec<String>,
    pub typed_entities: Vec<TypedEntityLlm>,
    pub hypothetical_questions: Vec<String>,
    pub semantic_keys: Vec<String>,
    pub category: String,
}

/// Typed entity as returned by the LLM.
#[derive(Debug, Deserialize)]
pub struct TypedEntityLlm {
    pub name: String,
    pub entity_type: String,
}

/// Response from the re-enrichment LLM call (title + summary only).
#[derive(Debug, Deserialize)]
struct ReEnrichResponse {
    title: String,
    summary: String,
}

const SYSTEM_PROMPT: &str = "\
You are a document enrichment engine. Given a text chunk from a document, \
extract exactly 7 metadata fields:

1. title: A concise descriptive title for this chunk (5-10 words).
2. summary: A 1-2 sentence summary of the chunk's key content.
3. keywords: 3-8 topical keywords (lowercase, no duplicates).
4. typed_entities: Named entities with their type (Person, Organization, \
   Technology, Location, Concept, Event, Product, Metric, Date).
5. hypothetical_questions: 2-4 questions a user might ask that this chunk answers.
6. semantic_keys: 2-5 lowercase-hyphenated topic keys (e.g. \"neural-network-training\", \
   \"api-authentication\"). Reuse existing keys from the dictionary when applicable.
7. category: Exactly one of: background, methodology, results, discussion, \
   configuration, reference, definition, procedure, example, other.

Rules:
- semantic_keys should be specific enough to distinguish topics but general enough \
  to group related chunks. Prefer reusing keys from the existing dictionary.
- All keywords and semantic_keys must be lowercase.
- typed_entities should only include explicitly mentioned entities.";

const RE_ENRICH_SYSTEM_PROMPT: &str = "\
You are a document enrichment engine. Given a merged text chunk, produce \
an updated title and summary that accurately reflect the combined content. \
Keep the title concise (5-10 words) and the summary to 1-2 sentences.";

fn enrichment_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Concise descriptive title (5-10 words)"
            },
            "summary": {
                "type": "string",
                "description": "1-2 sentence summary of key content"
            },
            "keywords": {
                "type": "array",
                "items": { "type": "string" },
                "description": "3-8 topical keywords (lowercase)"
            },
            "typed_entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "entity_type": { "type": "string" }
                    },
                    "required": ["name", "entity_type"],
                    "additionalProperties": false
                },
                "description": "Named entities with type labels"
            },
            "hypothetical_questions": {
                "type": "array",
                "items": { "type": "string" },
                "description": "2-4 questions this chunk answers"
            },
            "semantic_keys": {
                "type": "array",
                "items": { "type": "string" },
                "description": "2-5 lowercase-hyphenated topic keys"
            },
            "category": {
                "type": "string",
                "description": "One of: background, methodology, results, discussion, configuration, reference, definition, procedure, example, other"
            }
        },
        "required": [
            "title", "summary", "keywords", "typed_entities",
            "hypothetical_questions", "semantic_keys", "category"
        ],
        "additionalProperties": false
    })
}

fn re_enrich_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Updated concise title (5-10 words)"
            },
            "summary": {
                "type": "string",
                "description": "Updated 1-2 sentence summary"
            }
        },
        "required": ["title", "summary"],
        "additionalProperties": false
    })
}

/// Enrich a single chunk with 7 metadata fields via LLM.
///
/// The existing key dictionary is included in the prompt so the LLM can
/// reuse semantic keys for topical coherence across chunks.
pub async fn enrich_chunk(
    client: &CompletionClient,
    text: &str,
    existing_keys: &HashMap<String, Vec<usize>>,
) -> Result<EnrichmentResponse> {
    let mut user_prompt = String::with_capacity(text.len() + 256);

    if !existing_keys.is_empty() {
        user_prompt.push_str("Existing semantic key dictionary (reuse when applicable):\n");
        for key in existing_keys.keys() {
            user_prompt.push_str("- ");
            user_prompt.push_str(key);
            user_prompt.push('\n');
        }
        user_prompt.push('\n');
    }

    user_prompt.push_str("Text chunk:\n");
    user_prompt.push_str(text);

    let response = client
        .complete_json(SYSTEM_PROMPT, &user_prompt, enrichment_schema())
        .await
        .context("Enrichment LLM call failed")?;

    let parsed: EnrichmentResponse = serde_json::from_str(&response).map_err(|e| {
        anyhow::anyhow!("Failed to parse enrichment response: {e}\nRaw: {response}")
    })?;

    Ok(parsed)
}

/// Lightweight re-enrichment after merging: produce updated title + summary.
pub async fn re_enrich_merged(
    client: &CompletionClient,
    text: &str,
) -> Result<(String, String)> {
    let response = client
        .complete_json(RE_ENRICH_SYSTEM_PROMPT, text, re_enrich_schema())
        .await
        .context("Re-enrichment LLM call failed")?;

    let parsed: ReEnrichResponse = serde_json::from_str(&response).map_err(|e| {
        anyhow::anyhow!("Failed to parse re-enrichment response: {e}\nRaw: {response}")
    })?;

    Ok((parsed.title, parsed.summary))
}
