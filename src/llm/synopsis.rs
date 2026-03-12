//! LLM-based chunk synopsis generation.
//!
//! Generates a concise 1-2 sentence summary for each chunk using
//! structured JSON output from an OpenAI-compatible completion API.

use anyhow::{Context, Result};
use serde::Deserialize;

use super::CompletionClient;

#[derive(Debug, Deserialize)]
struct SynopsisResponse {
    synopsis: String,
}

const SYSTEM_PROMPT: &str = "\
You are a precise summarization engine. Given a text chunk from a document, \
produce a single concise synopsis (1-2 sentences) that captures the key claim \
or information in the chunk. Be specific — use proper nouns and concrete details \
rather than vague descriptions. Do not start with \"This chunk\" or \"The text\".";

/// Generate a synopsis for a single chunk.
pub async fn generate_synopsis(client: &CompletionClient, text: &str) -> Result<String> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "synopsis": {
                "type": "string",
                "description": "A concise 1-2 sentence summary of the chunk's key content."
            }
        },
        "required": ["synopsis"],
        "additionalProperties": false
    });

    let response = client
        .complete_json(SYSTEM_PROMPT, text, schema)
        .await
        .context("Synopsis generation failed")?;

    let parsed: SynopsisResponse =
        serde_json::from_str(&response).context("Failed to parse synopsis response")?;

    Ok(parsed.synopsis)
}
