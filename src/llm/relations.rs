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

/// Maximum concurrent LLM requests for per-chunk extraction.
pub(crate) const LLM_CONCURRENCY: usize = 8;

/// Extract relations for multiple chunks concurrently (bounded).
///
/// Returns one `Vec<RelationTriple>` per chunk, in the same order.
/// Fails on the first extraction error.
pub async fn extract_relations_batch(
    client: &CompletionClient,
    chunks: &[&str],
) -> Result<Vec<Vec<RelationTriple>>> {
    use futures::stream::{self, StreamExt, TryStreamExt};

    stream::iter(chunks.iter())
        .map(|chunk| extract_relations(client, chunk))
        .buffered(LLM_CONCURRENCY)
        .try_collect()
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmConfig;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Mock OpenAI-compatible chat endpoint: waits 100ms, then returns one
    /// relation triple. Counts requests.
    async fn spawn_llm_server() -> (String, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let server_counter = counter.clone();

        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                server_counter.fetch_add(1, Ordering::SeqCst);
                // Handle each connection concurrently so the server itself
                // is not the serialization point.
                tokio::spawn(async move {
                    let mut buf = [0u8; 65536];
                    let _ = socket.read(&mut buf).await;
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    let content = r#"{\"relations\":[{\"subject\":\"s\",\"predicate\":\"p\",\"object\":\"o\"}]}"#;
                    let body =
                        format!(r#"{{"choices":[{{"message":{{"content":"{content}"}}}}]}}"#);
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });

        (format!("http://{addr}"), counter)
    }

    #[tokio::test]
    async fn test_batch_extraction_runs_concurrently() {
        let (base_url, requests) = spawn_llm_server().await;
        let client = CompletionClient::new(LlmConfig {
            api_key: "test-key".into(),
            base_url,
            model: "test-model".into(),
        })
        .unwrap();

        let chunk = "This chunk is long enough to trigger a real extraction call.";
        let chunks = vec![chunk; 6];

        let start = std::time::Instant::now();
        let results = extract_relations_batch(&client, &chunks).await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 6, "one result per chunk, in order");
        assert!(results.iter().all(|r| r.len() == 1));
        assert_eq!(requests.load(Ordering::SeqCst), 6);
        // 6 requests at 100ms server latency each: sequential ≈ 600ms,
        // concurrent (buffered) well under half that.
        assert!(
            elapsed < std::time::Duration::from_millis(400),
            "batch extraction must overlap requests, took {elapsed:?}"
        );
    }
}
