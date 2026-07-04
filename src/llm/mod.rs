//! LLM completion client for structured extraction tasks.
//!
//! Provides an OpenAI-compatible chat completion client that uses
//! `response_format: json_schema` for guaranteed structured output.

pub mod enrichment;
pub mod intents;
pub mod relations;
pub mod synopsis;
pub mod topo_agents;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Configuration for the LLM completion client.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// API key for authentication.
    pub api_key: String,
    /// Base URL for the OpenAI-compatible API (e.g. "https://api.openai.com/v1").
    pub base_url: String,
    /// Model name (e.g. "gpt-4.1-mini").
    pub model: String,
}

impl LlmConfig {
    /// Resolve LLM configuration from explicit values, env vars, and .env.openai file.
    ///
    /// Priority: explicit args > env vars > .env.openai file.
    pub fn resolve(
        api_key: &Option<String>,
        base_url: &Option<String>,
        model: &Option<String>,
    ) -> Result<Self> {
        use crate::config::resolve_setting;

        let api_key = resolve_setting(api_key, "OPENAI_API_KEY", ".env.openai").ok_or_else(
            || anyhow::anyhow!("LLM API key not found (set OPENAI_API_KEY or --api-key)"),
        )?;

        let base_url = resolve_setting(base_url, "OPENAI_BASE_URL", ".env.openai")
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        let model = resolve_setting(model, "COGNIGRAPH_LLM_MODEL", ".env.openai")
            .unwrap_or_else(|| "gpt-4.1-mini".to_string());

        Ok(Self {
            api_key,
            base_url,
            model,
        })
    }
}

/// OpenAI-compatible chat completion client.
pub struct CompletionClient {
    client: reqwest::Client,
    config: LlmConfig,
}

impl CompletionClient {
    pub fn new(config: LlmConfig) -> Result<Self> {
        let client = crate::http_util::build_client(false)?;
        Ok(Self { client, config })
    }

    /// Send a chat completion request with structured JSON output.
    ///
    /// Returns the parsed response content as a string.
    pub async fn complete_json(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        json_schema: serde_json::Value,
    ) -> Result<String> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let request = ChatRequest {
            model: &self.config.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user",
                    content: user_prompt,
                },
            ],
            response_format: ResponseFormat {
                r#type: "json_schema",
                json_schema: JsonSchemaSpec {
                    name: "extraction",
                    strict: true,
                    schema: json_schema,
                },
            },
            temperature: 0.0,
        };

        let (status, body) = crate::http_util::send_with_retry(
            self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.config.api_key))
                .json(&request),
            "LLM completion",
        )
        .await?;

        if !status.is_success() {
            if let Ok(err) = serde_json::from_str::<ErrorResponse>(&body) {
                bail!("LLM API error ({}): {}", status, err.error.message);
            }
            bail!("LLM API error ({}): {}", status, body);
        }

        let parsed: ChatResponse =
            serde_json::from_str(&body).context("Failed to parse LLM response")?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("LLM returned no choices"))
    }
}

// ── Request/Response types ──────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    response_format: ResponseFormat,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ResponseFormat {
    r#type: &'static str,
    json_schema: JsonSchemaSpec,
}

#[derive(Serialize)]
struct JsonSchemaSpec {
    name: &'static str,
    strict: bool,
    schema: serde_json::Value,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: MessageContent,
}

#[derive(Deserialize)]
struct MessageContent {
    content: String,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

