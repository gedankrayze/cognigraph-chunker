//! Cognitive chunking subcommand.

use std::io::{self, Read};
use std::path::PathBuf;

use clap::Args;
use serde::Serialize;

use cognigraph_chunker::embeddings::cloudflare::{
    CloudflareProvider, resolve_cloudflare_credentials,
};
use cognigraph_chunker::embeddings::oauth::{OAuthProvider, resolve_oauth_credentials};
use cognigraph_chunker::embeddings::ollama::OllamaProvider;
use cognigraph_chunker::embeddings::onnx::OnnxProvider;
use cognigraph_chunker::embeddings::openai::OpenAiProvider;
use cognigraph_chunker::embeddings::reranker::RerankerProvider;
use cognigraph_chunker::embeddings::{EmbeddingProvider, ProviderType};
use cognigraph_chunker::output::OutputFormat;
use cognigraph_chunker::semantic::cognitive_types::{
    CognitiveConfig, CognitiveResult, CognitiveWeights,
};
use cognigraph_chunker::semantic::diagnostics::{emit_signals_tsv, signals_to_json};
use cognigraph_chunker::semantic::evaluation::format_metrics;
use cognigraph_chunker::semantic::{cognitive_chunk, cognitive_chunk_plain};

use super::global_opts::{self, GlobalOpts};

#[derive(Args)]
pub struct CognitiveArgs {
    /// Input file path, or "-" for stdin (default: stdin)
    #[arg(short, long, default_value = "-")]
    pub input: String,

    /// Embedding provider
    #[arg(short, long, value_enum, default_value_t = ProviderType::Ollama)]
    pub provider: ProviderType,

    /// Model name (provider-specific default if omitted)
    #[arg(short, long)]
    pub model: Option<String>,

    /// API key (for OpenAI; also reads OPENAI_API_KEY env or .env.openai file)
    #[arg(long)]
    pub api_key: Option<String>,

    /// Base URL override for the embedding API
    #[arg(long)]
    pub base_url: Option<String>,

    /// Path to ONNX model directory (for onnx provider)
    #[arg(long)]
    pub model_path: Option<String>,

    /// Cloudflare auth token
    #[arg(long)]
    pub cf_auth_token: Option<String>,

    /// Cloudflare account ID
    #[arg(long)]
    pub cf_account_id: Option<String>,

    /// Cloudflare AI Gateway name
    #[arg(long)]
    pub cf_ai_gateway: Option<String>,

    /// OAuth token endpoint URL
    #[arg(long)]
    pub oauth_token_url: Option<String>,

    /// OAuth client ID
    #[arg(long)]
    pub oauth_client_id: Option<String>,

    /// OAuth client secret
    #[arg(long)]
    pub oauth_client_secret: Option<String>,

    /// OAuth scope
    #[arg(long)]
    pub oauth_scope: Option<String>,

    /// OAuth base URL for the OpenAI-compatible API
    #[arg(long)]
    pub oauth_base_url: Option<String>,

    /// Accept invalid TLS certificates (for corporate proxies)
    #[arg(long)]
    pub danger_accept_invalid_certs: bool,

    /// Soft token budget per chunk (assembly prefers to stay under this)
    #[arg(long, default_value_t = 512)]
    pub soft_budget: usize,

    /// Hard token ceiling per chunk (never exceed unless single block is larger)
    #[arg(long, default_value_t = 768)]
    pub hard_budget: usize,

    /// Window size for cross-similarity computation (must be odd, >= 3)
    #[arg(long, default_value_t = 3)]
    pub sim_window: usize,

    /// Savitzky-Golay smoothing window size (must be odd)
    #[arg(long, default_value_t = 11)]
    pub sg_window: usize,

    /// Savitzky-Golay polynomial order
    #[arg(long, default_value_t = 3)]
    pub poly_order: usize,

    /// Emit full boundary signal diagnostics to stderr
    #[arg(long)]
    pub emit_signals: bool,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Plain)]
    pub format: OutputFormat,

    /// Disable markdown-aware parsing (treat input as plain text)
    #[arg(long)]
    pub no_markdown: bool,

    /// Include relation triples in output metadata
    #[arg(long)]
    pub relations: bool,

    /// Language override (auto-detect if omitted).
    /// Supported: en, de, fr, es, pt, it, nl, ru, zh, ja, ko, ar, tr, pl
    #[arg(long)]
    pub language: Option<String>,

    /// Reranker for ambiguous boundary refinement.
    /// Accepted values:
    ///   nvidia          — NVIDIA NIM API (requires NVIDIA_API_KEY env var)
    ///   cohere          — Cohere Rerank API (requires COHERE_API_KEY env var)
    ///   cloudflare      — Cloudflare Workers AI (requires CLOUDFLARE_AUTH_TOKEN + CLOUDFLARE_ACCOUNT_ID)
    ///   oauth           — OAuth-authenticated rerank endpoint (reuses .env.oauth credentials)
    ///   onnx:<path>     — local ONNX model directory (must contain model.onnx + tokenizer.json)
    ///   <path>          — alias for onnx:<path> (backwards compatible)
    #[arg(long)]
    pub reranker: Option<String>,

    /// Output as a graph structure (nodes + edges) instead of flat chunks.
    /// Overrides --format when set.
    #[arg(long)]
    pub graph: bool,

    /// Generate LLM-based synopsis for each chunk (requires OpenAI API key).
    #[arg(long)]
    pub synopsis: bool,
}

pub async fn run(args: &CognitiveArgs, global: &GlobalOpts) -> anyhow::Result<()> {
    let text = read_input(&args.input, global.max_input_size)?;
    let text_str = String::from_utf8_lossy(&text);

    global.detail(&format!(
        "[cognitive] input: {} bytes, provider: {:?}, markdown: {}, budget: {}/{}",
        text.len(),
        args.provider,
        !args.no_markdown,
        args.soft_budget,
        args.hard_budget,
    ));

    let language = args
        .language
        .as_deref()
        .map(|s| {
            s.parse::<cognigraph_chunker::semantic::enrichment::language::LanguageGroup>()
                .map_err(|e| anyhow::anyhow!(e))
        })
        .transpose()?;

    let config = CognitiveConfig {
        weights: CognitiveWeights::default(),
        soft_budget: args.soft_budget,
        hard_budget: args.hard_budget,
        sim_window: args.sim_window,
        sg_window: args.sg_window,
        poly_order: args.poly_order,
        max_blocks: 10_000,
        emit_signals: args.emit_signals,
        language,
    };

    match args.provider {
        ProviderType::Ollama => {
            let provider = OllamaProvider::new(args.base_url.clone(), args.model.clone())?;
            run_pipeline(&text_str, &provider, &config, args, global).await
        }
        ProviderType::Openai => {
            let api_key = resolve_openai_key(&args.api_key)?;
            let provider = OpenAiProvider::new(api_key, args.base_url.clone(), args.model.clone())?;
            run_pipeline(&text_str, &provider, &config, args, global).await
        }
        ProviderType::Onnx => {
            let model_path = args
                .model_path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--model-path is required for onnx provider."))?;
            let provider = OnnxProvider::new(model_path)?;
            run_pipeline(&text_str, &provider, &config, args, global).await
        }
        ProviderType::Cloudflare => {
            let (token, account_id, gateway) = resolve_cloudflare_credentials(
                &args.cf_auth_token,
                &args.cf_account_id,
                &args.cf_ai_gateway,
            )?;
            let provider = CloudflareProvider::new(token, account_id, args.model.clone(), gateway)?;
            global.detail("[cloudflare] verifying auth token...");
            provider.verify_token().await?;
            global.detail("[cloudflare] token verified");
            run_pipeline(&text_str, &provider, &config, args, global).await
        }
        ProviderType::Oauth => {
            let creds = resolve_oauth_credentials(
                &args.oauth_token_url,
                &args.oauth_client_id,
                &args.oauth_client_secret,
                &args.oauth_scope,
                &args.oauth_base_url,
                &args.model,
            )?;
            let provider = OAuthProvider::new(
                creds.token_url,
                creds.client_id,
                creds.client_secret,
                creds.scope,
                creds.base_url,
                creds.model,
                args.danger_accept_invalid_certs,
            )?;
            global.detail("[oauth] acquiring token...");
            provider.verify_credentials().await?;
            global.detail("[oauth] token acquired");
            run_pipeline(&text_str, &provider, &config, args, global).await
        }
    }
}

async fn run_pipeline<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    config: &CognitiveConfig,
    args: &CognitiveArgs,
    global: &GlobalOpts,
) -> anyhow::Result<()> {
    let mut result = if let Some(ref reranker_spec) = args.reranker {
        let reranker = build_reranker(reranker_spec, global)?;
        if args.no_markdown {
            cognigraph_chunker::semantic::cognitive_chunk_plain_with_reranker(
                text, provider, config, &reranker,
            )
            .await?
        } else {
            cognigraph_chunker::semantic::cognitive_chunk_with_reranker(
                text, provider, config, &reranker,
            )
            .await?
        }
    } else if args.no_markdown {
        cognitive_chunk_plain(text, provider, config).await?
    } else {
        cognitive_chunk(text, provider, config).await?
    };

    // Emit diagnostic signals if requested
    if args.emit_signals && !result.signals.is_empty() {
        emit_signals_tsv(&result.signals);
    }

    global.info(&format!(
        "[cognitive] {} blocks → {} chunks",
        result.block_count,
        result.chunks.len(),
    ));

    // LLM-based relation extraction (post-assembly)
    if args.relations {
        global.detail("[relations] extracting via LLM...");
        let llm_config = cognigraph_chunker::llm::LlmConfig::resolve(
            &args.api_key,
            &args.base_url,
            &None, // model resolved from env
        )?;
        let llm_client = cognigraph_chunker::llm::CompletionClient::new(llm_config)?;

        for chunk in &mut result.chunks {
            match cognigraph_chunker::llm::relations::extract_relations(&llm_client, &chunk.text)
                .await
            {
                Ok(rels) => {
                    global.detail(&format!(
                        "[relations] chunk {}: {} triples",
                        chunk.offset_start,
                        rels.len()
                    ));
                    chunk.dominant_relations = rels;
                }
                Err(e) => {
                    global.detail(&format!(
                        "[relations] chunk {} failed: {e}",
                        chunk.offset_start
                    ));
                }
            }
        }
        global.detail("[relations] done");
    }

    // LLM-based synopsis generation (post-assembly)
    if args.synopsis {
        global.detail("[synopsis] generating via LLM...");
        let llm_config =
            cognigraph_chunker::llm::LlmConfig::resolve(&args.api_key, &args.base_url, &None)?;
        let llm_client = cognigraph_chunker::llm::CompletionClient::new(llm_config)?;

        for chunk in &mut result.chunks {
            match cognigraph_chunker::llm::synopsis::generate_synopsis(&llm_client, &chunk.text)
                .await
            {
                Ok(synopsis) => {
                    global.detail(&format!(
                        "[synopsis] chunk {}: {}",
                        chunk.offset_start, synopsis
                    ));
                    chunk.synopsis = Some(synopsis);
                }
                Err(e) => {
                    global.detail(&format!(
                        "[synopsis] chunk {} failed: {e}",
                        chunk.offset_start
                    ));
                }
            }
        }
        global.detail("[synopsis] done");
    }

    // Convert to output format
    if args.graph {
        let graph = cognigraph_chunker::semantic::graph_export::to_chunk_graph(&result);
        println!("{}", serde_json::to_string_pretty(&graph).unwrap());
    } else {
        write_cognitive_output(&result, args.format, args.emit_signals, args.relations);
    }

    // Print stats
    let stats_chunks: Vec<(String, usize)> = result
        .chunks
        .iter()
        .map(|c| (c.text.clone(), c.offset_start))
        .collect();
    global_opts::print_stats(&stats_chunks, global);

    // Print evaluation metrics
    if global.stats {
        eprintln!("{}", format_metrics(&result.evaluation));
    }

    Ok(())
}

fn write_cognitive_output(
    result: &CognitiveResult,
    format: OutputFormat,
    include_signals: bool,
    include_relations: bool,
) {
    let shared_entities_json: serde_json::Value = if result.shared_entities.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::to_value(&result.shared_entities).unwrap_or(serde_json::Value::Null)
    };
    match format {
        OutputFormat::Plain => {
            for (i, chunk) in result.chunks.iter().enumerate() {
                if i > 0 {
                    println!();
                }
                print!("{}", chunk.text);
            }
            println!();
        }
        OutputFormat::Json => {
            let entries: Vec<CognitiveChunkEntry> = result
                .chunks
                .iter()
                .enumerate()
                .map(|(i, c)| CognitiveChunkEntry {
                    index: i,
                    text: &c.text,
                    offset_start: c.offset_start,
                    offset_end: c.offset_end,
                    length: c.text.len(),
                    heading_path: &c.heading_path,
                    dominant_entities: &c.dominant_entities,
                    dominant_relations: if include_relations {
                        c.dominant_relations
                            .iter()
                            .map(|r| RelationEntry {
                                subject: r.subject.clone(),
                                predicate: r.predicate.clone(),
                                object: r.object.clone(),
                            })
                            .collect()
                    } else {
                        vec![]
                    },
                    token_estimate: c.token_estimate,
                    continuity_confidence: c.continuity_confidence,
                    synopsis: c.synopsis.clone(),
                    prev_chunk: c.prev_chunk,
                    next_chunk: c.next_chunk,
                })
                .collect();

            let eval = serde_json::json!({
                "entity_orphan_rate": result.evaluation.entity_orphan_rate,
                "pronoun_boundary_rate": result.evaluation.pronoun_boundary_rate,
                "heading_attachment_rate": result.evaluation.heading_attachment_rate,
                "discourse_break_rate": result.evaluation.discourse_break_rate,
                "triple_severance_rate": result.evaluation.triple_severance_rate,
            });

            if include_signals && !result.signals.is_empty() {
                let mut output = serde_json::json!({
                    "chunks": entries,
                    "signals": signals_to_json(&result.signals),
                    "block_count": result.block_count,
                    "evaluation": eval,
                });
                if !shared_entities_json.is_null() {
                    output["shared_entities"] = shared_entities_json;
                }
                println!("{}", serde_json::to_string_pretty(&output).unwrap());
            } else {
                let mut output = serde_json::json!({
                    "chunks": entries,
                    "evaluation": eval,
                });
                if !shared_entities_json.is_null() {
                    output["shared_entities"] = shared_entities_json;
                }
                println!("{}", serde_json::to_string_pretty(&output).unwrap());
            }
        }
        OutputFormat::Jsonl => {
            for (i, chunk) in result.chunks.iter().enumerate() {
                let entry = CognitiveChunkEntry {
                    index: i,
                    text: &chunk.text,
                    offset_start: chunk.offset_start,
                    offset_end: chunk.offset_end,
                    length: chunk.text.len(),
                    heading_path: &chunk.heading_path,
                    dominant_entities: &chunk.dominant_entities,
                    dominant_relations: chunk
                        .dominant_relations
                        .iter()
                        .map(|r| RelationEntry {
                            subject: r.subject.clone(),
                            predicate: r.predicate.clone(),
                            object: r.object.clone(),
                        })
                        .collect(),
                    token_estimate: chunk.token_estimate,
                    continuity_confidence: chunk.continuity_confidence,
                    synopsis: chunk.synopsis.clone(),
                    prev_chunk: chunk.prev_chunk,
                    next_chunk: chunk.next_chunk,
                };
                println!("{}", serde_json::to_string(&entry).unwrap());
            }
        }
    }
}

#[derive(Serialize)]
struct CognitiveChunkEntry<'a> {
    index: usize,
    text: &'a str,
    offset_start: usize,
    offset_end: usize,
    length: usize,
    heading_path: &'a [String],
    dominant_entities: &'a [String],
    #[serde(skip_serializing_if = "Vec::is_empty")]
    dominant_relations: Vec<RelationEntry>,
    token_estimate: usize,
    continuity_confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    synopsis: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prev_chunk: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_chunk: Option<usize>,
}

#[derive(Serialize)]
struct RelationEntry {
    subject: String,
    predicate: String,
    object: String,
}

fn read_input(input: &str, max_size: usize) -> anyhow::Result<Vec<u8>> {
    if input == "-" {
        let mut buf = Vec::new();
        io::stdin()
            .take(max_size as u64 + 1)
            .read_to_end(&mut buf)?;
        anyhow::ensure!(
            buf.len() <= max_size,
            "Stdin input exceeds maximum allowed size ({max_size} bytes)."
        );
        Ok(buf)
    } else {
        let path = PathBuf::from(input);
        anyhow::ensure!(path.exists(), "File not found: {}", path.display());
        let meta = std::fs::metadata(&path)?;
        anyhow::ensure!(
            meta.len() <= max_size as u64,
            "File size ({} bytes) exceeds maximum ({max_size} bytes).",
            meta.len()
        );
        Ok(std::fs::read(&path)?)
    }
}

fn resolve_openai_key(flag: &Option<String>) -> anyhow::Result<String> {
    if let Some(key) = flag {
        return Ok(key.clone());
    }
    if let Ok(key) = std::env::var("OPENAI_API_KEY")
        && !key.is_empty()
    {
        return Ok(key);
    }
    if let Ok(content) = std::fs::read_to_string(".env.openai") {
        for line in content.lines() {
            let line = line.trim();
            if let Some(val) = line.strip_prefix("OPENAI_API_KEY=") {
                let val = val.trim();
                if !val.is_empty() {
                    return Ok(val.to_string());
                }
            }
        }
    }
    anyhow::bail!("OpenAI API key not found.")
}

/// Parse the `--reranker` flag and construct the appropriate provider.
///
/// Accepted formats:
/// - `"nvidia"` → NVIDIA NIM (env: NVIDIA_API_KEY)
/// - `"cohere"` → Cohere Rerank (env: COHERE_API_KEY)
/// - `"onnx:<path>"` → local ONNX model directory
/// - `"<path>"` → backwards-compatible alias for `onnx:<path>`
fn build_reranker(
    spec: &str,
    global: &GlobalOpts,
) -> anyhow::Result<cognigraph_chunker::embeddings::reranker::AnyReranker> {
    use cognigraph_chunker::embeddings::reranker::AnyReranker;

    match spec.to_lowercase().as_str() {
        "nvidia" => {
            load_env_file(".env.nvidia");
            let reranker = cognigraph_chunker::embeddings::reranker::NvidiaReranker::from_env()?;
            global.detail(&format!("[reranker] NVIDIA NIM: {}", reranker.model_name()));
            Ok(AnyReranker::Nvidia(reranker))
        }
        "cohere" => {
            load_env_file(".env.cohere");
            let reranker = cognigraph_chunker::embeddings::reranker::CohereReranker::from_env()?;
            global.detail(&format!("[reranker] Cohere: {}", reranker.model_name()));
            Ok(AnyReranker::Cohere(reranker))
        }
        "cloudflare" => {
            load_env_file(".env.cloudflare");
            let reranker =
                cognigraph_chunker::embeddings::reranker::CloudflareReranker::from_env()?;
            global.detail(&format!("[reranker] Cloudflare: {}", reranker.model_name()));
            Ok(AnyReranker::Cloudflare(reranker))
        }
        "oauth" => {
            load_env_file(".env.oauth");
            let reranker =
                cognigraph_chunker::embeddings::reranker::OAuthReranker::from_env(false)?;
            global.detail(&format!("[reranker] OAuth: {}", reranker.model_name()));
            Ok(AnyReranker::OAuth(reranker))
        }
        other => {
            let path = other.strip_prefix("onnx:").unwrap_or(other);
            global.detail(&format!("[reranker] loading ONNX model from {path}"));
            let reranker = cognigraph_chunker::embeddings::reranker::OnnxReranker::new(path)?;
            global.detail(&format!(
                "[reranker] model loaded: {}",
                reranker.model_name()
            ));
            Ok(AnyReranker::Onnx(Box::new(reranker)))
        }
    }
}

/// Load a dotenv-style file, setting variables that aren't already in the environment.
fn load_env_file(path: &str) {
    if let Ok(content) = std::fs::read_to_string(path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, val)) = line.split_once('=') {
                let key = key.trim();
                let val = val.trim();
                if !val.is_empty() && std::env::var(key).is_err() {
                    unsafe {
                        std::env::set_var(key, val);
                    }
                }
            }
        }
    }
}
