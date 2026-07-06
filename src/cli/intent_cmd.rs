//! Intent-driven chunking subcommand.

use clap::Args;
use serde::Serialize;

use cognigraph_chunker::embeddings::{EmbedProviderOpts, EmbeddingProvider};
use cognigraph_chunker::llm::{CompletionClient, LlmConfig};
use cognigraph_chunker::output::OutputFormat;
use cognigraph_chunker::semantic::intent_chunk::{IntentConfig, intent_chunk, intent_chunk_plain};
use cognigraph_chunker::semantic::intent_types::IntentResult;

use super::global_opts::{self, GlobalOpts};
use super::merge_opts::{MergeOpts, maybe_merge};

#[derive(Args)]
pub struct IntentArgs {
    /// Input file path, or "-" for stdin (default: stdin)
    #[arg(short, long, default_value = "-")]
    pub input: String,

    #[command(flatten)]
    pub provider_opts: EmbedProviderOpts,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Plain)]
    pub format: OutputFormat,

    /// Disable markdown-aware parsing (treat input as plain text)
    #[arg(long)]
    pub no_markdown: bool,

    /// Soft token budget per chunk
    #[arg(long, default_value_t = 512)]
    pub soft_budget: usize,

    /// Hard token ceiling per chunk
    #[arg(long, default_value_t = 768)]
    pub hard_budget: usize,

    /// LLM model for intent generation
    #[arg(long, default_value = "gpt-4.1-mini")]
    pub intent_model: String,

    /// Maximum number of intents to generate
    #[arg(long, default_value_t = 20)]
    pub max_intents: usize,

    /// Base URL for the LLM API (defaults to OpenAI)
    #[arg(long)]
    pub llm_base_url: Option<String>,

    /// Post-process chunks by merging small ones to fit a token budget
    #[command(flatten)]
    pub merge: MergeOpts,
}

pub async fn run(args: &IntentArgs, global: &GlobalOpts) -> anyhow::Result<()> {
    let text = super::read_input(&args.input, global.max_input_size)?;
    let text_str = String::from_utf8_lossy(&text);

    global.detail(&format!(
        "[intent] input: {} bytes, provider: {:?}, markdown: {}, budget: {}/{}, max_intents: {}",
        text.len(),
        args.provider_opts.provider,
        !args.no_markdown,
        args.soft_budget,
        args.hard_budget,
        args.max_intents,
    ));

    // Resolve LLM config for intent generation
    let llm_config = LlmConfig::resolve(
        &args.provider_opts.api_key,
        &args.llm_base_url,
        &Some(args.intent_model.clone()),
    )?;
    let llm_client = CompletionClient::new(llm_config)?;

    let config = IntentConfig {
        max_intents: args.max_intents,
        soft_budget: args.soft_budget,
        hard_budget: args.hard_budget,
    };

    let provider = args.provider_opts.build_provider().await?;
    run_pipeline(&text_str, &provider, &llm_client, &config, args, global).await
}

async fn run_pipeline<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    llm_client: &CompletionClient,
    config: &IntentConfig,
    args: &IntentArgs,
    global: &GlobalOpts,
) -> anyhow::Result<()> {
    let result = if args.no_markdown {
        intent_chunk_plain(text, provider, llm_client, config).await?
    } else {
        intent_chunk(text, provider, llm_client, config).await?
    };

    global.info(&format!(
        "[intent] {} blocks → {} chunks, {} intents, partition_score: {:.4}",
        result.block_count,
        result.chunks.len(),
        result.intents.len(),
        result.partition_score,
    ));

    // Optional merge
    let chunks_as_pairs: Vec<(String, usize)> = result
        .chunks
        .iter()
        .map(|c| (c.text.clone(), c.offset_start))
        .collect();
    let final_chunks = maybe_merge(chunks_as_pairs, &args.merge, global);

    // Write output
    write_intent_output(&result, &final_chunks, args.format);

    // Print stats
    global_opts::print_stats(&final_chunks, global);

    Ok(())
}

fn write_intent_output(result: &IntentResult, chunks: &[(String, usize)], format: OutputFormat) {
    match format {
        OutputFormat::Plain => {
            for (i, (text, _)) in chunks.iter().enumerate() {
                if i > 0 {
                    println!();
                }
                print!("{text}");
            }
            println!();
        }
        OutputFormat::Json => {
            let entries: Vec<IntentChunkEntry> = if chunks.len() == result.chunks.len() {
                // No merge happened: use full metadata
                result
                    .chunks
                    .iter()
                    .enumerate()
                    .map(|(i, c)| IntentChunkEntry {
                        index: i,
                        text: &c.text,
                        offset_start: c.offset_start,
                        offset_end: c.offset_end,
                        length: c.text.len(),
                        token_estimate: c.token_estimate,
                        best_intent: Some(c.best_intent),
                        alignment_score: Some(c.alignment_score),
                        heading_path: c.heading_path.clone(),
                    })
                    .collect()
            } else {
                // Merged: basic metadata only
                chunks
                    .iter()
                    .enumerate()
                    .map(|(i, (text, offset))| IntentChunkEntry {
                        index: i,
                        text,
                        offset_start: *offset,
                        offset_end: offset + text.len(),
                        length: text.len(),
                        token_estimate: text.split_whitespace().count(),
                        best_intent: None,
                        alignment_score: None,
                        heading_path: vec![],
                    })
                    .collect()
            };

            let output = serde_json::json!({
                "chunks": entries,
                "intents": result.intents,
                "partition_score": result.partition_score,
                "block_count": result.block_count,
            });
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
        OutputFormat::Jsonl => {
            if chunks.len() == result.chunks.len() {
                for (i, c) in result.chunks.iter().enumerate() {
                    let entry = IntentChunkEntry {
                        index: i,
                        text: &c.text,
                        offset_start: c.offset_start,
                        offset_end: c.offset_end,
                        length: c.text.len(),
                        token_estimate: c.token_estimate,
                        best_intent: Some(c.best_intent),
                        alignment_score: Some(c.alignment_score),
                        heading_path: c.heading_path.clone(),
                    };
                    println!("{}", serde_json::to_string(&entry).unwrap());
                }
            } else {
                for (i, (text, offset)) in chunks.iter().enumerate() {
                    let entry = IntentChunkEntry {
                        index: i,
                        text,
                        offset_start: *offset,
                        offset_end: offset + text.len(),
                        length: text.len(),
                        token_estimate: text.split_whitespace().count(),
                        best_intent: None,
                        alignment_score: None,
                        heading_path: vec![],
                    };
                    println!("{}", serde_json::to_string(&entry).unwrap());
                }
            }
        }
    }
}

#[derive(Serialize)]
struct IntentChunkEntry<'a> {
    index: usize,
    text: &'a str,
    offset_start: usize,
    offset_end: usize,
    length: usize,
    token_estimate: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_intent: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    alignment_score: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    heading_path: Vec<String>,
}
