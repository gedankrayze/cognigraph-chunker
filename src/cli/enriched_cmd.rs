//! Enriched chunking subcommand.

use std::io::{self, Read};
use std::path::PathBuf;

use clap::Args;
use serde::Serialize;

use cognigraph_chunker::llm::{CompletionClient, LlmConfig};
use cognigraph_chunker::output::OutputFormat;
use cognigraph_chunker::semantic::enriched_chunk::{EnrichedConfig, enriched_chunk, enriched_chunk_plain};
use cognigraph_chunker::semantic::enriched_types::{EnrichedResult, TypedEntity};

use super::global_opts::{self, GlobalOpts};

#[derive(Args)]
pub struct EnrichedArgs {
    /// Input file path, or "-" for stdin (default: stdin)
    #[arg(short, long, default_value = "-")]
    pub input: String,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Plain)]
    pub format: OutputFormat,

    /// Disable markdown-aware parsing (treat input as plain text)
    #[arg(long)]
    pub no_markdown: bool,

    /// Soft token budget per chunk (assembly prefers to stay under this)
    #[arg(long, default_value_t = 512)]
    pub soft_budget: usize,

    /// Hard token ceiling per chunk (never exceed unless single block is larger)
    #[arg(long, default_value_t = 768)]
    pub hard_budget: usize,

    /// Disable semantic-key recombination
    #[arg(long)]
    pub no_recombine: bool,

    /// Disable re-enrichment of merged chunks
    #[arg(long)]
    pub no_re_enrich: bool,

    /// LLM model for enrichment (default: gpt-4.1-mini)
    #[arg(long)]
    pub enrichment_model: Option<String>,

    /// API key (also reads OPENAI_API_KEY env or .env.openai file)
    #[arg(long)]
    pub api_key: Option<String>,

    /// Base URL for the LLM API
    #[arg(long)]
    pub llm_base_url: Option<String>,
}

pub async fn run(args: &EnrichedArgs, global: &GlobalOpts) -> anyhow::Result<()> {
    let text = read_input(&args.input, global.max_input_size)?;
    let text_str = String::from_utf8_lossy(&text);

    global.detail(&format!(
        "[enriched] input: {} bytes, markdown: {}, budget: {}/{}",
        text.len(),
        !args.no_markdown,
        args.soft_budget,
        args.hard_budget,
    ));

    // Resolve LLM config
    let llm_config = LlmConfig::resolve(
        &args.api_key,
        &args.llm_base_url,
        &args.enrichment_model,
    )?;
    let llm_client = CompletionClient::new(llm_config)?;

    let config = EnrichedConfig {
        soft_budget: args.soft_budget,
        hard_budget: args.hard_budget,
        recombine: !args.no_recombine,
        re_enrich: !args.no_re_enrich,
    };

    let result = if args.no_markdown {
        enriched_chunk_plain(&text_str, &llm_client, &config).await?
    } else {
        enriched_chunk(&text_str, &llm_client, &config).await?
    };

    global.info(&format!(
        "[enriched] {} blocks -> {} chunks, {} merges",
        result.block_count,
        result.chunks.len(),
        result.merge_history.len(),
    ));

    write_enriched_output(&result, args.format);

    // Print stats
    let stats_chunks: Vec<(String, usize)> = result
        .chunks
        .iter()
        .map(|c| (c.text.clone(), c.offset_start))
        .collect();
    global_opts::print_stats(&stats_chunks, global);

    Ok(())
}

fn write_enriched_output(result: &EnrichedResult, format: OutputFormat) {
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
            let entries: Vec<EnrichedChunkEntry> = result
                .chunks
                .iter()
                .enumerate()
                .map(|(i, c)| EnrichedChunkEntry {
                    index: i,
                    text: &c.text,
                    offset_start: c.offset_start,
                    offset_end: c.offset_end,
                    length: c.text.len(),
                    token_estimate: c.token_estimate,
                    title: &c.title,
                    summary: &c.summary,
                    keywords: &c.keywords,
                    typed_entities: &c.typed_entities,
                    hypothetical_questions: &c.hypothetical_questions,
                    semantic_keys: &c.semantic_keys,
                    category: &c.category,
                    heading_path: &c.heading_path,
                })
                .collect();

            let output = serde_json::json!({
                "chunks": entries,
                "key_dictionary": result.key_dictionary,
                "merge_history": result.merge_history,
                "block_count": result.block_count,
            });
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
        OutputFormat::Jsonl => {
            for (i, chunk) in result.chunks.iter().enumerate() {
                let entry = EnrichedChunkEntry {
                    index: i,
                    text: &chunk.text,
                    offset_start: chunk.offset_start,
                    offset_end: chunk.offset_end,
                    length: chunk.text.len(),
                    token_estimate: chunk.token_estimate,
                    title: &chunk.title,
                    summary: &chunk.summary,
                    keywords: &chunk.keywords,
                    typed_entities: &chunk.typed_entities,
                    hypothetical_questions: &chunk.hypothetical_questions,
                    semantic_keys: &chunk.semantic_keys,
                    category: &chunk.category,
                    heading_path: &chunk.heading_path,
                };
                println!("{}", serde_json::to_string(&entry).unwrap());
            }
        }
    }
}

#[derive(Serialize)]
struct EnrichedChunkEntry<'a> {
    index: usize,
    text: &'a str,
    offset_start: usize,
    offset_end: usize,
    length: usize,
    token_estimate: usize,
    title: &'a str,
    summary: &'a str,
    keywords: &'a [String],
    typed_entities: &'a [TypedEntity],
    hypothetical_questions: &'a [String],
    semantic_keys: &'a [String],
    category: &'a str,
    heading_path: &'a [String],
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
