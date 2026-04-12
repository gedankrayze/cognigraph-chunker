//! Topology-aware chunking subcommand.

use std::io::{self, Read};
use std::path::PathBuf;

use clap::Args;
use serde::Serialize;

use cognigraph_chunker::llm::{CompletionClient, LlmConfig};
use cognigraph_chunker::output::OutputFormat;
use cognigraph_chunker::semantic::topo_chunk::{TopoConfig, topo_chunk};

use super::global_opts::{self, GlobalOpts};

#[derive(Args)]
pub struct TopoArgs {
    /// Input file path, or "-" for stdin (default: stdin)
    #[arg(short, long, default_value = "-")]
    pub input: String,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Plain)]
    pub format: OutputFormat,

    /// Soft token budget per chunk
    #[arg(long, default_value_t = 512)]
    pub soft_budget: usize,

    /// Hard token ceiling per chunk
    #[arg(long, default_value_t = 768)]
    pub hard_budget: usize,

    /// Emit the SIR in JSON output
    #[arg(long)]
    pub emit_sir: bool,

    /// LLM model for topology agents
    #[arg(long, default_value = "gpt-4.1-mini")]
    pub topo_model: String,

    /// API key for the LLM (also reads OPENAI_API_KEY env or .env.openai)
    #[arg(long)]
    pub api_key: Option<String>,

    /// Base URL for the LLM API (defaults to OpenAI)
    #[arg(long)]
    pub llm_base_url: Option<String>,
}

pub async fn run(args: &TopoArgs, global: &GlobalOpts) -> anyhow::Result<()> {
    let text = read_input(&args.input, global.max_input_size)?;
    let text_str = String::from_utf8_lossy(&text);

    global.detail(&format!(
        "[topo] input: {} bytes, budget: {}/{}, model: {}",
        text.len(),
        args.soft_budget,
        args.hard_budget,
        args.topo_model,
    ));

    // Resolve LLM config
    let llm_config = LlmConfig::resolve(
        &args.api_key,
        &args.llm_base_url,
        &Some(args.topo_model.clone()),
    )?;
    let llm_client = CompletionClient::new(llm_config)?;

    let config = TopoConfig {
        soft_budget: args.soft_budget,
        hard_budget: args.hard_budget,
        emit_sir: args.emit_sir,
    };

    let result = topo_chunk(&text_str, &llm_client, &config).await?;

    global.info(&format!(
        "[topo] {} blocks -> {} chunks, {} classifications",
        result.block_count,
        result.chunks.len(),
        result.classifications.len(),
    ));

    // Write output
    match args.format {
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
            let entries: Vec<TopoChunkEntry> = result
                .chunks
                .iter()
                .enumerate()
                .map(|(i, c)| TopoChunkEntry {
                    index: i,
                    text: &c.text,
                    offset_start: c.offset_start,
                    offset_end: c.offset_end,
                    length: c.text.len(),
                    token_estimate: c.token_estimate,
                    heading_path: c.heading_path.clone(),
                    section_classification: &c.section_classification,
                    cross_references: c.cross_references.clone(),
                })
                .collect();

            let mut output = serde_json::json!({
                "chunks": entries,
                "block_count": result.block_count,
                "classifications": result.classifications,
            });

            if args.emit_sir {
                output["sir"] = serde_json::to_value(&result.sir).unwrap();
            }

            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
        OutputFormat::Jsonl => {
            for (i, c) in result.chunks.iter().enumerate() {
                let entry = TopoChunkEntry {
                    index: i,
                    text: &c.text,
                    offset_start: c.offset_start,
                    offset_end: c.offset_end,
                    length: c.text.len(),
                    token_estimate: c.token_estimate,
                    heading_path: c.heading_path.clone(),
                    section_classification: &c.section_classification,
                    cross_references: c.cross_references.clone(),
                };
                println!("{}", serde_json::to_string(&entry).unwrap());
            }
        }
    }

    // Print stats
    let chunks_as_pairs: Vec<(String, usize)> = result
        .chunks
        .iter()
        .map(|c| (c.text.clone(), c.offset_start))
        .collect();
    global_opts::print_stats(&chunks_as_pairs, global);

    Ok(())
}

#[derive(Serialize)]
struct TopoChunkEntry<'a> {
    index: usize,
    text: &'a str,
    offset_start: usize,
    offset_end: usize,
    length: usize,
    token_estimate: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    heading_path: Vec<String>,
    section_classification: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    cross_references: Vec<usize>,
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
