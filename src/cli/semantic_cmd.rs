//! Semantic chunking subcommand.

use std::io::{self, Read};
use std::path::PathBuf;

use clap::Args;

use cognigraph_chunker::embeddings::ollama::OllamaProvider;
use cognigraph_chunker::embeddings::onnx::OnnxProvider;
use cognigraph_chunker::embeddings::openai::OpenAiProvider;
use cognigraph_chunker::embeddings::{EmbeddingProvider, ProviderType};
use cognigraph_chunker::output::{OutputFormat, write_chunks};
use cognigraph_chunker::semantic::{SemanticConfig, semantic_chunk, semantic_chunk_plain};

use super::global_opts::{self, GlobalOpts};
use super::merge_opts::{MergeOpts, maybe_merge};

#[derive(Args)]
pub struct SemanticArgs {
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

    /// Window size for cross-similarity computation (must be odd, >= 3)
    #[arg(long, default_value_t = 3)]
    pub sim_window: usize,

    /// Savitzky-Golay smoothing window size (must be odd)
    #[arg(long, default_value_t = 11)]
    pub sg_window: usize,

    /// Savitzky-Golay polynomial order
    #[arg(long, default_value_t = 3)]
    pub poly_order: usize,

    /// Percentile threshold for split point filtering (0.0-1.0)
    #[arg(long, default_value_t = 0.5)]
    pub threshold: f64,

    /// Minimum block gap between split points
    #[arg(long, default_value_t = 2)]
    pub min_distance: usize,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Plain)]
    pub format: OutputFormat,

    /// Emit raw and smoothed distance curves to stderr (for debugging)
    #[arg(long)]
    pub emit_distances: bool,

    /// Disable markdown-aware parsing (treat input as plain text)
    #[arg(long)]
    pub no_markdown: bool,

    #[command(flatten)]
    pub merge_opts: MergeOpts,
}

pub async fn run(args: &SemanticArgs, global: &GlobalOpts) -> anyhow::Result<()> {
    let text = read_input(&args.input, global.max_input_size)?;
    let text_str = String::from_utf8_lossy(&text);

    global.detail(&format!(
        "[semantic] input: {} bytes, provider: {:?}, markdown: {}",
        text.len(),
        args.provider,
        !args.no_markdown
    ));

    let config = SemanticConfig {
        sim_window: args.sim_window,
        sg_window: args.sg_window,
        poly_order: args.poly_order,
        threshold: args.threshold,
        min_distance: args.min_distance,
        ..SemanticConfig::default()
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
            let model_path = args.model_path.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--model-path is required for onnx provider.\n\
                     Provide the path to a directory containing model.onnx and tokenizer.json."
                )
            })?;
            let provider = OnnxProvider::new(model_path)?;
            run_pipeline(&text_str, &provider, &config, args, global).await
        }
    }
}

async fn run_pipeline<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    config: &SemanticConfig,
    args: &SemanticArgs,
    global: &GlobalOpts,
) -> anyhow::Result<()> {
    let result = if args.no_markdown {
        semantic_chunk_plain(text, provider, config).await?
    } else {
        semantic_chunk(text, provider, config).await?
    };

    if args.emit_distances {
        emit_distances_to_stderr(&result.similarities, &result.smoothed);
    }

    let chunks = maybe_merge(result.chunks, &args.merge_opts, global);

    // Print semantic stats (before write, so it appears above output in piped scenarios)
    print_semantic_info(
        &result.block_stats,
        &result.split_indices,
        chunks.len(),
        args.no_markdown,
        global,
    );

    write_chunks(&chunks, args.format);
    global_opts::print_stats(&chunks, global);

    Ok(())
}

fn print_semantic_info(
    stats: &cognigraph_chunker::semantic::BlockStats,
    split_indices: &cognigraph_chunker::core::savgol::FilteredIndices,
    final_chunk_count: usize,
    plain_mode: bool,
    global: &GlobalOpts,
) {
    let total = stats.total();

    if plain_mode {
        global.info(&format!(
            "[semantic] {} sentences → {} chunks ({} split points)",
            total,
            final_chunk_count,
            split_indices.indices.len(),
        ));
    } else {
        let mut parts = Vec::new();
        if stats.sentences > 0 {
            parts.push(format!("{} sentences", stats.sentences));
        }
        if stats.tables > 0 {
            parts.push(format!("{} tables", stats.tables));
        }
        if stats.code_blocks > 0 {
            parts.push(format!("{} code blocks", stats.code_blocks));
        }
        if stats.headings > 0 {
            parts.push(format!("{} headings", stats.headings));
        }
        if stats.lists > 0 {
            parts.push(format!("{} lists", stats.lists));
        }
        if stats.block_quotes > 0 {
            parts.push(format!("{} block quotes", stats.block_quotes));
        }

        global.info(&format!(
            "[semantic] {} blocks ({}) → {} chunks ({} split points)",
            total,
            parts.join(", "),
            final_chunk_count,
            split_indices.indices.len(),
        ));
    }
}

fn emit_distances_to_stderr(raw: &[f64], smoothed: &[f64]) {
    eprintln!("--- similarity curve ---");
    for (i, (r, s)) in raw.iter().zip(smoothed.iter()).enumerate() {
        eprintln!("{}\t{:.6}\t{:.6}", i, r, s);
    }
    eprintln!("--- end ---");
}

fn read_input(input: &str, max_size: usize) -> anyhow::Result<Vec<u8>> {
    if input == "-" {
        let mut buf = Vec::new();
        io::stdin()
            .take(max_size as u64 + 1)
            .read_to_end(&mut buf)?;
        anyhow::ensure!(
            buf.len() <= max_size,
            "Stdin input exceeds maximum allowed size ({max_size} bytes). \
             Use --max-input-size to increase the limit."
        );
        Ok(buf)
    } else {
        let path = PathBuf::from(input);
        anyhow::ensure!(
            path.exists(),
            "File not found: {}\nCheck the path and try again.",
            path.display()
        );
        let meta = std::fs::metadata(&path)?;
        anyhow::ensure!(
            meta.len() <= max_size as u64,
            "File size ({} bytes) exceeds maximum allowed size ({max_size} bytes). \
             Use --max-input-size to increase the limit.",
            meta.len()
        );
        Ok(std::fs::read(&path)?)
    }
}

/// Resolve OpenAI API key from flag, env var, or .env.openai file.
fn resolve_openai_key(flag: &Option<String>) -> anyhow::Result<String> {
    if let Some(key) = flag {
        return Ok(key.clone());
    }

    if let Ok(key) = std::env::var("OPENAI_API_KEY")
        && !key.is_empty()
    {
        return Ok(key);
    }

    // Try reading from .env.openai file
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

    anyhow::bail!(
        "OpenAI API key not found.\n\
         Provide it via one of:\n  \
         --api-key <KEY>\n  \
         OPENAI_API_KEY environment variable\n  \
         .env.openai file (OPENAI_API_KEY=sk-...)"
    )
}
