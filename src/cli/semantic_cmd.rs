//! Semantic chunking subcommand.

use clap::Args;

use cognigraph_chunker::embeddings::{EmbedProviderOpts, EmbeddingProvider};
use cognigraph_chunker::output::{OutputFormat, write_chunks};
use cognigraph_chunker::semantic::{SemanticConfig, semantic_chunk, semantic_chunk_plain};

use super::global_opts::{self, GlobalOpts};
use super::merge_opts::{MergeOpts, maybe_merge};

#[derive(Args)]
pub struct SemanticArgs {
    /// Input file path, or "-" for stdin (default: stdin)
    #[arg(short, long, default_value = "-")]
    pub input: String,

    #[command(flatten)]
    pub provider_opts: EmbedProviderOpts,

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
    let text = super::read_input(&args.input, global.max_input_size)?;
    let text_str = String::from_utf8_lossy(&text);

    global.detail(&format!(
        "[semantic] input: {} bytes, provider: {:?}, markdown: {}",
        text.len(),
        args.provider_opts.provider,
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

    let provider = args.provider_opts.build_provider().await?;
    run_pipeline(&text_str, &provider, &config, args, global).await
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
