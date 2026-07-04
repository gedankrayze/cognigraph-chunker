//! Adaptive chunking subcommand.

use clap::Args;

use cognigraph_chunker::embeddings::{EmbedProviderOpts, EmbeddingProvider};
use cognigraph_chunker::llm::{CompletionClient, LlmConfig};
use cognigraph_chunker::output::OutputFormat;
use cognigraph_chunker::semantic::adaptive_chunk::{AdaptiveConfig, adaptive_chunk};
use cognigraph_chunker::semantic::adaptive_types::AdaptiveResult;
use cognigraph_chunker::semantic::quality_metrics::MetricWeights;

use super::global_opts::GlobalOpts;

#[derive(Args)]
pub struct AdaptiveArgs {
    /// Input file path, or "-" for stdin (default: stdin)
    #[arg(short, long, default_value = "-")]
    pub input: String,

    #[command(flatten)]
    pub provider_opts: EmbedProviderOpts,

    /// LLM model for intent/enriched/topo methods (default: gpt-4.1-mini)
    #[arg(long)]
    pub llm_model: Option<String>,

    /// LLM base URL override (default: OpenAI)
    #[arg(long)]
    pub llm_base_url: Option<String>,

    /// Comma-separated candidate methods (default: semantic,cognitive; with --api-key: all)
    #[arg(long)]
    pub candidates: Option<String>,

    /// Bypass pre-screening and run all listed candidates
    #[arg(long)]
    pub force_candidates: bool,

    /// Metric weights as key=value pairs (e.g. "sc=0.3,icc=0.2,dcc=0.2,bi=0.15,rc=0.15")
    #[arg(long)]
    pub metric_weights: Option<String>,

    /// Include the full quality report in output
    #[arg(long)]
    pub report: bool,

    /// Soft token budget per chunk
    #[arg(long, default_value_t = 512)]
    pub soft_budget: usize,

    /// Hard token ceiling per chunk
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

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Json)]
    pub format: OutputFormat,
}

pub async fn run(args: &AdaptiveArgs, global: &GlobalOpts) -> anyhow::Result<()> {
    let text = super::read_input(&args.input, global.max_input_size)?;
    let text_str = String::from_utf8_lossy(&text);

    global.detail(&format!(
        "[adaptive] input: {} bytes, provider: {:?}, budget: {}/{}",
        text.len(),
        args.provider_opts.provider,
        args.soft_budget,
        args.hard_budget,
    ));

    // Resolve candidates
    let has_api_key = args.provider_opts.api_key.is_some()
        || std::env::var("OPENAI_API_KEY").is_ok()
        || std::fs::read_to_string(".env.openai").is_ok();

    let candidates: Vec<String> = if let Some(ref c) = args.candidates {
        c.split(',').map(|s| s.trim().to_string()).collect()
    } else if has_api_key {
        vec![
            "semantic".to_string(),
            "cognitive".to_string(),
            "intent".to_string(),
            "enriched".to_string(),
            "topo".to_string(),
        ]
    } else {
        vec!["semantic".to_string(), "cognitive".to_string()]
    };

    // Parse metric weights
    let metric_weights = if let Some(ref w) = args.metric_weights {
        parse_metric_weights(w)?
    } else {
        MetricWeights::default()
    };

    let config = AdaptiveConfig {
        candidates,
        force_candidates: args.force_candidates,
        soft_budget: args.soft_budget,
        hard_budget: args.hard_budget,
        metric_weights,
        sim_window: args.sim_window,
        sg_window: args.sg_window,
        poly_order: args.poly_order,
    };

    // Build optional LLM client
    let llm_client = if has_api_key {
        match LlmConfig::resolve(
            &args.provider_opts.api_key,
            &args.llm_base_url,
            &args.llm_model,
        ) {
            Ok(llm_config) => match CompletionClient::new(llm_config) {
                Ok(client) => Some(client),
                Err(e) => {
                    global.detail(&format!("[adaptive] LLM client init failed: {e}"));
                    None
                }
            },
            Err(e) => {
                global.detail(&format!("[adaptive] LLM config resolve failed: {e}"));
                None
            }
        }
    } else {
        None
    };

    let provider = args.provider_opts.build_provider().await?;
    run_adaptive(
        &text_str,
        &provider,
        llm_client.as_ref(),
        &config,
        args,
        global,
    )
    .await
}

async fn run_adaptive<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    llm_client: Option<&CompletionClient>,
    config: &AdaptiveConfig,
    args: &AdaptiveArgs,
    global: &GlobalOpts,
) -> anyhow::Result<()> {
    let result = adaptive_chunk(text, provider, llm_client, config).await?;

    global.info(&format!(
        "[adaptive] winner: {} ({} chunks, composite: {:.4})",
        result.winner,
        result.count,
        result
            .report
            .candidates
            .first()
            .map(|c| c.metrics.composite)
            .unwrap_or(0.0),
    ));

    write_output(&result, args);
    Ok(())
}

fn write_output(result: &AdaptiveResult, args: &AdaptiveArgs) {
    match args.format {
        OutputFormat::Plain => {
            println!("Winner: {}", result.winner);
            println!("Chunks: {}", result.count);
            if args.report {
                println!("\n--- Quality Report ---");
                for c in &result.report.candidates {
                    println!(
                        "  {}: composite={:.4} sc={:.4} icc={:.4} dcc={:.4} bi={:.4} rc={:.4} ({})",
                        c.method,
                        c.metrics.composite,
                        c.metrics.size_compliance,
                        c.metrics.intrachunk_cohesion,
                        c.metrics.contextual_coherence,
                        c.metrics.block_integrity,
                        c.metrics.reference_completeness,
                        c.chunk_count,
                    );
                }
                println!("\n--- Pre-screening ---");
                for d in &result.report.pre_screening {
                    let status = if d.included { "OK" } else { "SKIP" };
                    println!("  [{}] {}: {}", status, d.method, d.reason);
                }
            }
            println!();
            for chunk in &result.chunks {
                if let Some(text) = chunk.get("text").and_then(|v| v.as_str()) {
                    println!("{}", text);
                    println!();
                }
            }
        }
        OutputFormat::Json => {
            if args.report {
                println!("{}", serde_json::to_string_pretty(&result).unwrap());
            } else {
                let output = serde_json::json!({
                    "winner": result.winner,
                    "chunks": result.chunks,
                    "count": result.count,
                });
                println!("{}", serde_json::to_string_pretty(&output).unwrap());
            }
        }
        OutputFormat::Jsonl => {
            for chunk in &result.chunks {
                println!("{}", serde_json::to_string(chunk).unwrap());
            }
        }
    }
}

fn parse_metric_weights(input: &str) -> anyhow::Result<MetricWeights> {
    let mut weights = MetricWeights::default();
    for pair in input.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let (key, val) = pair
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("Invalid weight pair: {pair}"))?;
        let val: f64 = val
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid weight value: {val}"))?;
        match key.trim() {
            "sc" | "size_compliance" => weights.sc = val,
            "icc" | "intrachunk_cohesion" => weights.icc = val,
            "dcc" | "contextual_coherence" => weights.dcc = val,
            "bi" | "block_integrity" => weights.bi = val,
            "rc" | "reference_completeness" => weights.rc = val,
            other => anyhow::bail!("Unknown metric weight key: {other}"),
        }
    }
    Ok(weights)
}

