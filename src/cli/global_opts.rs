//! Global CLI options shared across all subcommands.

use clap::Args;

/// Default maximum input size: 50 MiB.
const DEFAULT_MAX_INPUT_SIZE: usize = 50 * 1024 * 1024;

/// Global output control flags.
#[derive(Args, Clone)]
pub struct GlobalOpts {
    /// Show detailed processing information
    #[arg(long, global = true)]
    pub verbose: bool,

    /// Suppress all informational output (only errors and chunk data)
    #[arg(long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,

    /// Print chunk statistics after output (count, avg/min/max size)
    #[arg(long, global = true)]
    pub stats: bool,

    /// Maximum input size in bytes (default: 50 MiB)
    #[arg(long, global = true, default_value_t = DEFAULT_MAX_INPUT_SIZE)]
    pub max_input_size: usize,
}

impl GlobalOpts {
    /// Print a message to stderr unless --quiet is set.
    pub fn info(&self, msg: &str) {
        if !self.quiet {
            eprintln!("{msg}");
        }
    }

    /// Print a message to stderr only if --verbose is set.
    pub fn detail(&self, msg: &str) {
        if self.verbose {
            eprintln!("{msg}");
        }
    }
}

/// Print chunk statistics to stderr.
pub fn print_stats(chunks: &[(String, usize)], opts: &GlobalOpts) {
    if !opts.stats {
        return;
    }

    let count = chunks.len();
    if count == 0 {
        eprintln!("[stats] 0 chunks");
        return;
    }

    let sizes_bytes: Vec<usize> = chunks.iter().map(|(t, _)| t.len()).collect();
    let sizes_tokens: Vec<usize> = chunks.iter().map(|(t, _)| t.split_whitespace().count()).collect();

    let total_bytes: usize = sizes_bytes.iter().sum();
    let total_tokens: usize = sizes_tokens.iter().sum();
    let min_bytes = sizes_bytes.iter().min().unwrap();
    let max_bytes = sizes_bytes.iter().max().unwrap();
    let avg_bytes = total_bytes / count;
    let min_tokens = sizes_tokens.iter().min().unwrap();
    let max_tokens = sizes_tokens.iter().max().unwrap();
    let avg_tokens = total_tokens / count;

    eprintln!("[stats] {} chunks | {} bytes total | {} tokens total", count, total_bytes, total_tokens);
    eprintln!("[stats] bytes  — avg: {}, min: {}, max: {}", avg_bytes, min_bytes, max_bytes);
    eprintln!("[stats] tokens — avg: {}, min: {}, max: {}", avg_tokens, min_tokens, max_tokens);
}
