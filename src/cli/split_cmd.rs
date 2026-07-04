//! Delimiter splitting subcommand.

use clap::Args;

use cognigraph_chunker::core::split::{IncludeDelim, split_at_delimiters, split_at_patterns};
use cognigraph_chunker::core::{parse_delimiters, parse_patterns};
use cognigraph_chunker::output::{OutputFormat, write_chunks};

use super::global_opts::{self, GlobalOpts};
use super::merge_opts::{MergeOpts, maybe_merge};

#[derive(Args)]
pub struct SplitArgs {
    /// Input file path, or "-" for stdin (default: stdin)
    #[arg(short, long, default_value = "-")]
    pub input: String,

    /// Single-byte delimiters to split on (e.g., ".?!")
    #[arg(short, long)]
    pub delimiters: Option<String>,

    /// Multi-byte patterns to split on, comma-separated (e.g., ". ,? ,! ")
    #[arg(short, long)]
    pub patterns: Option<String>,

    /// Where to attach the delimiter
    #[arg(long, value_enum, default_value_t = IncludeDelimArg::Prev)]
    pub include_delim: IncludeDelimArg,

    /// Minimum characters per segment; shorter segments are merged
    #[arg(long, default_value_t = 0)]
    pub min_chars: usize,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Plain)]
    pub format: OutputFormat,

    #[command(flatten)]
    pub merge_opts: MergeOpts,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum IncludeDelimArg {
    /// Attach delimiter to previous segment (e.g., "Hello." | " World.")
    Prev,
    /// Attach delimiter to next segment (e.g., "Hello" | ". World")
    Next,
    /// Don't include delimiter in either segment
    None,
}

impl From<IncludeDelimArg> for IncludeDelim {
    fn from(arg: IncludeDelimArg) -> Self {
        match arg {
            IncludeDelimArg::Prev => IncludeDelim::Prev,
            IncludeDelimArg::Next => IncludeDelim::Next,
            IncludeDelimArg::None => IncludeDelim::None,
        }
    }
}

pub fn run(args: &SplitArgs, global: &GlobalOpts) -> anyhow::Result<()> {
    let text = super::read_input(&args.input, global.max_input_size)?;
    let include_delim: IncludeDelim = args.include_delim.into();

    global.detail(&format!("[split] input: {} bytes", text.len()));

    let offsets = if let Some(ref patterns_str) = args.patterns {
        let pattern_strings: Vec<String> = parse_patterns(patterns_str);
        let pattern_refs: Vec<&[u8]> = pattern_strings.iter().map(|s| s.as_bytes()).collect();
        global.detail(&format!(
            "[split] using {} multi-byte patterns",
            pattern_refs.len()
        ));
        split_at_patterns(&text, &pattern_refs, include_delim, args.min_chars)
    } else {
        let delim_bytes = if let Some(ref d) = args.delimiters {
            parse_delimiters(d)
        } else {
            b"\n.?".to_vec()
        };
        global.detail(&format!(
            "[split] using {} single-byte delimiters",
            delim_bytes.len()
        ));
        split_at_delimiters(&text, &delim_bytes, include_delim, args.min_chars)
    };

    let results: Vec<(String, usize)> = offsets
        .iter()
        .map(|&(start, end)| {
            let segment = String::from_utf8_lossy(&text[start..end]).into_owned();
            (segment, start)
        })
        .collect();

    let results = maybe_merge(results, &args.merge_opts, global);
    write_chunks(&results, args.format);
    global_opts::print_stats(&results, global);
    Ok(())
}

