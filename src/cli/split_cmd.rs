//! Delimiter splitting subcommand.

use std::io::{self, Read};
use std::path::PathBuf;

use clap::Args;

use cognigraph_chunker::core::split::{IncludeDelim, split_at_delimiters, split_at_patterns};
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
    let text = read_input(&args.input, global.max_input_size)?;
    let include_delim: IncludeDelim = args.include_delim.into();

    global.detail(&format!("[split] input: {} bytes", text.len()));

    let offsets = if let Some(ref patterns_str) = args.patterns {
        let pattern_strings: Vec<String> = parse_patterns(patterns_str);
        let pattern_refs: Vec<&[u8]> = pattern_strings.iter().map(|s| s.as_bytes()).collect();
        global.detail(&format!("[split] using {} multi-byte patterns", pattern_refs.len()));
        split_at_patterns(&text, &pattern_refs, include_delim, args.min_chars)
    } else {
        let delim_bytes = if let Some(ref d) = args.delimiters {
            parse_delimiters(d)
        } else {
            b"\n.?".to_vec()
        };
        global.detail(&format!("[split] using {} single-byte delimiters", delim_bytes.len()));
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

fn read_input(input: &str, max_size: usize) -> anyhow::Result<Vec<u8>> {
    if input == "-" {
        let mut buf = Vec::new();
        io::stdin().take(max_size as u64 + 1).read_to_end(&mut buf)?;
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

/// Parse comma-separated patterns, interpreting escape sequences.
pub(crate) fn parse_patterns(s: &str) -> Vec<String> {
    s.split(',')
        .map(|p| {
            let bytes = parse_delimiters(p);
            String::from_utf8_lossy(&bytes).into_owned()
        })
        .collect()
}

/// Parse delimiter string, interpreting escape sequences like \n, \t.
fn parse_delimiters(s: &str) -> Vec<u8> {
    let mut result = Vec::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push(b'\n'),
                Some('t') => result.push(b'\t'),
                Some('r') => result.push(b'\r'),
                Some('\\') => result.push(b'\\'),
                Some(other) => {
                    result.push(b'\\');
                    let mut buf = [0u8; 4];
                    result.extend_from_slice(other.encode_utf8(&mut buf).as_bytes());
                }
                None => result.push(b'\\'),
            }
        } else {
            let mut buf = [0u8; 4];
            result.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
    }
    result
}
