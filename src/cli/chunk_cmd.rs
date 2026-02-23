//! Fixed-size chunking subcommand.

use std::io::{self, Read};
use std::path::PathBuf;

use clap::Args;

use cognigraph_chunker::core::chunk::chunk;
use cognigraph_chunker::output::{OutputFormat, write_chunks};

use super::global_opts::{self, GlobalOpts};
use super::merge_opts::{MergeOpts, maybe_merge};

#[derive(Args)]
pub struct ChunkArgs {
    /// Input file path, or "-" for stdin (default: stdin)
    #[arg(short, long, default_value = "-")]
    pub input: String,

    /// Target chunk size in bytes
    #[arg(short, long, default_value_t = 4096)]
    pub size: usize,

    /// Single-byte delimiters to split on (e.g., "\n.?")
    #[arg(short, long)]
    pub delimiters: Option<String>,

    /// Multi-byte pattern to split on (e.g., "\u{2581}")
    #[arg(short, long)]
    pub pattern: Option<String>,

    /// Put delimiter at start of next chunk instead of end of current
    #[arg(long)]
    pub prefix: bool,

    /// Split at start of consecutive delimiter runs
    #[arg(long)]
    pub consecutive: bool,

    /// Search forward if no boundary found in backward window
    #[arg(long)]
    pub forward_fallback: bool,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Plain)]
    pub format: OutputFormat,

    #[command(flatten)]
    pub merge_opts: MergeOpts,
}

pub fn run(args: &ChunkArgs, global: &GlobalOpts) -> anyhow::Result<()> {
    let text = read_input(&args.input)?;

    global.detail(&format!(
        "[chunk] input: {} bytes, target size: {} bytes",
        text.len(),
        args.size
    ));

    let mut chunker = chunk(&text).size(args.size);

    // Store owned values so borrows live long enough
    let delim_bytes;
    let pattern_bytes;

    if let Some(ref pat) = args.pattern {
        pattern_bytes = pat.as_bytes().to_vec();
        chunker = chunker.pattern(&pattern_bytes);
    } else if let Some(ref delims) = args.delimiters {
        delim_bytes = parse_delimiters(delims);
        chunker = chunker.delimiters(&delim_bytes);
    }

    if args.prefix {
        chunker = chunker.prefix();
    }
    if args.consecutive {
        chunker = chunker.consecutive();
    }
    if args.forward_fallback {
        chunker = chunker.forward_fallback();
    }

    let mut offset = 0;
    let mut results: Vec<(String, usize)> = Vec::new();
    for chunk_bytes in chunker {
        let text = String::from_utf8_lossy(chunk_bytes).into_owned();
        results.push((text, offset));
        offset += chunk_bytes.len();
    }

    let results = maybe_merge(results, &args.merge_opts, global);
    write_chunks(&results, args.format);
    global_opts::print_stats(&results, global);
    Ok(())
}

fn read_input(input: &str) -> anyhow::Result<Vec<u8>> {
    if input == "-" {
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        Ok(buf)
    } else {
        let path = PathBuf::from(input);
        anyhow::ensure!(
            path.exists(),
            "File not found: {}\nCheck the path and try again.",
            path.display()
        );
        Ok(std::fs::read(&path)?)
    }
}

/// Parse delimiter string, interpreting escape sequences like \n, \t.
pub(crate) fn parse_delimiters(s: &str) -> Vec<u8> {
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
