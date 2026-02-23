//! Output formatting for chunk results.

use serde::Serialize;

/// Format for CLI output.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    /// One chunk per line, separated by blank lines
    Plain,
    /// JSON array of all chunks
    Json,
    /// One JSON object per line
    Jsonl,
}

#[derive(Serialize)]
struct ChunkEntry<'a> {
    index: usize,
    text: &'a str,
    offset: usize,
    length: usize,
}

/// Write chunks to stdout in the specified format.
pub fn write_chunks(chunks: &[(String, usize)], format: OutputFormat) {
    match format {
        OutputFormat::Plain => {
            for (i, (text, _offset)) in chunks.iter().enumerate() {
                if i > 0 {
                    println!();
                }
                print!("{text}");
            }
            // Ensure trailing newline
            println!();
        }
        OutputFormat::Json => {
            let entries: Vec<ChunkEntry> = chunks
                .iter()
                .enumerate()
                .map(|(i, (text, offset))| ChunkEntry {
                    index: i,
                    text,
                    offset: *offset,
                    length: text.len(),
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&entries).unwrap());
        }
        OutputFormat::Jsonl => {
            for (i, (text, offset)) in chunks.iter().enumerate() {
                let entry = ChunkEntry {
                    index: i,
                    text,
                    offset: *offset,
                    length: text.len(),
                };
                println!("{}", serde_json::to_string(&entry).unwrap());
            }
        }
    }
}
