//! CLI subcommands.

use std::io::Read;
use std::path::PathBuf;

pub mod adaptive_cmd;
pub mod chunk_cmd;
pub mod cognitive_cmd;
pub mod enriched_cmd;
pub mod global_opts;
pub mod intent_cmd;
pub mod merge_opts;
pub mod semantic_cmd;
pub mod serve_cmd;
pub mod split_cmd;
pub mod topo_cmd;

/// Read input from a file path or stdin (`"-"`), enforcing a maximum size.
pub(crate) fn read_input(input: &str, max_size: usize) -> anyhow::Result<Vec<u8>> {
    if input == "-" {
        let mut buf = Vec::new();
        std::io::stdin()
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
