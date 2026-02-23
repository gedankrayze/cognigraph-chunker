//! Shared merge options and logic for all subcommands.

use clap::Args;

use cognigraph_chunker::core::merge::merge_splits;

use super::global_opts::GlobalOpts;

/// Merge options that can be added to any subcommand.
#[derive(Args)]
pub struct MergeOpts {
    /// Post-process chunks by merging small ones to fit a token budget
    #[arg(long)]
    pub merge: bool,

    /// Target token count per merged chunk (used with --merge)
    #[arg(long, default_value_t = 512)]
    pub chunk_size: usize,
}

/// Count tokens using whitespace splitting (fast approximation).
fn count_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Apply merge post-processing to chunks if --merge is enabled.
///
/// Takes chunks as (text, offset) pairs and returns merged chunks.
pub fn maybe_merge(
    chunks: Vec<(String, usize)>,
    opts: &MergeOpts,
    global: &GlobalOpts,
) -> Vec<(String, usize)> {
    if !opts.merge || chunks.len() <= 1 {
        return chunks;
    }

    let texts: Vec<&str> = chunks.iter().map(|(t, _)| t.as_str()).collect();
    let token_counts: Vec<usize> = texts.iter().map(|t| count_tokens(t)).collect();

    let result = merge_splits(&texts, &token_counts, opts.chunk_size);

    // Reconstruct (text, offset) pairs for merged chunks.
    let mut merged_chunks = Vec::with_capacity(result.merged.len());
    let mut orig_idx = 0;

    for merged_text in &result.merged {
        let offset = chunks[orig_idx].1;
        merged_chunks.push((merged_text.clone(), offset));

        // Advance past the original chunks that were merged into this one
        let mut consumed_len = 0;
        while orig_idx < chunks.len() && consumed_len < merged_text.len() {
            consumed_len += chunks[orig_idx].0.len();
            orig_idx += 1;
        }
    }

    global.info(&format!(
        "[merge] {} chunks → {} merged (target: {} tokens)",
        chunks.len(),
        merged_chunks.len(),
        opts.chunk_size,
    ));

    merged_chunks
}
