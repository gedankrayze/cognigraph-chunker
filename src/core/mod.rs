//! Core chunking, splitting, merging, and signal processing algorithms.
//!
//! Ported from the `chunk` research library as first-party code.

pub mod chunk;
pub mod delim;
pub mod merge;
pub mod savgol;
pub mod split;

// Re-export primary types and functions
pub use chunk::{Chunker, OwnedChunker, chunk};
pub use delim::{DEFAULT_DELIMITERS, DEFAULT_TARGET_SIZE};
pub use merge::{MergeResult, find_merge_indices, merge_splits};
pub use savgol::{
    FilteredIndices, MinimaResult, filter_split_indices, find_local_minima_interpolated,
    savgol_filter, windowed_cross_similarity,
};
pub use split::{
    IncludeDelim, PatternSplitter, Splitter, split, split_at_delimiters, split_at_patterns,
};
