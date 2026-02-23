//! Size-based text chunking at delimiter boundaries.
//!
//! This module provides the [`Chunker`] and [`OwnedChunker`] types for splitting
//! text into chunks of a target size, preferring to break at delimiter boundaries.

use super::delim::{DEFAULT_DELIMITERS, DEFAULT_TARGET_SIZE, build_table, compute_split_at};

/// Chunk text at delimiter boundaries.
///
/// Returns a builder that can be configured with `.size()` and `.delimiters()`,
/// or used directly as an iterator with defaults (4KB chunks, `\n.?` delimiters).
///
/// - For 1-3 delimiters: uses SIMD-accelerated memchr
/// - For 4+ delimiters: uses lookup table
pub fn chunk(text: &[u8]) -> Chunker<'_> {
    Chunker::new(text)
}

/// Chunker splits text at delimiter boundaries.
///
/// Created via [`chunk()`], can be configured with `.size()` and `.delimiters()`.
/// For multi-byte delimiters, use `.pattern()` instead.
pub struct Chunker<'a> {
    text: &'a [u8],
    target_size: usize,
    delimiters: &'a [u8],
    pattern: Option<&'a [u8]>,
    pos: usize,
    table: Option<[bool; 256]>,
    initialized: bool,
    prefix_mode: bool,
    /// When true, find the START of consecutive pattern runs (not middle)
    consecutive: bool,
    /// When true, search forward if no pattern found in backward window
    forward_fallback: bool,
}

impl<'a> Chunker<'a> {
    fn new(text: &'a [u8]) -> Self {
        Self {
            text,
            target_size: DEFAULT_TARGET_SIZE,
            delimiters: DEFAULT_DELIMITERS,
            pattern: None,
            pos: 0,
            table: None,
            initialized: false,
            prefix_mode: false,
            consecutive: false,
            forward_fallback: false,
        }
    }

    /// Set the target chunk size in bytes.
    pub fn size(mut self, size: usize) -> Self {
        self.target_size = size.max(1);
        self
    }

    /// Set single-byte delimiters to split on.
    ///
    /// Mutually exclusive with `pattern()` - last one set wins.
    pub fn delimiters(mut self, delimiters: &'a [u8]) -> Self {
        self.delimiters = delimiters;
        self.pattern = None;
        self
    }

    /// Set a multi-byte pattern to split on.
    ///
    /// Use this for multi-byte delimiters like UTF-8 characters (e.g., metaspace `▁`).
    /// Mutually exclusive with `delimiters()` - last one set wins.
    pub fn pattern(mut self, pattern: &'a [u8]) -> Self {
        self.pattern = Some(pattern);
        self.delimiters = &[];
        self
    }

    /// Put delimiter at the start of the next chunk (prefix mode).
    pub fn prefix(mut self) -> Self {
        self.prefix_mode = true;
        self
    }

    /// Put delimiter at the end of the current chunk (suffix mode, default).
    pub fn suffix(mut self) -> Self {
        self.prefix_mode = false;
        self
    }

    /// Enable consecutive delimiter/pattern handling.
    ///
    /// When splitting, ensures we split at the START of a consecutive run
    /// of the same delimiter/pattern, not in the middle.
    pub fn consecutive(mut self) -> Self {
        self.consecutive = true;
        self
    }

    /// Enable forward fallback search.
    ///
    /// When no delimiter/pattern is found in the backward search window,
    /// search forward from target_end instead of doing a hard split.
    pub fn forward_fallback(mut self) -> Self {
        self.forward_fallback = true;
        self
    }

    /// Initialize lookup table if needed (called on first iteration).
    fn init(&mut self) {
        if !self.initialized {
            self.table = build_table(self.delimiters);
            self.initialized = true;
        }
    }
}

impl<'a> Iterator for Chunker<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        self.init();

        if self.pos >= self.text.len() {
            return None;
        }

        let remaining = self.text.len() - self.pos;

        // Last chunk - return remainder
        if remaining <= self.target_size {
            let chunk = &self.text[self.pos..];
            self.pos = self.text.len();
            return Some(chunk);
        }

        let end = self.pos.saturating_add(self.target_size).min(self.text.len());

        let split_at = compute_split_at(
            self.text,
            self.pos,
            end,
            self.pattern,
            self.delimiters,
            self.table.as_ref(),
            self.prefix_mode,
            self.consecutive,
            self.forward_fallback,
        );

        let chunk = &self.text[self.pos..split_at];
        self.pos = split_at;
        Some(chunk)
    }
}

/// Owned chunker for FFI bindings (Python, WASM).
///
/// Unlike [`Chunker`], this owns its data and returns owned chunks.
/// Use this when you need to cross FFI boundaries where lifetimes can't be tracked.
pub struct OwnedChunker {
    text: Vec<u8>,
    target_size: usize,
    delimiters: Vec<u8>,
    pattern: Option<Vec<u8>>,
    pos: usize,
    table: Option<[bool; 256]>,
    initialized: bool,
    prefix_mode: bool,
    consecutive: bool,
    forward_fallback: bool,
}

impl OwnedChunker {
    /// Create a new owned chunker with the given text.
    pub fn new(text: Vec<u8>) -> Self {
        Self {
            text,
            target_size: DEFAULT_TARGET_SIZE,
            delimiters: DEFAULT_DELIMITERS.to_vec(),
            pattern: None,
            pos: 0,
            table: None,
            initialized: false,
            prefix_mode: false,
            consecutive: false,
            forward_fallback: false,
        }
    }

    /// Set the target chunk size in bytes.
    pub fn size(mut self, size: usize) -> Self {
        self.target_size = size.max(1);
        self
    }

    /// Set single-byte delimiters to split on.
    pub fn delimiters(mut self, delimiters: Vec<u8>) -> Self {
        self.delimiters = delimiters;
        self.pattern = None;
        self
    }

    /// Set a multi-byte pattern to split on.
    pub fn pattern(mut self, pattern: Vec<u8>) -> Self {
        self.pattern = Some(pattern);
        self.delimiters = vec![];
        self
    }

    /// Put delimiter at the start of the next chunk (prefix mode).
    pub fn prefix(mut self) -> Self {
        self.prefix_mode = true;
        self
    }

    /// Put delimiter at the end of the current chunk (suffix mode, default).
    pub fn suffix(mut self) -> Self {
        self.prefix_mode = false;
        self
    }

    /// Enable consecutive delimiter/pattern handling.
    pub fn consecutive(mut self) -> Self {
        self.consecutive = true;
        self
    }

    /// Enable forward fallback search.
    pub fn forward_fallback(mut self) -> Self {
        self.forward_fallback = true;
        self
    }

    /// Initialize lookup table if needed.
    fn init(&mut self) {
        if !self.initialized {
            self.table = build_table(&self.delimiters);
            self.initialized = true;
        }
    }

    /// Get the next chunk, or None if exhausted.
    pub fn next_chunk(&mut self) -> Option<Vec<u8>> {
        self.init();

        if self.pos >= self.text.len() {
            return None;
        }

        let remaining = self.text.len() - self.pos;

        // Last chunk - return remainder
        if remaining <= self.target_size {
            let chunk = self.text[self.pos..].to_vec();
            self.pos = self.text.len();
            return Some(chunk);
        }

        let end = self.pos.saturating_add(self.target_size).min(self.text.len());

        let split_at = compute_split_at(
            &self.text,
            self.pos,
            end,
            self.pattern.as_deref(),
            &self.delimiters,
            self.table.as_ref(),
            self.prefix_mode,
            self.consecutive,
            self.forward_fallback,
        );

        let chunk = self.text[self.pos..split_at].to_vec();
        self.pos = split_at;
        Some(chunk)
    }

    /// Reset the chunker to start from the beginning.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Get a reference to the underlying text.
    pub fn text(&self) -> &[u8] {
        &self.text
    }

    /// Collect all chunk offsets as (start, end) pairs.
    pub fn collect_offsets(&mut self) -> Vec<(usize, usize)> {
        self.init();

        let mut offsets = Vec::new();
        let mut pos = 0;

        while pos < self.text.len() {
            let remaining = self.text.len() - pos;

            if remaining <= self.target_size {
                offsets.push((pos, self.text.len()));
                break;
            }

            let end = pos.saturating_add(self.target_size).min(self.text.len());

            let split_at = compute_split_at(
                &self.text,
                pos,
                end,
                self.pattern.as_deref(),
                &self.delimiters,
                self.table.as_ref(),
                self.prefix_mode,
                self.consecutive,
                self.forward_fallback,
            );

            offsets.push((pos, split_at));
            pos = split_at;
        }

        offsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let text = b"Hello. World. Test.";
        let chunks: Vec<_> = chunk(text).size(10).delimiters(b".").collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], b"Hello.");
        assert_eq!(chunks[1], b" World.");
        assert_eq!(chunks[2], b" Test.");
    }

    #[test]
    fn test_newline_delimiter() {
        let text = b"Line one\nLine two\nLine three";
        let chunks: Vec<_> = chunk(text).size(15).delimiters(b"\n").collect();
        assert_eq!(chunks[0], b"Line one\n");
        assert_eq!(chunks[1], b"Line two\n");
        assert_eq!(chunks[2], b"Line three");
    }

    #[test]
    fn test_no_delimiter_hard_split() {
        let text = b"abcdefghij";
        let chunks: Vec<_> = chunk(text).size(5).delimiters(b".").collect();
        assert_eq!(chunks[0], b"abcde");
        assert_eq!(chunks[1], b"fghij");
    }

    #[test]
    fn test_empty_text() {
        let text = b"";
        let chunks: Vec<_> = chunk(text).size(10).delimiters(b".").collect();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_text_smaller_than_target() {
        let text = b"Small";
        let chunks: Vec<_> = chunk(text).size(100).delimiters(b".").collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], b"Small");
    }

    #[test]
    fn test_total_bytes_preserved() {
        let text = b"The quick brown fox jumps over the lazy dog. How vexingly quick!";
        let chunks: Vec<_> = chunk(text).size(20).delimiters(b"\n.?!").collect();
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, text.len());
    }

    #[test]
    fn test_prefix_mode() {
        let text = b"Hello World Test";
        let chunks: Vec<_> = chunk(text).size(8).delimiters(b" ").prefix().collect();
        assert_eq!(chunks[0], b"Hello");
        assert_eq!(chunks[1], b" World");
        assert_eq!(chunks[2], b" Test");
    }

    #[test]
    fn test_suffix_mode() {
        let text = b"Hello World Test";
        let chunks: Vec<_> = chunk(text).size(8).delimiters(b" ").suffix().collect();
        assert_eq!(chunks[0], b"Hello ");
        assert_eq!(chunks[1], b"World ");
        assert_eq!(chunks[2], b"Test");
    }

    #[test]
    fn test_consecutive_delimiters() {
        let text = b"Hello\n\n\nWorld";
        let chunks: Vec<_> = chunk(text)
            .delimiters(b"\n")
            .size(8)
            .prefix()
            .consecutive()
            .collect();
        assert_eq!(chunks[0], b"Hello");
        assert_eq!(chunks[1], b"\n\n\nWorld");
    }

    #[test]
    fn test_forward_fallback() {
        let text = b"verylongword next";
        let chunks: Vec<_> = chunk(text)
            .delimiters(b" ")
            .size(6)
            .prefix()
            .forward_fallback()
            .collect();
        assert_eq!(chunks[0], b"verylongword");
        assert_eq!(chunks[1], b" next");
    }

    #[test]
    fn test_pattern_metaspace() {
        let metaspace = "▁".as_bytes();
        let text = "Hello▁World▁Test".as_bytes();
        let chunks: Vec<_> = chunk(text).size(15).pattern(metaspace).prefix().collect();
        assert_eq!(chunks[0], "Hello".as_bytes());
        assert_eq!(chunks[1], "▁World▁Test".as_bytes());
    }

    #[test]
    fn test_owned_chunker() {
        let text = b"Hello. World. Test.".to_vec();
        let mut chunker = OwnedChunker::new(text).size(10).delimiters(b".".to_vec());

        let mut chunks = Vec::new();
        while let Some(c) = chunker.next_chunk() {
            chunks.push(c);
        }

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], b"Hello.");
    }

    #[test]
    fn test_owned_chunker_collect_offsets() {
        let text = b"Hello. World. Test.".to_vec();
        let mut chunker = OwnedChunker::new(text.clone())
            .size(10)
            .delimiters(b".".to_vec());

        let offsets = chunker.collect_offsets();
        assert_eq!(offsets.len(), 3);
        assert_eq!(&text[offsets[0].0..offsets[0].1], b"Hello.");
    }
}
