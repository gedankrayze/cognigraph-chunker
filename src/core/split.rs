//! Delimiter-based text splitting.
//!
//! This module provides functions to split text at every delimiter occurrence.
//! Unlike the [`chunk`](super::chunk) module which creates size-based chunks,
//! this splits at **every** delimiter.

use super::delim::{DEFAULT_DELIMITERS, build_table, find_first_delimiter};
use daggrs::{DoubleArrayAhoCorasick, MatchKind, Trie};

/// Where to include the delimiter in splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IncludeDelim {
    /// Attach delimiter to the previous segment (e.g., "Hello." | " World.")
    #[default]
    Prev,
    /// Attach delimiter to the next segment (e.g., "Hello" | ". World")
    Next,
    /// Don't include delimiter in either segment
    None,
}

/// Split text at every delimiter occurrence, returning offsets.
pub fn split_at_delimiters(
    text: &[u8],
    delimiters: &[u8],
    include_delim: IncludeDelim,
    min_chars: usize,
) -> Vec<(usize, usize)> {
    if text.is_empty() {
        return vec![];
    }

    if delimiters.is_empty() {
        return vec![(0, text.len())];
    }

    let table = build_table(delimiters);
    let estimated_segments = (text.len() / 40).max(4);
    let mut splits: Vec<(usize, usize)> = Vec::with_capacity(estimated_segments);

    let mut segment_start = 0;
    let mut pos = 0;
    let mut accum_start: usize = 0;
    let mut accum_end: usize = 0;

    macro_rules! emit_segment {
        ($seg_start:expr, $seg_end:expr) => {
            let seg_start = $seg_start;
            let seg_end = $seg_end;

            if min_chars == 0 {
                splits.push((seg_start, seg_end));
            } else if accum_start == accum_end {
                accum_start = seg_start;
                accum_end = seg_end;
            } else {
                let accum_len = accum_end - accum_start;
                let seg_len = seg_end - seg_start;

                if accum_len < min_chars || seg_len < min_chars {
                    accum_end = seg_end;
                } else {
                    splits.push((accum_start, accum_end));
                    accum_start = seg_start;
                    accum_end = seg_end;
                }
            }
        };
    }

    while pos < text.len() {
        let delim_pos = find_first_delimiter(&text[pos..], delimiters, table.as_ref());

        match delim_pos {
            Some(rel_pos) => {
                let abs_pos = pos + rel_pos;

                match include_delim {
                    IncludeDelim::Prev => {
                        let seg_end = abs_pos + 1;
                        if segment_start < seg_end {
                            emit_segment!(segment_start, seg_end);
                        }
                        segment_start = seg_end;
                    }
                    IncludeDelim::Next => {
                        if segment_start < abs_pos {
                            emit_segment!(segment_start, abs_pos);
                        }
                        segment_start = abs_pos;
                    }
                    IncludeDelim::None => {
                        if segment_start < abs_pos {
                            emit_segment!(segment_start, abs_pos);
                        }
                        segment_start = abs_pos + 1;
                    }
                }
                pos = abs_pos + 1;
            }
            None => {
                if segment_start < text.len() {
                    emit_segment!(segment_start, text.len());
                }
                break;
            }
        }
    }

    // Handle trailing content after last delimiter
    if segment_start < text.len()
        && (splits.is_empty() || splits.last().is_none_or(|&(_, e)| e < text.len()))
        && (min_chars == 0 || accum_end < text.len())
    {
        emit_segment!(segment_start, text.len());
    }

    // Emit final accumulated segment (for min_chars mode)
    if min_chars > 0 && accum_start < accum_end {
        splits.push((accum_start, accum_end));
    }

    splits
}

/// Builder for delimiter-based splitting with more options.
///
/// Created via [`split()`], can be configured with various options.
pub fn split(text: &[u8]) -> Splitter<'_> {
    Splitter::new(text)
}

/// Splitter splits text at every delimiter occurrence.
pub struct Splitter<'a> {
    text: &'a [u8],
    delimiters: &'a [u8],
    include_delim: IncludeDelim,
    min_chars: usize,
}

impl<'a> Splitter<'a> {
    fn new(text: &'a [u8]) -> Self {
        Self {
            text,
            delimiters: DEFAULT_DELIMITERS,
            include_delim: IncludeDelim::Prev,
            min_chars: 0,
        }
    }

    /// Set delimiters to split on.
    pub fn delimiters(mut self, delimiters: &'a [u8]) -> Self {
        self.delimiters = delimiters;
        self
    }

    /// Include delimiter with previous segment (default).
    pub fn include_prev(mut self) -> Self {
        self.include_delim = IncludeDelim::Prev;
        self
    }

    /// Include delimiter with next segment.
    pub fn include_next(mut self) -> Self {
        self.include_delim = IncludeDelim::Next;
        self
    }

    /// Don't include delimiter in either segment.
    pub fn include_none(mut self) -> Self {
        self.include_delim = IncludeDelim::None;
        self
    }

    /// Set minimum characters per segment (merges shorter segments).
    pub fn min_chars(mut self, min: usize) -> Self {
        self.min_chars = min;
        self
    }

    /// Collect all split offsets.
    pub fn collect(self) -> Vec<(usize, usize)> {
        split_at_delimiters(
            self.text,
            self.delimiters,
            self.include_delim,
            self.min_chars,
        )
    }

    /// Collect splits as byte slices.
    pub fn collect_slices(self) -> Vec<&'a [u8]> {
        let text = self.text;
        let offsets =
            split_at_delimiters(text, self.delimiters, self.include_delim, self.min_chars);
        offsets
            .into_iter()
            .map(|(start, end)| &text[start..end])
            .collect()
    }
}

/// A compiled multi-pattern splitter for efficient repeated splitting.
///
/// Unlike [`split_at_patterns`] which rebuilds the Aho-Corasick automaton on each call,
/// `PatternSplitter` compiles the automaton once and reuses it (~25x faster).
pub struct PatternSplitter {
    daac: DoubleArrayAhoCorasick,
}

impl PatternSplitter {
    /// Create a new PatternSplitter with the given patterns.
    pub fn new(patterns: &[&[u8]]) -> Self {
        let mut trie = Trie::new();
        for (i, pattern) in patterns.iter().enumerate() {
            trie.add(pattern, i as u32);
        }
        trie.build(MatchKind::LeftmostFirst);
        let daac = trie.compile();
        Self { daac }
    }

    /// Split text using the compiled patterns.
    pub fn split(
        &self,
        text: &[u8],
        include_delim: IncludeDelim,
        min_chars: usize,
    ) -> Vec<(usize, usize)> {
        if text.is_empty() {
            return vec![];
        }

        let mut split_points: Vec<(usize, usize)> = Vec::new();
        for m in self.daac.find_iter(text) {
            split_points.push((m.start, m.end - m.start));
        }

        if split_points.is_empty() {
            return vec![(0, text.len())];
        }

        split_points.sort_by_key(|&(pos, _)| pos);

        let mut splits: Vec<(usize, usize)> = Vec::with_capacity(split_points.len() + 1);
        let mut segment_start = 0;
        let mut accum_start: usize = 0;
        let mut accum_end: usize = 0;

        macro_rules! emit_segment {
            ($seg_start:expr, $seg_end:expr) => {
                let seg_start = $seg_start;
                let seg_end = $seg_end;

                if seg_start >= seg_end {
                    // Skip empty segments
                } else if min_chars == 0 {
                    splits.push((seg_start, seg_end));
                } else if accum_start == accum_end {
                    accum_start = seg_start;
                    accum_end = seg_end;
                } else {
                    let accum_len = accum_end - accum_start;
                    let seg_len = seg_end - seg_start;

                    if accum_len < min_chars || seg_len < min_chars {
                        accum_end = seg_end;
                    } else {
                        splits.push((accum_start, accum_end));
                        accum_start = seg_start;
                        accum_end = seg_end;
                    }
                }
            };
        }

        for (match_pos, pattern_len) in split_points {
            match include_delim {
                IncludeDelim::Prev => {
                    let seg_end = match_pos + pattern_len;
                    emit_segment!(segment_start, seg_end);
                    segment_start = seg_end;
                }
                IncludeDelim::Next => {
                    if segment_start < match_pos {
                        emit_segment!(segment_start, match_pos);
                    }
                    segment_start = match_pos;
                }
                IncludeDelim::None => {
                    if segment_start < match_pos {
                        emit_segment!(segment_start, match_pos);
                    }
                    segment_start = match_pos + pattern_len;
                }
            }
        }

        if segment_start < text.len() {
            emit_segment!(segment_start, text.len());
        }

        if min_chars > 0 && accum_start < accum_end {
            splits.push((accum_start, accum_end));
        }

        splits
    }
}

/// Split text at every occurrence of any multi-byte pattern.
pub fn split_at_patterns(
    text: &[u8],
    patterns: &[&[u8]],
    include_delim: IncludeDelim,
    min_chars: usize,
) -> Vec<(usize, usize)> {
    if text.is_empty() {
        return vec![];
    }

    if patterns.is_empty() {
        return vec![(0, text.len())];
    }

    let mut trie = Trie::new();
    for (i, pattern) in patterns.iter().enumerate() {
        trie.add(pattern, i as u32);
    }
    trie.build(MatchKind::LeftmostFirst);
    let daac = trie.compile();

    let mut split_points: Vec<(usize, usize)> = Vec::new();
    for m in daac.find_iter(text) {
        split_points.push((m.start, m.end - m.start));
    }

    if split_points.is_empty() {
        return vec![(0, text.len())];
    }

    split_points.sort_by_key(|&(pos, _)| pos);

    let mut splits: Vec<(usize, usize)> = Vec::with_capacity(split_points.len() + 1);
    let mut segment_start = 0;
    let mut accum_start: usize = 0;
    let mut accum_end: usize = 0;

    macro_rules! emit_segment {
        ($seg_start:expr, $seg_end:expr) => {
            let seg_start = $seg_start;
            let seg_end = $seg_end;

            if seg_start >= seg_end {
                // Skip empty segments
            } else if min_chars == 0 {
                splits.push((seg_start, seg_end));
            } else if accum_start == accum_end {
                accum_start = seg_start;
                accum_end = seg_end;
            } else {
                let accum_len = accum_end - accum_start;
                let seg_len = seg_end - seg_start;

                if accum_len < min_chars || seg_len < min_chars {
                    accum_end = seg_end;
                } else {
                    splits.push((accum_start, accum_end));
                    accum_start = seg_start;
                    accum_end = seg_end;
                }
            }
        };
    }

    for (match_pos, pattern_len) in split_points {
        match include_delim {
            IncludeDelim::Prev => {
                let seg_end = match_pos + pattern_len;
                emit_segment!(segment_start, seg_end);
                segment_start = seg_end;
            }
            IncludeDelim::Next => {
                if segment_start < match_pos {
                    emit_segment!(segment_start, match_pos);
                }
                segment_start = match_pos;
            }
            IncludeDelim::None => {
                if segment_start < match_pos {
                    emit_segment!(segment_start, match_pos);
                }
                segment_start = match_pos + pattern_len;
            }
        }
    }

    if segment_start < text.len() {
        emit_segment!(segment_start, text.len());
    }

    if min_chars > 0 && accum_start < accum_end {
        splits.push((accum_start, accum_end));
    }

    splits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_basic() {
        let text = b"Hello. World. Test.";
        let offsets = split_at_delimiters(text, b".", IncludeDelim::Prev, 0);
        assert_eq!(offsets.len(), 3);
        assert_eq!(&text[offsets[0].0..offsets[0].1], b"Hello.");
        assert_eq!(&text[offsets[1].0..offsets[1].1], b" World.");
        assert_eq!(&text[offsets[2].0..offsets[2].1], b" Test.");
    }

    #[test]
    fn test_split_include_next() {
        let text = b"Hello. World. Test.";
        let offsets = split_at_delimiters(text, b".", IncludeDelim::Next, 0);
        assert_eq!(offsets.len(), 4);
        assert_eq!(&text[offsets[0].0..offsets[0].1], b"Hello");
        assert_eq!(&text[offsets[1].0..offsets[1].1], b". World");
        assert_eq!(&text[offsets[2].0..offsets[2].1], b". Test");
        assert_eq!(&text[offsets[3].0..offsets[3].1], b".");
    }

    #[test]
    fn test_split_preserves_all_bytes() {
        let text = b"The quick brown fox. Jumps over? The lazy dog!";
        let offsets = split_at_delimiters(text, b".?!", IncludeDelim::Prev, 0);

        let total: usize = offsets.iter().map(|(s, e)| e - s).sum();
        assert_eq!(total, text.len());

        for i in 1..offsets.len() {
            assert_eq!(offsets[i - 1].1, offsets[i].0);
        }
    }

    #[test]
    fn test_split_builder() {
        let text = b"Hello. World? Test!";
        let slices = split(text)
            .delimiters(b".?!")
            .include_prev()
            .collect_slices();
        assert_eq!(slices.len(), 3);
        assert_eq!(slices[0], b"Hello.");
        assert_eq!(slices[1], b" World?");
        assert_eq!(slices[2], b" Test!");
    }

    #[test]
    fn test_split_patterns_basic() {
        let text = b"Hello. World. Test.";
        let patterns: &[&[u8]] = &[b". "];
        let offsets = split_at_patterns(text, patterns, IncludeDelim::Prev, 0);
        assert_eq!(offsets.len(), 3);
        assert_eq!(&text[offsets[0].0..offsets[0].1], b"Hello. ");
        assert_eq!(&text[offsets[1].0..offsets[1].1], b"World. ");
        assert_eq!(&text[offsets[2].0..offsets[2].1], b"Test.");
    }

    #[test]
    fn test_split_patterns_preserves_all_bytes() {
        let text = b"The quick brown fox. Jumps over? The lazy dog!";
        let patterns: &[&[u8]] = &[b". ", b"? "];
        let offsets = split_at_patterns(text, patterns, IncludeDelim::Prev, 0);

        let total: usize = offsets.iter().map(|(s, e)| e - s).sum();
        assert_eq!(total, text.len());

        for i in 1..offsets.len() {
            assert_eq!(offsets[i - 1].1, offsets[i].0);
        }
    }
}
