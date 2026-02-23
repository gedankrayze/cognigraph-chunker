//! Sentence splitting for semantic chunking.

use unicode_segmentation::UnicodeSegmentation;

/// A sentence with its byte offset in the original text.
#[derive(Debug, Clone)]
pub struct Sentence<'a> {
    pub text: &'a str,
    pub offset: usize,
}

/// Split text into sentences using Unicode sentence boundaries.
///
/// Returns sentences with their byte offsets in the original text.
/// Empty/whitespace-only sentences are filtered out.
pub fn split_sentences(text: &str) -> Vec<Sentence<'_>> {
    text.unicode_sentences()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let offset = s.as_ptr() as usize - text.as_ptr() as usize;
            Sentence { text: s, offset }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_basic() {
        let text = "Hello world. This is a test. Another sentence here.";
        let sentences = split_sentences(text);
        assert!(sentences.len() >= 2);
        // Verify offsets are correct
        for s in &sentences {
            assert_eq!(&text[s.offset..s.offset + s.text.len()], s.text);
        }
    }

    #[test]
    fn test_split_sentences_empty() {
        let sentences = split_sentences("");
        assert!(sentences.is_empty());
    }

    #[test]
    fn test_split_sentences_single() {
        let text = "Just one sentence.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].offset, 0);
    }
}
