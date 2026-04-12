//! Markdown-aware block splitting for semantic chunking.
//!
//! Parses markdown AST to identify block-level elements:
//! - Tables and code blocks are kept as atomic units (not split into rows/lines)
//! - Paragraphs are sentence-split for fine-grained embedding
//! - Headings, lists, and block quotes are kept whole

use pulldown_cmark::{Event, Options, Parser, Tag};

use super::sentence::split_sentences;

/// The kind of block extracted from the document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BlockKind {
    /// A sentence extracted from a paragraph.
    Sentence,
    /// A complete markdown table (kept atomic).
    Table,
    /// A fenced or indented code block (kept atomic).
    CodeBlock,
    /// A heading line.
    Heading,
    /// A complete list (kept atomic).
    List,
    /// A block quote (kept atomic).
    BlockQuote,
}

/// A block of text with its byte offset in the source document.
#[derive(Debug, Clone)]
pub struct Block<'a> {
    pub text: &'a str,
    pub offset: usize,
    pub kind: BlockKind,
}

/// Split markdown text into semantic blocks.
///
/// Tables, code blocks, lists, and block quotes are kept as single blocks.
/// Paragraphs are sentence-split using Unicode sentence boundaries.
/// Headings are emitted as individual blocks.
pub fn split_blocks(text: &str) -> Vec<Block<'_>> {
    let options =
        Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;

    let iter = Parser::new_ext(text, options).into_offset_iter();

    let mut blocks = Vec::new();
    let mut compound_start: Option<(usize, BlockKind)> = None;
    let mut depth: usize = 0;

    for (event, range) in iter {
        match event {
            Event::Start(ref tag) => {
                if compound_start.is_some() {
                    // Inside a compound block — just track nesting depth
                    depth += 1;
                } else {
                    let kind = match tag {
                        Tag::Table(_) => Some(BlockKind::Table),
                        Tag::CodeBlock(_) => Some(BlockKind::CodeBlock),
                        Tag::List(_) => Some(BlockKind::List),
                        Tag::BlockQuote(_) => Some(BlockKind::BlockQuote),
                        Tag::Heading { .. } => Some(BlockKind::Heading),
                        Tag::Paragraph => Some(BlockKind::Sentence),
                        _ => None,
                    };
                    if let Some(kind) = kind {
                        compound_start = Some((range.start, kind));
                        depth = 1;
                    }
                }
            }
            Event::End(_) => {
                if compound_start.is_some() {
                    depth -= 1;
                    if depth == 0 {
                        let (start, kind) = compound_start.take().unwrap();
                        let end = range.end;
                        let block_text = &text[start..end];

                        match kind {
                            BlockKind::Sentence => {
                                // Sentence-split paragraphs
                                for sent in split_sentences(block_text) {
                                    blocks.push(Block {
                                        text: sent.text,
                                        offset: start + sent.offset,
                                        kind: BlockKind::Sentence,
                                    });
                                }
                            }
                            _ => {
                                // Atomic block — keep whole
                                blocks.push(Block {
                                    text: block_text.trim_end_matches('\n'),
                                    offset: start,
                                    kind,
                                });
                            }
                        }
                    }
                }
            }
            _ => {
                // Text, Code, SoftBreak, HardBreak, Rule, Html, etc.
                // Handled implicitly via offset ranges of their parent blocks.
            }
        }
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_kept_atomic() {
        let md =
            "Some intro text.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nAfter the table.\n";
        let blocks = split_blocks(md);

        let table_blocks: Vec<_> = blocks
            .iter()
            .filter(|b| b.kind == BlockKind::Table)
            .collect();
        assert_eq!(table_blocks.len(), 1, "Table should be one atomic block");
        assert!(
            table_blocks[0].text.contains("| A | B |"),
            "Table block should contain full table"
        );
        assert!(
            table_blocks[0].text.contains("| 3 | 4 |"),
            "Table block should contain all rows"
        );
    }

    #[test]
    fn test_code_block_kept_atomic() {
        let md = "Before code.\n\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\n\nAfter code.\n";
        let blocks = split_blocks(md);

        let code_blocks: Vec<_> = blocks
            .iter()
            .filter(|b| b.kind == BlockKind::CodeBlock)
            .collect();
        assert_eq!(
            code_blocks.len(),
            1,
            "Code block should be one atomic block"
        );
        assert!(code_blocks[0].text.contains("fn main()"));
    }

    #[test]
    fn test_paragraphs_sentence_split() {
        let md = "First sentence. Second sentence. Third sentence.\n";
        let blocks = split_blocks(md);

        let sentence_blocks: Vec<_> = blocks
            .iter()
            .filter(|b| b.kind == BlockKind::Sentence)
            .collect();
        assert!(
            sentence_blocks.len() >= 2,
            "Paragraph should be split into sentences, got {}",
            sentence_blocks.len()
        );
    }

    #[test]
    fn test_heading_as_block() {
        let md = "## Section Title\n\nSome paragraph text.\n";
        let blocks = split_blocks(md);

        let heading_blocks: Vec<_> = blocks
            .iter()
            .filter(|b| b.kind == BlockKind::Heading)
            .collect();
        assert_eq!(heading_blocks.len(), 1);
        assert!(heading_blocks[0].text.contains("Section Title"));
    }

    #[test]
    fn test_mixed_document() {
        let md = "\
# Title

Introduction paragraph. With two sentences.

| Col1 | Col2 |
|------|------|
| a    | b    |

```python
print('hello')
```

Closing paragraph.
";
        let blocks = split_blocks(md);

        let kinds: Vec<BlockKind> = blocks.iter().map(|b| b.kind).collect();
        assert!(kinds.contains(&BlockKind::Heading), "Should have heading");
        assert!(kinds.contains(&BlockKind::Table), "Should have table");
        assert!(
            kinds.contains(&BlockKind::CodeBlock),
            "Should have code block"
        );
        assert!(
            kinds.contains(&BlockKind::Sentence),
            "Should have sentences"
        );

        // Table and code block should each appear exactly once
        assert_eq!(kinds.iter().filter(|k| **k == BlockKind::Table).count(), 1);
        assert_eq!(
            kinds.iter().filter(|k| **k == BlockKind::CodeBlock).count(),
            1
        );
    }

    #[test]
    fn test_offsets_are_correct() {
        let md = "Hello world.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nGoodbye.\n";
        let blocks = split_blocks(md);

        for block in &blocks {
            // The block text should appear at the claimed offset in the source
            let source_slice = &md[block.offset..block.offset + block.text.len()];
            assert_eq!(
                source_slice, block.text,
                "Offset mismatch for block: {:?}",
                block.kind
            );
        }
    }
}
