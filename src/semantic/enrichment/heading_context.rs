//! Heading path propagation for structural context.
//!
//! Tracks the heading ancestry (breadcrumb path) for each block
//! by maintaining a stack of heading levels as blocks are processed.

use std::collections::HashSet;

use super::super::blocks::{Block, BlockKind};

/// Compute heading paths for a sequence of blocks.
///
/// Each block receives the heading ancestry active at its position.
/// Also returns the set of all heading terms (lowercased) for entity matching.
pub fn compute_heading_paths(blocks: &[Block<'_>]) -> (Vec<Vec<String>>, HashSet<String>) {
    let mut paths: Vec<Vec<String>> = Vec::with_capacity(blocks.len());
    let mut heading_stack: Vec<(usize, String)> = Vec::new(); // (level, text)
    let mut heading_terms = HashSet::new();

    for block in blocks {
        if block.kind == BlockKind::Heading {
            let (level, text) = parse_heading(block.text);
            let clean_text = text.to_string();

            // Pop headings at same or deeper level
            while heading_stack.last().is_some_and(|(l, _)| *l >= level) {
                heading_stack.pop();
            }

            heading_stack.push((level, clean_text.clone()));

            // Add heading terms for entity matching
            for word in clean_text.split_whitespace() {
                let lower = word
                    .trim_matches(|c: char| c.is_ascii_punctuation())
                    .to_lowercase();
                if lower.len() >= 3 {
                    heading_terms.insert(lower);
                }
            }
            // Also add the full heading as a term
            let full_lower = clean_text.to_lowercase();
            if full_lower.len() >= 3 {
                heading_terms.insert(full_lower);
            }
        }

        // Current path for this block
        let path: Vec<String> = heading_stack.iter().map(|(_, t)| t.clone()).collect();
        paths.push(path);
    }

    (paths, heading_terms)
}

/// Check whether two heading paths share the same immediate parent.
pub fn heading_continuity(path_a: &[String], path_b: &[String]) -> f64 {
    if path_a.is_empty() && path_b.is_empty() {
        return 1.0; // Both at document root
    }
    if path_a == path_b {
        return 1.0; // Same heading context
    }

    // Check shared prefix length
    let shared = path_a
        .iter()
        .zip(path_b.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let max_depth = path_a.len().max(path_b.len());
    if max_depth == 0 {
        return 1.0;
    }

    shared as f64 / max_depth as f64
}

/// Parse heading level and text from markdown heading block.
///
/// Input: "# Title" or "## Section" or "### Subsection"
/// Returns: (level, "Title"), (2, "Section"), (3, "Subsection")
fn parse_heading(text: &str) -> (usize, &str) {
    let trimmed = text.trim();
    let level = trimmed.chars().take_while(|&c| c == '#').count();
    let heading_text = trimmed[level..].trim();
    (level.max(1), heading_text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::blocks::Block;

    #[test]
    fn test_heading_path_propagation() {
        let blocks = vec![
            Block {
                text: "# Introduction",
                offset: 0,
                kind: BlockKind::Heading,
            },
            Block {
                text: "Some intro text.",
                offset: 16,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "## Architecture",
                offset: 34,
                kind: BlockKind::Heading,
            },
            Block {
                text: "Architecture details.",
                offset: 51,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "### Scoring",
                offset: 73,
                kind: BlockKind::Heading,
            },
            Block {
                text: "Scoring formula.",
                offset: 86,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "## Evaluation",
                offset: 104,
                kind: BlockKind::Heading,
            },
            Block {
                text: "Evaluation details.",
                offset: 119,
                kind: BlockKind::Sentence,
            },
        ];

        let (paths, terms) = compute_heading_paths(&blocks);

        // "# Introduction" → path: ["Introduction"]
        assert_eq!(paths[0], vec!["Introduction"]);
        // Sentence under intro → same path
        assert_eq!(paths[1], vec!["Introduction"]);
        // "## Architecture" → ["Introduction", "Architecture"]
        assert_eq!(paths[2], vec!["Introduction", "Architecture"]);
        // Sentence under arch → same
        assert_eq!(paths[3], vec!["Introduction", "Architecture"]);
        // "### Scoring" → ["Introduction", "Architecture", "Scoring"]
        assert_eq!(paths[4], vec!["Introduction", "Architecture", "Scoring"]);
        // "## Evaluation" pops Architecture and Scoring
        assert_eq!(paths[6], vec!["Introduction", "Evaluation"]);
        assert_eq!(paths[7], vec!["Introduction", "Evaluation"]);

        // Heading terms should include relevant words
        assert!(terms.contains("architecture"));
        assert!(terms.contains("scoring"));
        assert!(terms.contains("evaluation"));
    }

    #[test]
    fn test_heading_continuity() {
        let a = vec!["Intro".to_string(), "Architecture".to_string()];
        let b = vec!["Intro".to_string(), "Architecture".to_string()];
        assert_eq!(heading_continuity(&a, &b), 1.0);

        let c = vec!["Intro".to_string(), "Evaluation".to_string()];
        assert!((heading_continuity(&a, &c) - 0.5).abs() < 0.01);

        let d: Vec<String> = vec![];
        let e: Vec<String> = vec![];
        assert_eq!(heading_continuity(&d, &e), 1.0);
    }

    #[test]
    fn test_parse_heading() {
        assert_eq!(parse_heading("# Title"), (1, "Title"));
        assert_eq!(parse_heading("## Section"), (2, "Section"));
        assert_eq!(parse_heading("### Sub Section"), (3, "Sub Section"));
    }
}
