//! Cognitive enrichment pipeline.
//!
//! Transforms raw `Block`s into `BlockEnvelope`s with entity, discourse,
//! and heading context signals attached.
//!
//! Automatically detects the document language and selects appropriate
//! discourse markers, pronoun/demonstrative lists, and entity heuristics.

pub mod discourse;
pub mod entities;
pub mod heading_context;
pub mod language;
pub mod multilingual_discourse;
pub mod script_entities;

use super::blocks::{Block, BlockKind};
use super::cognitive_types::{BlockEnvelope, ContinuationFlags};
use entities::{extract_entities, extract_noun_phrases};
use heading_context::compute_heading_paths;
use language::{
    LanguageGroup, demonstrative_prefixes_for, detect_language, pronouns_for, stopwords_for,
};
use multilingual_discourse::detect_discourse_markers_multilingual;
use script_entities::extract_script_entities;

/// Detect if text starts with a pronoun for the given language.
fn starts_with_pronoun_multilingual(text: &str, lang: LanguageGroup) -> bool {
    let first_word = text
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase();

    pronouns_for(lang).iter().any(|&p| p == first_word)
}

/// Detect if text starts with a demonstrative prefix for the given language.
fn starts_with_demonstrative_multilingual(text: &str, lang: LanguageGroup) -> bool {
    let lower = text.trim_start().to_lowercase();
    demonstrative_prefixes_for(lang)
        .iter()
        .any(|prefix| lower.starts_with(prefix))
}

/// Enrich a sequence of parsed blocks with cognitive signals.
///
/// Detects the document language from the concatenated text and applies
/// language-appropriate enrichment heuristics.
pub fn enrich_blocks(blocks: &[Block<'_>]) -> Vec<BlockEnvelope> {
    // Detect language from the first ~500 chars of content (skip headings)
    let sample: String = blocks
        .iter()
        .filter(|b| b.kind != BlockKind::Heading)
        .map(|b| b.text)
        .take(5)
        .collect::<Vec<_>>()
        .join(" ");
    let lang = detect_language(&sample);

    enrich_blocks_with_language(blocks, lang)
}

/// Enrich blocks with a specific language (for testing or when language is known).
pub fn enrich_blocks_with_language(
    blocks: &[Block<'_>],
    lang: LanguageGroup,
) -> Vec<BlockEnvelope> {
    let (heading_paths, heading_terms) = compute_heading_paths(blocks);

    // Determine if we should use script-based entity extraction
    let use_script_entities = matches!(
        lang,
        LanguageGroup::Japanese
            | LanguageGroup::Chinese
            | LanguageGroup::Korean
            | LanguageGroup::Arabic
    );

    // Language-specific stopwords for entity span filtering
    let stopwords = stopwords_for(lang);

    blocks
        .iter()
        .zip(heading_paths)
        .enumerate()
        .map(|(i, (block, heading_path))| {
            let text = block.text;

            // Entity extraction: combine capitalization-based + script-based
            let mut entities = extract_entities(text, &heading_terms, stopwords);
            if use_script_entities {
                entities.extend(extract_script_entities(text));
            }

            let noun_phrases = extract_noun_phrases(text, stopwords);

            // Language-aware discourse marker detection
            let discourse_markers = detect_discourse_markers_multilingual(text, lang);

            let continuation_flags = ContinuationFlags {
                starts_with_pronoun: starts_with_pronoun_multilingual(text, lang),
                starts_with_demonstrative: starts_with_demonstrative_multilingual(text, lang),
                starts_with_discourse: !discourse_markers.is_empty(),
                continues_list: block.kind == BlockKind::List
                    && i > 0
                    && blocks[i - 1].kind == BlockKind::List,
            };

            // Simple token estimate: ~4 characters per token
            let token_estimate = text.len().div_ceil(4);

            BlockEnvelope {
                text: text.to_string(),
                offset_start: block.offset,
                offset_end: block.offset + text.len(),
                block_type: block.kind,
                heading_path,
                embedding: None,
                entities,
                noun_phrases,
                discourse_markers,
                continuation_flags,
                token_estimate,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::blocks::split_blocks;
    use crate::semantic::cognitive_types::DiscourseMarker;

    #[test]
    fn test_enrich_basic_document() {
        let md = "\
# Architecture

The CogniGraph Chunker uses a pipeline approach.
Furthermore, it supports multiple providers.

## Scoring

This model computes boundary scores.
";
        let blocks = split_blocks(md);
        let enriched = enrich_blocks(&blocks);

        assert!(!enriched.is_empty());

        // First block should be heading with heading path
        assert_eq!(enriched[0].block_type, BlockKind::Heading);
        assert_eq!(enriched[0].heading_path, vec!["Architecture"]);

        // Sentence with "Furthermore" should have discourse marker
        let furthermore_block = enriched.iter().find(|b| b.text.contains("Furthermore"));
        assert!(furthermore_block.is_some());
        let fb = furthermore_block.unwrap();
        assert!(!fb.discourse_markers.is_empty());
        assert!(fb.continuation_flags.starts_with_discourse);

        // "This model" should trigger demonstrative flag
        let this_block = enriched.iter().find(|b| b.text.starts_with("This model"));
        assert!(this_block.is_some());
        assert!(
            this_block
                .unwrap()
                .continuation_flags
                .starts_with_demonstrative
        );

        // Blocks under ## Scoring should have heading path ["Architecture", "Scoring"]
        let scoring_block = enriched.iter().find(|b| b.text.contains("boundary scores"));
        assert!(scoring_block.is_some());
        assert_eq!(
            scoring_block.unwrap().heading_path,
            vec!["Architecture", "Scoring"]
        );
    }

    #[test]
    fn test_token_estimate() {
        let blocks = vec![Block {
            text: "Hello world, this is a test sentence.",
            offset: 0,
            kind: BlockKind::Sentence,
        }];
        let enriched = enrich_blocks(&blocks);
        // 36 chars → ~9 tokens
        assert!(enriched[0].token_estimate > 0);
        assert!(enriched[0].token_estimate < 20);
    }

    #[test]
    fn test_german_enrichment() {
        let blocks = vec![
            Block {
                text: "Der CogniGraph Chunker verarbeitet Dokumente.",
                offset: 0,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "Außerdem unterstützt es mehrere Anbieter.",
                offset: 47,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "Dieses System ist modular aufgebaut.",
                offset: 89,
                kind: BlockKind::Sentence,
            },
        ];
        let enriched = enrich_blocks_with_language(&blocks, LanguageGroup::German);

        // "Außerdem" should trigger Continuation discourse marker
        assert_eq!(
            enriched[1].discourse_markers,
            vec![DiscourseMarker::Continuation],
            "German 'Außerdem' should be detected as Continuation"
        );
        assert!(enriched[1].continuation_flags.starts_with_discourse);

        // "Dieses" should trigger demonstrative
        assert!(
            enriched[2].continuation_flags.starts_with_demonstrative,
            "German 'Dieses' should trigger demonstrative flag"
        );
    }

    #[test]
    fn test_japanese_enrichment() {
        let blocks = vec![
            Block {
                text: "コグニグラフチャンカーはドキュメントを処理します。",
                offset: 0,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "さらに、複数のプロバイダーをサポートしています。",
                offset: 50,
                kind: BlockKind::Sentence,
            },
        ];
        let enriched = enrich_blocks_with_language(&blocks, LanguageGroup::Japanese);

        // Katakana entities should be extracted
        assert!(
            enriched[0]
                .entities
                .iter()
                .any(|e| e.surface_form.contains("コグニグラフ")),
            "Should extract Katakana entity, got: {:?}",
            enriched[0].entities
        );

        // "さらに" should trigger Continuation
        assert_eq!(
            enriched[1].discourse_markers,
            vec![DiscourseMarker::Continuation],
            "Japanese 'さらに' should be detected as Continuation"
        );
    }

    #[test]
    fn test_chinese_enrichment() {
        let blocks = vec![
            Block {
                text: "认知图分块器使用ONNX模型。",
                offset: 0,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "然而，这种方法有局限性。",
                offset: 30,
                kind: BlockKind::Sentence,
            },
        ];
        let enriched = enrich_blocks_with_language(&blocks, LanguageGroup::Chinese);

        // Latin "ONNX" should be extracted as entity in Chinese text
        assert!(
            enriched[0]
                .entities
                .iter()
                .any(|e| e.surface_form == "ONNX"),
            "Should extract Latin 'ONNX' entity in Chinese text, got: {:?}",
            enriched[0].entities
        );

        // "然而" should trigger Contrast
        assert_eq!(
            enriched[1].discourse_markers,
            vec![DiscourseMarker::Contrast],
            "Chinese '然而' should be detected as Contrast"
        );
    }
}
