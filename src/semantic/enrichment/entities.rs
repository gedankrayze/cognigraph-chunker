//! Lightweight entity extraction without heavy NLP dependencies.
//!
//! Level A: capitalized spans, repeated noun phrases, heading terms.
//! Level B: demonstratives, pronoun detection, subject repetition.

use std::collections::HashSet;

use super::super::cognitive_types::{EntityType, NormalizedEntity};

/// Extract entity mentions from a block of text.
///
/// Uses cheap lexical and structural heuristics (no NER model required).
/// The `stopwords` parameter provides language-specific function words to exclude
/// from capitalized entity spans (e.g., "The", "And" in English, "Der", "Die" in German).
pub fn extract_entities(
    text: &str,
    heading_terms: &HashSet<String>,
    stopwords: &[&str],
) -> Vec<NormalizedEntity> {
    let mut entities = Vec::new();

    // Level A: Named entities — capitalized multi-word spans
    extract_capitalized_spans(text, stopwords, &mut entities);

    // Level A: Heading terms that appear in the text
    extract_heading_term_matches(text, heading_terms, &mut entities);

    // Level B: Demonstrative references — "this X", "these X", "such X", "the X"
    extract_demonstratives(text, &mut entities);

    // Level B: Pronoun starts
    extract_pronouns(text, &mut entities);

    // Deduplicate by normalized form
    deduplicate(&mut entities);

    entities
}

/// Extract noun phrases from text (simple heuristic: sequences of capitalized words).
pub fn extract_noun_phrases(text: &str, stopwords: &[&str]) -> Vec<String> {
    let mut phrases = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| c.is_ascii_punctuation());
        if clean.is_empty() {
            continue;
        }
        let first_char = clean.chars().next().unwrap();
        if first_char.is_uppercase() && clean.len() > 1 && !is_stopword(clean, stopwords) {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(clean);
        } else {
            if current.split_whitespace().count() >= 2 {
                phrases.push(current.clone());
            }
            current.clear();
        }
    }
    if current.split_whitespace().count() >= 2 {
        phrases.push(current);
    }

    phrases
}

/// Detect if text starts with a pronoun (sentence-initial).
pub fn starts_with_pronoun(text: &str) -> bool {
    let first_word = first_word_lower(text);
    matches!(
        first_word.as_str(),
        "it" | "they" | "he" | "she" | "we" | "its" | "their" | "his" | "her" | "our"
    )
}

/// Detect if text starts with a demonstrative phrase.
pub fn starts_with_demonstrative(text: &str) -> bool {
    let lower = text.trim_start().to_lowercase();
    // Two-word demonstratives: "this X", "that X", "these X", "those X", "such X"
    DEMONSTRATIVE_PREFIXES
        .iter()
        .any(|prefix| lower.starts_with(prefix))
}

const DEMONSTRATIVE_PREFIXES: &[&str] = &[
    "this ",
    "that ",
    "these ",
    "those ",
    "such ",
    "the same ",
    "the above ",
    "the following ",
];

/// Compute entity overlap between two sets of entities (0.0 to 1.0).
pub fn entity_overlap(a: &[NormalizedEntity], b: &[NormalizedEntity]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }

    let set_a: HashSet<&str> = a.iter().map(|e| e.normalized.as_str()).collect();
    let set_b: HashSet<&str> = b.iter().map(|e| e.normalized.as_str()).collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Compute noun-phrase overlap between two blocks (0.0 to 1.0).
pub fn noun_phrase_overlap(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }

    let set_a: HashSet<String> = a.iter().map(|s| s.to_lowercase()).collect();
    let set_b: HashSet<String> = b.iter().map(|s| s.to_lowercase()).collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// ── Internal helpers ────────────────────────────────────────────────

fn extract_capitalized_spans(text: &str, stopwords: &[&str], out: &mut Vec<NormalizedEntity>) {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0;

    while i < words.len() {
        let clean = words[i].trim_matches(|c: char| c.is_ascii_punctuation());
        if clean.is_empty() || clean.len() < 2 {
            i += 1;
            continue;
        }

        let first_char = clean.chars().next().unwrap();
        if first_char.is_uppercase() && !is_stopword(clean, stopwords) {
            // Collect consecutive capitalized words
            let start = i;
            let mut span_words = vec![clean.to_string()];
            i += 1;

            while i < words.len() {
                let next = words[i].trim_matches(|c: char| c.is_ascii_punctuation());
                if !next.is_empty() && next.chars().next().unwrap().is_uppercase() && next.len() > 1
                {
                    span_words.push(next.to_string());
                    i += 1;
                } else {
                    break;
                }
            }

            // Skip single capitalized words at sentence start (index 0 or after period)
            let is_sentence_start =
                start == 0 || (start > 0 && words[start - 1].ends_with(['.', '!', '?']));

            if span_words.len() >= 2 || (span_words.len() == 1 && !is_sentence_start) {
                let surface = span_words.join(" ");
                let normalized = surface.to_lowercase();
                out.push(NormalizedEntity {
                    surface_form: surface,
                    normalized,
                    entity_type: EntityType::Named,
                });
            }
        } else {
            i += 1;
        }
    }
}

fn extract_heading_term_matches(
    text: &str,
    heading_terms: &HashSet<String>,
    out: &mut Vec<NormalizedEntity>,
) {
    let text_lower = text.to_lowercase();
    for term in heading_terms {
        if text_lower.contains(term.as_str()) {
            out.push(NormalizedEntity {
                surface_form: term.clone(),
                normalized: term.clone(),
                entity_type: EntityType::Named,
            });
        }
    }
}

fn extract_demonstratives(text: &str, out: &mut Vec<NormalizedEntity>) {
    let lower = text.to_lowercase();

    for prefix in DEMONSTRATIVE_PREFIXES {
        if let Some(rest) = lower.strip_prefix(prefix) {
            // Extract the noun that follows the demonstrative
            if let Some(noun) = rest.split_whitespace().next() {
                let noun = noun.trim_matches(|c: char| c.is_ascii_punctuation());
                if noun.len() >= 2 {
                    let surface = format!("{}{}", prefix.trim(), &format!(" {noun}"));
                    out.push(NormalizedEntity {
                        surface_form: surface.clone(),
                        normalized: noun.to_string(),
                        entity_type: EntityType::Demonstrative,
                    });
                }
            }
        }
    }
}

fn extract_pronouns(text: &str, out: &mut Vec<NormalizedEntity>) {
    let first = first_word_lower(text);
    if matches!(
        first.as_str(),
        "it" | "they" | "he" | "she" | "we" | "its" | "their" | "his" | "her" | "our"
    ) {
        out.push(NormalizedEntity {
            surface_form: first.clone(),
            normalized: first,
            entity_type: EntityType::Pronoun,
        });
    }
}

fn is_stopword(word: &str, stopwords: &[&str]) -> bool {
    let lower = word.to_lowercase();
    stopwords.iter().any(|&sw| sw == lower)
}

fn first_word_lower(text: &str) -> String {
    text.split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase()
}

fn deduplicate(entities: &mut Vec<NormalizedEntity>) {
    let mut seen = HashSet::new();
    entities.retain(|e| {
        let key = (e.normalized.clone(), e.entity_type);
        seen.insert(key)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::semantic::enrichment::language::{LanguageGroup, stopwords_for};

    const EN_STOPS: LanguageGroup = LanguageGroup::English;

    fn en_stopwords() -> &'static [&'static str] {
        stopwords_for(EN_STOPS)
    }

    #[test]
    fn test_capitalized_spans() {
        let text = "The CogniGraph Chunker processes documents efficiently.";
        let entities = extract_entities(text, &HashSet::new(), en_stopwords());
        let named: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Named)
            .collect();
        assert!(
            named.iter().any(|e| e.normalized == "cognigraph chunker"),
            "Should detect 'CogniGraph Chunker' as named entity, got: {:?}",
            named
        );
    }

    #[test]
    fn test_pronoun_detection() {
        assert!(starts_with_pronoun("It processes documents."));
        assert!(starts_with_pronoun("They are efficient."));
        assert!(!starts_with_pronoun("The chunker is fast."));
        assert!(!starts_with_pronoun("CogniGraph works well."));
    }

    #[test]
    fn test_demonstrative_detection() {
        assert!(starts_with_demonstrative("This model is fast."));
        assert!(starts_with_demonstrative("These results show improvement."));
        assert!(starts_with_demonstrative("Such systems require care."));
        assert!(!starts_with_demonstrative("A model is fast."));
    }

    #[test]
    fn test_entity_overlap() {
        let a = vec![NormalizedEntity {
            surface_form: "CogniGraph".into(),
            normalized: "cognigraph".into(),
            entity_type: EntityType::Named,
        }];
        let b = vec![
            NormalizedEntity {
                surface_form: "CogniGraph".into(),
                normalized: "cognigraph".into(),
                entity_type: EntityType::Named,
            },
            NormalizedEntity {
                surface_form: "ONNX".into(),
                normalized: "onnx".into(),
                entity_type: EntityType::Named,
            },
        ];
        let overlap = entity_overlap(&a, &b);
        assert!((overlap - 0.5).abs() < 0.01); // 1 shared / 2 total
    }

    #[test]
    fn test_heading_term_match() {
        let mut heading_terms = HashSet::new();
        heading_terms.insert("architecture".to_string());
        let entities = extract_entities(
            "The architecture is modular.",
            &heading_terms,
            en_stopwords(),
        );
        assert!(
            entities
                .iter()
                .any(|e| e.normalized == "architecture" && e.entity_type == EntityType::Named)
        );
    }

    #[test]
    fn test_noun_phrases() {
        let text = "The CogniGraph Chunker uses Cross Encoders for better accuracy.";
        let phrases = extract_noun_phrases(text, en_stopwords());
        assert!(phrases.iter().any(|p| p == "CogniGraph Chunker"));
        assert!(phrases.iter().any(|p| p == "Cross Encoders"));
    }
}
