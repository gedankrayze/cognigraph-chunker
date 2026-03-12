//! Unicode script-based entity extraction.
//!
//! Uses Unicode script properties to detect entity-like spans in non-Latin
//! texts where capitalization heuristics don't apply. Provides:
//!
//! - **CJK:** Katakana runs (loanwords/proper nouns in Japanese),
//!   Latin runs in CJK text (foreign names/acronyms).
//! - **General:** Script-transition boundaries as entity delimiters.

use unicode_script::{Script, UnicodeScript};

use super::super::cognitive_types::{EntityType, NormalizedEntity};

/// Extract entities from text using Unicode script heuristics.
///
/// This complements the capitalization-based entity extractor for scripts
/// where capitalization doesn't exist (CJK, Arabic, etc.).
pub fn extract_script_entities(text: &str) -> Vec<NormalizedEntity> {
    let mut entities = Vec::new();

    extract_katakana_spans(text, &mut entities);
    extract_latin_in_nonlatin(text, &mut entities);

    deduplicate(&mut entities);
    entities
}

/// Extract Katakana runs as entity candidates.
///
/// In Japanese, Katakana is used for foreign loanwords, proper nouns,
/// technical terms, and emphasis — all strong entity signals.
fn extract_katakana_spans(text: &str, out: &mut Vec<NormalizedEntity>) {
    let mut span_start = None;
    let mut span_chars = String::new();

    for (byte_idx, ch) in text.char_indices() {
        let script = ch.script();
        let is_katakana = script == Script::Katakana;
        // Allow middle dot (ナカグロ) and prolonged sound mark inside Katakana spans
        let is_katakana_punct = ch == '・' || ch == 'ー';

        if is_katakana || (is_katakana_punct && span_start.is_some()) {
            if span_start.is_none() {
                span_start = Some(byte_idx);
            }
            span_chars.push(ch);
        } else {
            if let Some(_start) = span_start {
                // Trim trailing punctuation
                let trimmed = span_chars.trim_end_matches(['・', 'ー']);
                if trimmed.chars().count() >= 2 {
                    push_entity(out, trimmed);
                }
            }
            span_start = None;
            span_chars.clear();
        }
    }

    // Handle trailing span
    if span_start.is_some() {
        let trimmed = span_chars.trim_end_matches(['・', 'ー']);
        if trimmed.chars().count() >= 2 {
            push_entity(out, trimmed);
        }
    }
}

/// Extract Latin script runs embedded in non-Latin text.
///
/// In CJK/Arabic/Cyrillic documents, Latin runs typically represent
/// foreign names, acronyms, or technical identifiers (e.g., "API", "ONNX",
/// "Claude" in a Japanese document).
fn extract_latin_in_nonlatin(text: &str, out: &mut Vec<NormalizedEntity>) {
    // First check if the text is predominantly non-Latin
    let total_alpha: usize = text.chars().filter(|c| c.is_alphabetic()).count();
    let latin_alpha: usize = text
        .chars()
        .filter(|c| c.is_alphabetic() && c.script() == Script::Latin)
        .count();

    if total_alpha == 0 {
        return;
    }

    // Only apply this heuristic if text is majority non-Latin
    let latin_ratio = latin_alpha as f64 / total_alpha as f64;
    if latin_ratio > 0.5 {
        return;
    }

    let mut span_start = None;
    let mut span_chars = String::new();

    for (byte_idx, ch) in text.char_indices() {
        let is_latin =
            ch.script() == Script::Latin || (ch.is_ascii_digit() && span_start.is_some());

        if is_latin {
            if span_start.is_none() {
                span_start = Some(byte_idx);
            }
            span_chars.push(ch);
        } else {
            if span_start.is_some() {
                let trimmed = span_chars.trim();
                if trimmed.len() >= 2 && trimmed.chars().any(|c| c.is_alphabetic()) {
                    push_entity(out, trimmed);
                }
            }
            span_start = None;
            span_chars.clear();
        }
    }

    if span_start.is_some() {
        let trimmed = span_chars.trim();
        if trimmed.len() >= 2 && trimmed.chars().any(|c| c.is_alphabetic()) {
            push_entity(out, trimmed);
        }
    }
}

fn push_entity(out: &mut Vec<NormalizedEntity>, surface: &str) {
    out.push(NormalizedEntity {
        surface_form: surface.to_string(),
        normalized: surface.to_lowercase(),
        entity_type: EntityType::Named,
    });
}

fn deduplicate(entities: &mut Vec<NormalizedEntity>) {
    let mut seen = std::collections::HashSet::new();
    entities.retain(|e| {
        let key = (e.normalized.clone(), e.entity_type);
        seen.insert(key)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_katakana_entity_extraction() {
        // "CogniGraph (コグニグラフ) Chunker processes documents"
        let text = "コグニグラフチャンカーはドキュメントを処理します。";
        let entities = extract_script_entities(text);
        let surfaces: Vec<&str> = entities.iter().map(|e| e.surface_form.as_str()).collect();
        assert!(
            surfaces.iter().any(|s| s.contains("コグニグラフ")),
            "Should detect Katakana span, got: {surfaces:?}"
        );
    }

    #[test]
    fn test_katakana_with_nakaguro() {
        // Middle dot (・) is used in katakana multi-word names
        let text = "マシン・ラーニングは広い分野です。";
        let entities = extract_script_entities(text);
        let surfaces: Vec<&str> = entities.iter().map(|e| e.surface_form.as_str()).collect();
        assert!(
            surfaces.iter().any(|s| s.contains("マシン・ラーニング")),
            "Should detect Katakana span with nakaguro, got: {surfaces:?}"
        );
    }

    #[test]
    fn test_latin_in_japanese() {
        let text = "このシステムはONNXモデルとClaude APIを使用します。";
        let entities = extract_script_entities(text);
        let surfaces: Vec<&str> = entities.iter().map(|e| e.surface_form.as_str()).collect();
        assert!(
            surfaces.contains(&"ONNX"),
            "Should detect Latin 'ONNX' in Japanese text, got: {surfaces:?}"
        );
        assert!(
            surfaces.iter().any(|s| s.contains("Claude")),
            "Should detect Latin 'Claude' in Japanese text, got: {surfaces:?}"
        );
    }

    #[test]
    fn test_latin_in_chinese() {
        let text = "认知图分块器使用ONNX模型进行推理。";
        let entities = extract_script_entities(text);
        let surfaces: Vec<&str> = entities.iter().map(|e| e.surface_form.as_str()).collect();
        assert!(
            surfaces.contains(&"ONNX"),
            "Should detect 'ONNX' in Chinese text, got: {surfaces:?}"
        );
    }

    #[test]
    fn test_no_false_positives_in_english() {
        // In predominantly Latin text, should not extract every word
        let text = "The CogniGraph Chunker processes documents efficiently.";
        let entities = extract_script_entities(text);
        assert!(
            entities.is_empty(),
            "Should not extract entities from English text (Latin-majority), got: {entities:?}"
        );
    }

    #[test]
    fn test_latin_in_arabic() {
        let text = "يستخدم النظام نموذج BERT للتصنيف.";
        let entities = extract_script_entities(text);
        let surfaces: Vec<&str> = entities.iter().map(|e| e.surface_form.as_str()).collect();
        assert!(
            surfaces.contains(&"BERT"),
            "Should detect 'BERT' in Arabic text, got: {surfaces:?}"
        );
    }

    #[test]
    fn test_short_katakana_ignored() {
        // Single katakana character should be ignored
        let text = "これはテストです。ア。";
        let entities = extract_script_entities(text);
        // "ア" alone (1 char) should be filtered out, "テスト" (3 chars) should be kept
        let surfaces: Vec<&str> = entities.iter().map(|e| e.surface_form.as_str()).collect();
        assert!(
            !surfaces.contains(&"ア"),
            "Single katakana should be filtered, got: {surfaces:?}"
        );
    }
}
