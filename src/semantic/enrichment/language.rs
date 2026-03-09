//! Language detection for multilingual enrichment.
//!
//! Uses `whatlang` for trigram-based language identification.
//! The detected language selects which discourse markers, pronoun lists,
//! and entity heuristics to apply.

use std::str::FromStr;

use whatlang::{Lang, detect};

/// Supported language groups for enrichment dispatch.
///
/// Languages are grouped by shared heuristic strategies rather than
/// one-to-one with ISO codes, to keep the dispatch manageable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LanguageGroup {
    /// English
    English,
    /// German
    German,
    /// French
    French,
    /// Spanish
    Spanish,
    /// Portuguese
    Portuguese,
    /// Italian
    Italian,
    /// Dutch
    Dutch,
    /// Russian
    Russian,
    /// Chinese (Simplified + Traditional)
    Chinese,
    /// Japanese
    Japanese,
    /// Korean
    Korean,
    /// Arabic
    Arabic,
    /// Turkish
    Turkish,
    /// Polish
    Polish,
    /// Any other detected language — falls back to script-based heuristics only.
    Other,
}

impl FromStr for LanguageGroup {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "en" | "eng" | "english" => Ok(Self::English),
            "de" | "deu" | "german" => Ok(Self::German),
            "fr" | "fra" | "french" => Ok(Self::French),
            "es" | "spa" | "spanish" => Ok(Self::Spanish),
            "pt" | "por" | "portuguese" => Ok(Self::Portuguese),
            "it" | "ita" | "italian" => Ok(Self::Italian),
            "nl" | "nld" | "dutch" => Ok(Self::Dutch),
            "ru" | "rus" | "russian" => Ok(Self::Russian),
            "zh" | "cmn" | "chinese" => Ok(Self::Chinese),
            "ja" | "jpn" | "japanese" => Ok(Self::Japanese),
            "ko" | "kor" | "korean" => Ok(Self::Korean),
            "ar" | "ara" | "arabic" => Ok(Self::Arabic),
            "tr" | "tur" | "turkish" => Ok(Self::Turkish),
            "pl" | "pol" | "polish" => Ok(Self::Polish),
            "auto" => Ok(Self::Other), // Other triggers auto-detect in pipeline
            _ => Err(format!(
                "Unknown language '{s}'. Supported: en, de, fr, es, pt, it, nl, ru, zh, ja, ko, ar, tr, pl, auto"
            )),
        }
    }
}

impl std::fmt::Display for LanguageGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::English => write!(f, "en"),
            Self::German => write!(f, "de"),
            Self::French => write!(f, "fr"),
            Self::Spanish => write!(f, "es"),
            Self::Portuguese => write!(f, "pt"),
            Self::Italian => write!(f, "it"),
            Self::Dutch => write!(f, "nl"),
            Self::Russian => write!(f, "ru"),
            Self::Chinese => write!(f, "zh"),
            Self::Japanese => write!(f, "ja"),
            Self::Korean => write!(f, "ko"),
            Self::Arabic => write!(f, "ar"),
            Self::Turkish => write!(f, "tr"),
            Self::Polish => write!(f, "pl"),
            Self::Other => write!(f, "auto"),
        }
    }
}

/// Detect the primary language of a text.
///
/// Returns `LanguageGroup::English` if detection fails or text is too short.
pub fn detect_language(text: &str) -> LanguageGroup {
    let Some(info) = detect(text) else {
        return LanguageGroup::English;
    };

    match info.lang() {
        Lang::Eng => LanguageGroup::English,
        Lang::Deu => LanguageGroup::German,
        Lang::Fra => LanguageGroup::French,
        Lang::Spa => LanguageGroup::Spanish,
        Lang::Por => LanguageGroup::Portuguese,
        Lang::Ita => LanguageGroup::Italian,
        Lang::Nld => LanguageGroup::Dutch,
        Lang::Rus => LanguageGroup::Russian,
        Lang::Cmn => LanguageGroup::Chinese,
        Lang::Jpn => LanguageGroup::Japanese,
        Lang::Kor => LanguageGroup::Korean,
        Lang::Ara => LanguageGroup::Arabic,
        Lang::Tur => LanguageGroup::Turkish,
        Lang::Pol => LanguageGroup::Polish,
        _ => LanguageGroup::Other,
    }
}

/// Pronoun lists for supported languages.
///
/// Returns sentence-initial pronouns that indicate anaphoric reference.
pub fn pronouns_for(lang: LanguageGroup) -> &'static [&'static str] {
    match lang {
        LanguageGroup::English => &[
            "it", "they", "he", "she", "we", "its", "their", "his", "her", "our",
        ],
        LanguageGroup::German => &[
            "es", "sie", "er", "wir", "sein", "seine", "ihr", "ihre", "deren", "dessen",
        ],
        LanguageGroup::French => &[
            "il", "elle", "ils", "elles", "on", "nous", "son", "sa", "ses", "leur", "leurs",
        ],
        LanguageGroup::Spanish => &[
            "él", "ella", "ellos", "ellas", "su", "sus", "nosotros", "nosotras",
        ],
        LanguageGroup::Portuguese => &[
            "ele", "ela", "eles", "elas", "seu", "sua", "seus", "suas", "nós",
        ],
        LanguageGroup::Italian => &[
            "esso", "essa", "essi", "esse", "egli", "lui", "lei", "loro", "noi", "suo", "sua",
        ],
        LanguageGroup::Dutch => &[
            "het", "hij", "zij", "ze", "wij", "we", "hun", "haar", "zijn",
        ],
        LanguageGroup::Russian => &[
            "он", "она", "оно", "они", "его", "её", "их", "мы", "наш", "наша",
        ],
        LanguageGroup::Turkish => &["o", "onlar", "biz", "onun", "onların", "bizim"],
        LanguageGroup::Polish => &[
            "on", "ona", "ono", "oni", "one", "my", "ich", "jej", "jego", "nasz", "nasza",
        ],
        // CJK and Arabic: pronoun-start heuristic is less applicable
        _ => &[],
    }
}

/// Demonstrative prefixes for supported languages.
///
/// Returns prefix patterns that indicate backward reference ("this X", "diese X", etc.)
pub fn demonstrative_prefixes_for(lang: LanguageGroup) -> &'static [&'static str] {
    match lang {
        LanguageGroup::English => &[
            "this ",
            "that ",
            "these ",
            "those ",
            "such ",
            "the same ",
            "the above ",
            "the following ",
        ],
        LanguageGroup::German => &[
            "diese ", "dieser ", "dieses ", "diesen ", "diesem ", "jene ", "jener ", "jenes ",
            "jenen ", "jenem ", "solche ", "solcher ", "solches ",
        ],
        LanguageGroup::French => &[
            "ce ", "cet ", "cette ", "ces ", "celui ", "celle ", "ceux ", "celles ",
        ],
        LanguageGroup::Spanish => &[
            "este ",
            "esta ",
            "estos ",
            "estas ",
            "ese ",
            "esa ",
            "esos ",
            "esas ",
            "aquel ",
            "aquella ",
            "aquellos ",
            "aquellas ",
            "dicho ",
            "dicha ",
            "dichos ",
            "dichas ",
        ],
        LanguageGroup::Portuguese => &[
            "este ", "esta ", "estes ", "estas ", "esse ", "essa ", "esses ", "essas ", "aquele ",
            "aquela ", "aqueles ", "aquelas ",
        ],
        LanguageGroup::Italian => &[
            "questo ", "questa ", "questi ", "queste ", "quello ", "quella ", "quelli ", "quelle ",
            "tale ", "tali ",
        ],
        LanguageGroup::Dutch => &["dit ", "dat ", "deze ", "die ", "dergelijk ", "dergelijke "],
        LanguageGroup::Russian => &[
            "этот ",
            "эта ",
            "это ",
            "эти ",
            "тот ",
            "та ",
            "те ",
            "такой ",
            "такая ",
            "такое ",
            "такие ",
            "данный ",
            "данная ",
            "данное ",
            "данные ",
        ],
        LanguageGroup::Turkish => &["bu ", "şu ", "o ", "böyle ", "öyle ", "söz konusu "],
        LanguageGroup::Polish => &[
            "ten ", "ta ", "to ", "te ", "ci ", "tamten ", "tamta ", "tamto ", "taki ", "taka ",
            "takie ",
        ],
        _ => &[],
    }
}

/// Common articles/prepositions/stopwords to exclude from entity spans.
///
/// Used by the capitalized-span entity extractor to avoid treating
/// function words as entity components.
pub fn stopwords_for(lang: LanguageGroup) -> &'static [&'static str] {
    match lang {
        LanguageGroup::English => &[
            "the", "a", "an", "in", "on", "at", "to", "of", "by", "for", "with", "from", "and",
            "or", "but", "is", "are", "was", "were", "not", "no",
        ],
        LanguageGroup::German => &[
            "der", "die", "das", "ein", "eine", "und", "oder", "aber", "in", "an", "auf", "zu",
            "von", "mit", "für", "ist", "sind", "war", "nicht", "kein", "keine",
        ],
        LanguageGroup::French => &[
            "le", "la", "les", "un", "une", "des", "et", "ou", "mais", "dans", "sur", "de", "du",
            "au", "aux", "par", "pour", "avec", "est", "sont", "pas", "ne",
        ],
        LanguageGroup::Spanish => &[
            "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "en", "de",
            "del", "al", "por", "para", "con", "es", "son", "no",
        ],
        LanguageGroup::Portuguese => &[
            "o", "a", "os", "as", "um", "uma", "uns", "umas", "e", "ou", "mas", "em", "de", "do",
            "da", "dos", "das", "por", "para", "com", "é", "são", "não",
        ],
        LanguageGroup::Italian => &[
            "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "e", "o", "ma", "in", "di",
            "da", "del", "della", "per", "con", "è", "sono", "non",
        ],
        LanguageGroup::Dutch => &[
            "de", "het", "een", "en", "of", "maar", "in", "op", "van", "met", "voor", "door", "is",
            "zijn", "niet", "geen",
        ],
        LanguageGroup::Russian => &[
            "и", "или", "но", "в", "на", "с", "по", "для", "из", "к", "о", "от", "не", "это",
            "что", "как",
        ],
        LanguageGroup::Turkish => &["ve", "veya", "ama", "ile", "için", "bir", "bu", "da", "de"],
        LanguageGroup::Polish => &[
            "i", "lub", "ale", "w", "na", "z", "do", "od", "dla", "nie", "to", "jest", "są",
        ],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_english() {
        let lang = detect_language(
            "The CogniGraph Chunker processes documents by detecting semantic boundaries.",
        );
        assert_eq!(lang, LanguageGroup::English);
    }

    #[test]
    fn test_detect_german() {
        let lang = detect_language(
            "Der CogniGraph Chunker verarbeitet Dokumente durch die Erkennung semantischer Grenzen.",
        );
        assert_eq!(lang, LanguageGroup::German);
    }

    #[test]
    fn test_detect_french() {
        let lang = detect_language(
            "Le système de découpage cognitif traite les documents en détectant les frontières sémantiques.",
        );
        assert_eq!(lang, LanguageGroup::French);
    }

    #[test]
    fn test_detect_japanese() {
        let lang = detect_language(
            "認知グラフチャンカーは、意味的な境界を検出してドキュメントを処理します。",
        );
        assert_eq!(lang, LanguageGroup::Japanese);
    }

    #[test]
    fn test_detect_chinese() {
        let lang = detect_language("认知图分块器通过检测语义边界来处理文档。");
        assert_eq!(lang, LanguageGroup::Chinese);
    }

    #[test]
    fn test_detect_russian() {
        let lang = detect_language(
            "Когнитивный разделитель обрабатывает документы путём обнаружения семантических границ.",
        );
        assert_eq!(lang, LanguageGroup::Russian);
    }

    #[test]
    fn test_fallback_on_short_text() {
        // Very short text may not be reliably detectable
        let lang = detect_language("Hi");
        // whatlang may return English or Other for very short text — both are acceptable
        assert!(
            lang == LanguageGroup::English || lang == LanguageGroup::Other,
            "Short text should fall back gracefully, got: {lang:?}"
        );
    }

    #[test]
    fn test_pronoun_lists_nonempty() {
        assert!(!pronouns_for(LanguageGroup::English).is_empty());
        assert!(!pronouns_for(LanguageGroup::German).is_empty());
        assert!(!pronouns_for(LanguageGroup::French).is_empty());
    }

    #[test]
    fn test_demonstrative_lists_nonempty() {
        assert!(!demonstrative_prefixes_for(LanguageGroup::English).is_empty());
        assert!(!demonstrative_prefixes_for(LanguageGroup::Spanish).is_empty());
        assert!(!demonstrative_prefixes_for(LanguageGroup::Russian).is_empty());
    }
}
