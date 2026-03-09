//! Multilingual discourse marker tables.
//!
//! Provides discourse marker patterns for multiple languages,
//! selected at runtime based on detected language.
//!
//! Sources: Connective-Lex.info database, standard linguistic resources.

use super::super::cognitive_types::DiscourseMarker;
use super::language::LanguageGroup;

/// Get the discourse marker pattern table for a language.
///
/// Returns `(prefix, DiscourseMarker)` pairs ordered longest-first
/// within each category, matching the English table structure.
pub fn discourse_patterns_for(lang: LanguageGroup) -> &'static [(&'static str, DiscourseMarker)] {
    match lang {
        LanguageGroup::English => ENGLISH_PATTERNS,
        LanguageGroup::German => GERMAN_PATTERNS,
        LanguageGroup::French => FRENCH_PATTERNS,
        LanguageGroup::Spanish => SPANISH_PATTERNS,
        LanguageGroup::Portuguese => PORTUGUESE_PATTERNS,
        LanguageGroup::Italian => ITALIAN_PATTERNS,
        LanguageGroup::Dutch => DUTCH_PATTERNS,
        LanguageGroup::Russian => RUSSIAN_PATTERNS,
        LanguageGroup::Turkish => TURKISH_PATTERNS,
        LanguageGroup::Polish => POLISH_PATTERNS,
        LanguageGroup::Chinese => CHINESE_PATTERNS,
        LanguageGroup::Japanese => JAPANESE_PATTERNS,
        LanguageGroup::Korean => KOREAN_PATTERNS,
        LanguageGroup::Arabic => ARABIC_PATTERNS,
        LanguageGroup::Other => &[],
    }
}

/// Detect discourse markers using language-specific patterns.
///
/// Replaces the English-only `detect_discourse_markers` when a language is known.
pub fn detect_discourse_markers_multilingual(
    text: &str,
    lang: LanguageGroup,
) -> Vec<DiscourseMarker> {
    let lower = text.trim_start().to_lowercase();
    let patterns = discourse_patterns_for(lang);

    for &(pattern, marker) in patterns {
        if lower.starts_with(pattern) {
            return vec![marker];
        }
    }

    vec![]
}

// ── English ────────────────────────────────────────────────────────

const ENGLISH_PATTERNS: &[(&str, DiscourseMarker)] = &[
    // Continuation
    ("furthermore,", DiscourseMarker::Continuation),
    ("furthermore ", DiscourseMarker::Continuation),
    ("additionally,", DiscourseMarker::Continuation),
    ("additionally ", DiscourseMarker::Continuation),
    ("in addition,", DiscourseMarker::Continuation),
    ("in addition ", DiscourseMarker::Continuation),
    ("moreover,", DiscourseMarker::Continuation),
    ("moreover ", DiscourseMarker::Continuation),
    ("also,", DiscourseMarker::Continuation),
    ("also ", DiscourseMarker::Continuation),
    ("likewise,", DiscourseMarker::Continuation),
    ("likewise ", DiscourseMarker::Continuation),
    ("similarly,", DiscourseMarker::Continuation),
    ("similarly ", DiscourseMarker::Continuation),
    // Contrast
    ("on the other hand,", DiscourseMarker::Contrast),
    ("on the other hand ", DiscourseMarker::Contrast),
    ("in contrast,", DiscourseMarker::Contrast),
    ("in contrast ", DiscourseMarker::Contrast),
    ("on the contrary,", DiscourseMarker::Contrast),
    ("conversely,", DiscourseMarker::Contrast),
    ("nevertheless,", DiscourseMarker::Contrast),
    ("nonetheless,", DiscourseMarker::Contrast),
    ("however,", DiscourseMarker::Contrast),
    ("however ", DiscourseMarker::Contrast),
    ("although ", DiscourseMarker::Contrast),
    ("though ", DiscourseMarker::Contrast),
    ("but ", DiscourseMarker::Contrast),
    ("yet ", DiscourseMarker::Contrast),
    // Causation
    ("as a result,", DiscourseMarker::Causation),
    ("as a result ", DiscourseMarker::Causation),
    ("consequently,", DiscourseMarker::Causation),
    ("consequently ", DiscourseMarker::Causation),
    ("therefore,", DiscourseMarker::Causation),
    ("therefore ", DiscourseMarker::Causation),
    ("because ", DiscourseMarker::Causation),
    ("thus,", DiscourseMarker::Causation),
    ("thus ", DiscourseMarker::Causation),
    ("hence,", DiscourseMarker::Causation),
    ("hence ", DiscourseMarker::Causation),
    // Exemplification
    ("for instance,", DiscourseMarker::Exemplification),
    ("for instance ", DiscourseMarker::Exemplification),
    ("for example,", DiscourseMarker::Exemplification),
    ("for example ", DiscourseMarker::Exemplification),
    ("such as ", DiscourseMarker::Exemplification),
    ("e.g.,", DiscourseMarker::Exemplification),
    ("e.g. ", DiscourseMarker::Exemplification),
    // Elaboration
    ("in particular,", DiscourseMarker::Elaboration),
    ("in particular ", DiscourseMarker::Elaboration),
    ("specifically,", DiscourseMarker::Elaboration),
    ("specifically ", DiscourseMarker::Elaboration),
    ("more precisely,", DiscourseMarker::Elaboration),
    ("namely,", DiscourseMarker::Elaboration),
    ("namely ", DiscourseMarker::Elaboration),
    ("that is,", DiscourseMarker::Elaboration),
    ("i.e.,", DiscourseMarker::Elaboration),
    ("i.e. ", DiscourseMarker::Elaboration),
    // Conclusion
    ("in conclusion,", DiscourseMarker::Conclusion),
    ("in conclusion ", DiscourseMarker::Conclusion),
    ("in summary,", DiscourseMarker::Conclusion),
    ("in summary ", DiscourseMarker::Conclusion),
    ("to summarize,", DiscourseMarker::Conclusion),
    ("to conclude,", DiscourseMarker::Conclusion),
    ("overall,", DiscourseMarker::Conclusion),
    ("overall ", DiscourseMarker::Conclusion),
    ("finally,", DiscourseMarker::Conclusion),
    ("finally ", DiscourseMarker::Conclusion),
];

// ── German ─────────────────────────────────────────────────────────

const GERMAN_PATTERNS: &[(&str, DiscourseMarker)] = &[
    // Continuation
    ("darüber hinaus ", DiscourseMarker::Continuation),
    ("außerdem ", DiscourseMarker::Continuation),
    ("zusätzlich ", DiscourseMarker::Continuation),
    ("zudem ", DiscourseMarker::Continuation),
    ("ferner ", DiscourseMarker::Continuation),
    ("ebenso ", DiscourseMarker::Continuation),
    ("auch ", DiscourseMarker::Continuation),
    // Contrast
    ("auf der anderen seite ", DiscourseMarker::Contrast),
    ("im gegensatz dazu ", DiscourseMarker::Contrast),
    ("im gegensatz ", DiscourseMarker::Contrast),
    ("nichtsdestotrotz ", DiscourseMarker::Contrast),
    ("dennoch ", DiscourseMarker::Contrast),
    ("jedoch ", DiscourseMarker::Contrast),
    ("allerdings ", DiscourseMarker::Contrast),
    ("obwohl ", DiscourseMarker::Contrast),
    ("aber ", DiscourseMarker::Contrast),
    ("doch ", DiscourseMarker::Contrast),
    // Causation
    ("infolgedessen ", DiscourseMarker::Causation),
    ("folglich ", DiscourseMarker::Causation),
    ("daher ", DiscourseMarker::Causation),
    ("deshalb ", DiscourseMarker::Causation),
    ("deswegen ", DiscourseMarker::Causation),
    ("weil ", DiscourseMarker::Causation),
    ("denn ", DiscourseMarker::Causation),
    ("somit ", DiscourseMarker::Causation),
    // Exemplification
    ("zum beispiel ", DiscourseMarker::Exemplification),
    ("beispielsweise ", DiscourseMarker::Exemplification),
    ("etwa ", DiscourseMarker::Exemplification),
    ("z.b. ", DiscourseMarker::Exemplification),
    // Elaboration
    ("insbesondere ", DiscourseMarker::Elaboration),
    ("genauer gesagt ", DiscourseMarker::Elaboration),
    ("konkret ", DiscourseMarker::Elaboration),
    ("nämlich ", DiscourseMarker::Elaboration),
    ("das heißt ", DiscourseMarker::Elaboration),
    ("d.h. ", DiscourseMarker::Elaboration),
    // Conclusion
    ("zusammenfassend ", DiscourseMarker::Conclusion),
    ("abschließend ", DiscourseMarker::Conclusion),
    ("insgesamt ", DiscourseMarker::Conclusion),
    ("schließlich ", DiscourseMarker::Conclusion),
];

// ── French ─────────────────────────────────────────────────────────

const FRENCH_PATTERNS: &[(&str, DiscourseMarker)] = &[
    // Continuation
    ("de plus,", DiscourseMarker::Continuation),
    ("de plus ", DiscourseMarker::Continuation),
    ("en outre,", DiscourseMarker::Continuation),
    ("en outre ", DiscourseMarker::Continuation),
    ("par ailleurs,", DiscourseMarker::Continuation),
    ("également,", DiscourseMarker::Continuation),
    ("également ", DiscourseMarker::Continuation),
    ("aussi,", DiscourseMarker::Continuation),
    ("aussi ", DiscourseMarker::Continuation),
    ("de même,", DiscourseMarker::Continuation),
    // Contrast
    ("en revanche,", DiscourseMarker::Contrast),
    ("en revanche ", DiscourseMarker::Contrast),
    ("au contraire,", DiscourseMarker::Contrast),
    ("néanmoins,", DiscourseMarker::Contrast),
    ("néanmoins ", DiscourseMarker::Contrast),
    ("toutefois,", DiscourseMarker::Contrast),
    ("toutefois ", DiscourseMarker::Contrast),
    ("cependant,", DiscourseMarker::Contrast),
    ("cependant ", DiscourseMarker::Contrast),
    ("pourtant,", DiscourseMarker::Contrast),
    ("bien que ", DiscourseMarker::Contrast),
    ("mais ", DiscourseMarker::Contrast),
    // Causation
    ("par conséquent,", DiscourseMarker::Causation),
    ("en conséquence,", DiscourseMarker::Causation),
    ("c'est pourquoi ", DiscourseMarker::Causation),
    ("donc,", DiscourseMarker::Causation),
    ("donc ", DiscourseMarker::Causation),
    ("ainsi,", DiscourseMarker::Causation),
    ("ainsi ", DiscourseMarker::Causation),
    ("parce que ", DiscourseMarker::Causation),
    ("car ", DiscourseMarker::Causation),
    // Exemplification
    ("par exemple,", DiscourseMarker::Exemplification),
    ("par exemple ", DiscourseMarker::Exemplification),
    ("notamment,", DiscourseMarker::Exemplification),
    ("notamment ", DiscourseMarker::Exemplification),
    // Elaboration
    ("en particulier,", DiscourseMarker::Elaboration),
    ("en particulier ", DiscourseMarker::Elaboration),
    ("plus précisément,", DiscourseMarker::Elaboration),
    ("c'est-à-dire ", DiscourseMarker::Elaboration),
    // Conclusion
    ("en conclusion,", DiscourseMarker::Conclusion),
    ("en résumé,", DiscourseMarker::Conclusion),
    ("pour conclure,", DiscourseMarker::Conclusion),
    ("en somme,", DiscourseMarker::Conclusion),
    ("finalement,", DiscourseMarker::Conclusion),
    ("finalement ", DiscourseMarker::Conclusion),
    ("enfin,", DiscourseMarker::Conclusion),
    ("enfin ", DiscourseMarker::Conclusion),
];

// ── Spanish ────────────────────────────────────────────────────────

const SPANISH_PATTERNS: &[(&str, DiscourseMarker)] = &[
    // Continuation
    ("además,", DiscourseMarker::Continuation),
    ("además ", DiscourseMarker::Continuation),
    ("asimismo,", DiscourseMarker::Continuation),
    ("asimismo ", DiscourseMarker::Continuation),
    ("igualmente,", DiscourseMarker::Continuation),
    ("también,", DiscourseMarker::Continuation),
    ("también ", DiscourseMarker::Continuation),
    ("del mismo modo,", DiscourseMarker::Continuation),
    // Contrast
    ("por el contrario,", DiscourseMarker::Contrast),
    ("en contraste,", DiscourseMarker::Contrast),
    ("no obstante,", DiscourseMarker::Contrast),
    ("sin embargo,", DiscourseMarker::Contrast),
    ("sin embargo ", DiscourseMarker::Contrast),
    ("aunque ", DiscourseMarker::Contrast),
    ("pero ", DiscourseMarker::Contrast),
    // Causation
    ("en consecuencia,", DiscourseMarker::Causation),
    ("por consiguiente,", DiscourseMarker::Causation),
    ("por lo tanto,", DiscourseMarker::Causation),
    ("por lo tanto ", DiscourseMarker::Causation),
    ("por eso,", DiscourseMarker::Causation),
    ("por eso ", DiscourseMarker::Causation),
    ("porque ", DiscourseMarker::Causation),
    // Exemplification
    ("por ejemplo,", DiscourseMarker::Exemplification),
    ("por ejemplo ", DiscourseMarker::Exemplification),
    // Elaboration
    ("en particular,", DiscourseMarker::Elaboration),
    ("en particular ", DiscourseMarker::Elaboration),
    ("concretamente,", DiscourseMarker::Elaboration),
    ("es decir,", DiscourseMarker::Elaboration),
    ("es decir ", DiscourseMarker::Elaboration),
    // Conclusion
    ("en conclusión,", DiscourseMarker::Conclusion),
    ("en resumen,", DiscourseMarker::Conclusion),
    ("en suma,", DiscourseMarker::Conclusion),
    ("finalmente,", DiscourseMarker::Conclusion),
    ("finalmente ", DiscourseMarker::Conclusion),
    ("por último,", DiscourseMarker::Conclusion),
];

// ── Portuguese ─────────────────────────────────────────────────────

const PORTUGUESE_PATTERNS: &[(&str, DiscourseMarker)] = &[
    ("além disso,", DiscourseMarker::Continuation),
    ("além disso ", DiscourseMarker::Continuation),
    ("também,", DiscourseMarker::Continuation),
    ("também ", DiscourseMarker::Continuation),
    ("igualmente,", DiscourseMarker::Continuation),
    ("no entanto,", DiscourseMarker::Contrast),
    ("no entanto ", DiscourseMarker::Contrast),
    ("contudo,", DiscourseMarker::Contrast),
    ("porém,", DiscourseMarker::Contrast),
    ("porém ", DiscourseMarker::Contrast),
    ("todavia,", DiscourseMarker::Contrast),
    ("embora ", DiscourseMarker::Contrast),
    ("mas ", DiscourseMarker::Contrast),
    ("por conseguinte,", DiscourseMarker::Causation),
    ("portanto,", DiscourseMarker::Causation),
    ("portanto ", DiscourseMarker::Causation),
    ("porque ", DiscourseMarker::Causation),
    ("por exemplo,", DiscourseMarker::Exemplification),
    ("por exemplo ", DiscourseMarker::Exemplification),
    ("em particular,", DiscourseMarker::Elaboration),
    ("ou seja,", DiscourseMarker::Elaboration),
    ("em conclusão,", DiscourseMarker::Conclusion),
    ("em resumo,", DiscourseMarker::Conclusion),
    ("finalmente,", DiscourseMarker::Conclusion),
    ("finalmente ", DiscourseMarker::Conclusion),
];

// ── Italian ────────────────────────────────────────────────────────

const ITALIAN_PATTERNS: &[(&str, DiscourseMarker)] = &[
    ("inoltre,", DiscourseMarker::Continuation),
    ("inoltre ", DiscourseMarker::Continuation),
    ("per di più,", DiscourseMarker::Continuation),
    ("anche,", DiscourseMarker::Continuation),
    ("anche ", DiscourseMarker::Continuation),
    ("al contrario,", DiscourseMarker::Contrast),
    ("tuttavia,", DiscourseMarker::Contrast),
    ("tuttavia ", DiscourseMarker::Contrast),
    ("comunque,", DiscourseMarker::Contrast),
    ("però,", DiscourseMarker::Contrast),
    ("però ", DiscourseMarker::Contrast),
    ("sebbene ", DiscourseMarker::Contrast),
    ("ma ", DiscourseMarker::Contrast),
    ("di conseguenza,", DiscourseMarker::Causation),
    ("pertanto,", DiscourseMarker::Causation),
    ("quindi,", DiscourseMarker::Causation),
    ("quindi ", DiscourseMarker::Causation),
    ("perché ", DiscourseMarker::Causation),
    ("per esempio,", DiscourseMarker::Exemplification),
    ("per esempio ", DiscourseMarker::Exemplification),
    ("ad esempio,", DiscourseMarker::Exemplification),
    ("in particolare,", DiscourseMarker::Elaboration),
    ("cioè,", DiscourseMarker::Elaboration),
    ("cioè ", DiscourseMarker::Elaboration),
    ("in conclusione,", DiscourseMarker::Conclusion),
    ("in sintesi,", DiscourseMarker::Conclusion),
    ("infine,", DiscourseMarker::Conclusion),
    ("infine ", DiscourseMarker::Conclusion),
];

// ── Dutch ──────────────────────────────────────────────────────────

const DUTCH_PATTERNS: &[(&str, DiscourseMarker)] = &[
    ("bovendien,", DiscourseMarker::Continuation),
    ("bovendien ", DiscourseMarker::Continuation),
    ("daarnaast,", DiscourseMarker::Continuation),
    ("daarnaast ", DiscourseMarker::Continuation),
    ("tevens,", DiscourseMarker::Continuation),
    ("ook,", DiscourseMarker::Continuation),
    ("ook ", DiscourseMarker::Continuation),
    ("daarentegen,", DiscourseMarker::Contrast),
    ("desondanks,", DiscourseMarker::Contrast),
    ("niettemin,", DiscourseMarker::Contrast),
    ("echter,", DiscourseMarker::Contrast),
    ("echter ", DiscourseMarker::Contrast),
    ("hoewel ", DiscourseMarker::Contrast),
    ("maar ", DiscourseMarker::Contrast),
    ("daardoor,", DiscourseMarker::Causation),
    ("daarom,", DiscourseMarker::Causation),
    ("daarom ", DiscourseMarker::Causation),
    ("dus,", DiscourseMarker::Causation),
    ("dus ", DiscourseMarker::Causation),
    ("omdat ", DiscourseMarker::Causation),
    ("bijvoorbeeld,", DiscourseMarker::Exemplification),
    ("bijvoorbeeld ", DiscourseMarker::Exemplification),
    ("in het bijzonder,", DiscourseMarker::Elaboration),
    ("namelijk,", DiscourseMarker::Elaboration),
    ("namelijk ", DiscourseMarker::Elaboration),
    ("tot slot,", DiscourseMarker::Conclusion),
    ("samengevat,", DiscourseMarker::Conclusion),
    ("ten slotte,", DiscourseMarker::Conclusion),
    ("uiteindelijk,", DiscourseMarker::Conclusion),
];

// ── Russian ────────────────────────────────────────────────────────

const RUSSIAN_PATTERNS: &[(&str, DiscourseMarker)] = &[
    ("кроме того,", DiscourseMarker::Continuation),
    ("кроме того ", DiscourseMarker::Continuation),
    ("более того,", DiscourseMarker::Continuation),
    ("помимо этого,", DiscourseMarker::Continuation),
    ("также ", DiscourseMarker::Continuation),
    ("с другой стороны,", DiscourseMarker::Contrast),
    ("напротив,", DiscourseMarker::Contrast),
    ("тем не менее,", DiscourseMarker::Contrast),
    ("тем не менее ", DiscourseMarker::Contrast),
    ("однако,", DiscourseMarker::Contrast),
    ("однако ", DiscourseMarker::Contrast),
    ("хотя ", DiscourseMarker::Contrast),
    ("но ", DiscourseMarker::Contrast),
    ("в результате,", DiscourseMarker::Causation),
    ("следовательно,", DiscourseMarker::Causation),
    ("поэтому,", DiscourseMarker::Causation),
    ("поэтому ", DiscourseMarker::Causation),
    ("потому что ", DiscourseMarker::Causation),
    ("таким образом,", DiscourseMarker::Causation),
    ("например,", DiscourseMarker::Exemplification),
    ("например ", DiscourseMarker::Exemplification),
    ("в частности,", DiscourseMarker::Elaboration),
    ("в частности ", DiscourseMarker::Elaboration),
    ("а именно,", DiscourseMarker::Elaboration),
    ("то есть ", DiscourseMarker::Elaboration),
    ("в заключение,", DiscourseMarker::Conclusion),
    ("подводя итог,", DiscourseMarker::Conclusion),
    ("в итоге,", DiscourseMarker::Conclusion),
    ("наконец,", DiscourseMarker::Conclusion),
    ("наконец ", DiscourseMarker::Conclusion),
];

// ── Turkish ────────────────────────────────────────────────────────

const TURKISH_PATTERNS: &[(&str, DiscourseMarker)] = &[
    ("bunun yanı sıra,", DiscourseMarker::Continuation),
    ("ayrıca,", DiscourseMarker::Continuation),
    ("ayrıca ", DiscourseMarker::Continuation),
    ("üstelik,", DiscourseMarker::Continuation),
    ("buna karşın,", DiscourseMarker::Contrast),
    ("öte yandan,", DiscourseMarker::Contrast),
    ("bununla birlikte,", DiscourseMarker::Contrast),
    ("ancak,", DiscourseMarker::Contrast),
    ("ancak ", DiscourseMarker::Contrast),
    ("fakat ", DiscourseMarker::Contrast),
    ("ama ", DiscourseMarker::Contrast),
    ("dolayısıyla,", DiscourseMarker::Causation),
    ("bu nedenle,", DiscourseMarker::Causation),
    ("bu yüzden,", DiscourseMarker::Causation),
    ("çünkü ", DiscourseMarker::Causation),
    ("örneğin,", DiscourseMarker::Exemplification),
    ("örneğin ", DiscourseMarker::Exemplification),
    ("özellikle,", DiscourseMarker::Elaboration),
    ("özellikle ", DiscourseMarker::Elaboration),
    ("yani,", DiscourseMarker::Elaboration),
    ("yani ", DiscourseMarker::Elaboration),
    ("sonuç olarak,", DiscourseMarker::Conclusion),
    ("özetle,", DiscourseMarker::Conclusion),
    ("son olarak,", DiscourseMarker::Conclusion),
];

// ── Polish ─────────────────────────────────────────────────────────

const POLISH_PATTERNS: &[(&str, DiscourseMarker)] = &[
    ("ponadto,", DiscourseMarker::Continuation),
    ("ponadto ", DiscourseMarker::Continuation),
    ("dodatkowo,", DiscourseMarker::Continuation),
    ("również,", DiscourseMarker::Continuation),
    ("również ", DiscourseMarker::Continuation),
    ("także,", DiscourseMarker::Continuation),
    ("także ", DiscourseMarker::Continuation),
    ("z drugiej strony,", DiscourseMarker::Contrast),
    ("natomiast,", DiscourseMarker::Contrast),
    ("natomiast ", DiscourseMarker::Contrast),
    ("jednakże,", DiscourseMarker::Contrast),
    ("jednak,", DiscourseMarker::Contrast),
    ("jednak ", DiscourseMarker::Contrast),
    ("ale ", DiscourseMarker::Contrast),
    ("w rezultacie,", DiscourseMarker::Causation),
    ("w związku z tym,", DiscourseMarker::Causation),
    ("dlatego,", DiscourseMarker::Causation),
    ("dlatego ", DiscourseMarker::Causation),
    ("ponieważ ", DiscourseMarker::Causation),
    ("na przykład,", DiscourseMarker::Exemplification),
    ("na przykład ", DiscourseMarker::Exemplification),
    ("w szczególności,", DiscourseMarker::Elaboration),
    ("to znaczy,", DiscourseMarker::Elaboration),
    ("to znaczy ", DiscourseMarker::Elaboration),
    ("podsumowując,", DiscourseMarker::Conclusion),
    ("w podsumowaniu,", DiscourseMarker::Conclusion),
    ("wreszcie,", DiscourseMarker::Conclusion),
    ("wreszcie ", DiscourseMarker::Conclusion),
];

// ── Chinese ────────────────────────────────────────────────────────

const CHINESE_PATTERNS: &[(&str, DiscourseMarker)] = &[
    // Continuation
    ("此外，", DiscourseMarker::Continuation),
    ("此外", DiscourseMarker::Continuation),
    ("另外，", DiscourseMarker::Continuation),
    ("而且，", DiscourseMarker::Continuation),
    ("而且", DiscourseMarker::Continuation),
    ("同样，", DiscourseMarker::Continuation),
    ("同时，", DiscourseMarker::Continuation),
    ("同时", DiscourseMarker::Continuation),
    // Contrast
    ("相反，", DiscourseMarker::Contrast),
    ("然而，", DiscourseMarker::Contrast),
    ("然而", DiscourseMarker::Contrast),
    ("不过，", DiscourseMarker::Contrast),
    ("但是，", DiscourseMarker::Contrast),
    ("但是", DiscourseMarker::Contrast),
    ("虽然", DiscourseMarker::Contrast),
    ("尽管", DiscourseMarker::Contrast),
    // Causation
    ("因此，", DiscourseMarker::Causation),
    ("因此", DiscourseMarker::Causation),
    ("所以，", DiscourseMarker::Causation),
    ("所以", DiscourseMarker::Causation),
    ("因为", DiscourseMarker::Causation),
    ("由于", DiscourseMarker::Causation),
    // Exemplification
    ("例如，", DiscourseMarker::Exemplification),
    ("例如", DiscourseMarker::Exemplification),
    ("比如，", DiscourseMarker::Exemplification),
    ("比如", DiscourseMarker::Exemplification),
    // Elaboration
    ("具体来说，", DiscourseMarker::Elaboration),
    ("特别是，", DiscourseMarker::Elaboration),
    ("特别是", DiscourseMarker::Elaboration),
    ("也就是说，", DiscourseMarker::Elaboration),
    ("即，", DiscourseMarker::Elaboration),
    // Conclusion
    ("总之，", DiscourseMarker::Conclusion),
    ("总之", DiscourseMarker::Conclusion),
    ("综上所述，", DiscourseMarker::Conclusion),
    ("总结来说，", DiscourseMarker::Conclusion),
    ("最后，", DiscourseMarker::Conclusion),
    ("最后", DiscourseMarker::Conclusion),
];

// ── Japanese ───────────────────────────────────────────────────────

const JAPANESE_PATTERNS: &[(&str, DiscourseMarker)] = &[
    // Continuation
    ("さらに、", DiscourseMarker::Continuation),
    ("さらに", DiscourseMarker::Continuation),
    ("また、", DiscourseMarker::Continuation),
    ("また", DiscourseMarker::Continuation),
    ("加えて、", DiscourseMarker::Continuation),
    ("その上、", DiscourseMarker::Continuation),
    ("同様に、", DiscourseMarker::Continuation),
    // Contrast
    ("一方で、", DiscourseMarker::Contrast),
    ("一方、", DiscourseMarker::Contrast),
    ("それにもかかわらず、", DiscourseMarker::Contrast),
    ("しかしながら、", DiscourseMarker::Contrast),
    ("しかし、", DiscourseMarker::Contrast),
    ("しかし", DiscourseMarker::Contrast),
    ("けれども、", DiscourseMarker::Contrast),
    ("ただし、", DiscourseMarker::Contrast),
    ("だが、", DiscourseMarker::Contrast),
    // Causation
    ("その結果、", DiscourseMarker::Causation),
    ("したがって、", DiscourseMarker::Causation),
    ("したがって", DiscourseMarker::Causation),
    ("そのため、", DiscourseMarker::Causation),
    ("なぜなら", DiscourseMarker::Causation),
    // Exemplification
    ("例えば、", DiscourseMarker::Exemplification),
    ("例えば", DiscourseMarker::Exemplification),
    ("たとえば、", DiscourseMarker::Exemplification),
    // Elaboration
    ("具体的には、", DiscourseMarker::Elaboration),
    ("特に、", DiscourseMarker::Elaboration),
    ("特に", DiscourseMarker::Elaboration),
    ("すなわち、", DiscourseMarker::Elaboration),
    ("つまり、", DiscourseMarker::Elaboration),
    ("つまり", DiscourseMarker::Elaboration),
    // Conclusion
    ("結論として、", DiscourseMarker::Conclusion),
    ("要するに、", DiscourseMarker::Conclusion),
    ("まとめると、", DiscourseMarker::Conclusion),
    ("最後に、", DiscourseMarker::Conclusion),
    ("最後に", DiscourseMarker::Conclusion),
];

// ── Korean ─────────────────────────────────────────────────────────

const KOREAN_PATTERNS: &[(&str, DiscourseMarker)] = &[
    // Continuation
    ("게다가 ", DiscourseMarker::Continuation),
    ("또한 ", DiscourseMarker::Continuation),
    ("뿐만 아니라 ", DiscourseMarker::Continuation),
    ("마찬가지로 ", DiscourseMarker::Continuation),
    // Contrast
    ("반면에 ", DiscourseMarker::Contrast),
    ("그럼에도 불구하고 ", DiscourseMarker::Contrast),
    ("그러나 ", DiscourseMarker::Contrast),
    ("하지만 ", DiscourseMarker::Contrast),
    ("그렇지만 ", DiscourseMarker::Contrast),
    // Causation
    ("그 결과 ", DiscourseMarker::Causation),
    ("따라서 ", DiscourseMarker::Causation),
    ("그러므로 ", DiscourseMarker::Causation),
    ("왜냐하면 ", DiscourseMarker::Causation),
    // Exemplification
    ("예를 들어 ", DiscourseMarker::Exemplification),
    ("예컨대 ", DiscourseMarker::Exemplification),
    // Elaboration
    ("구체적으로 ", DiscourseMarker::Elaboration),
    ("특히 ", DiscourseMarker::Elaboration),
    ("즉 ", DiscourseMarker::Elaboration),
    // Conclusion
    ("결론적으로 ", DiscourseMarker::Conclusion),
    ("요약하면 ", DiscourseMarker::Conclusion),
    ("마지막으로 ", DiscourseMarker::Conclusion),
];

// ── Arabic ─────────────────────────────────────────────────────────

const ARABIC_PATTERNS: &[(&str, DiscourseMarker)] = &[
    // Continuation
    ("بالإضافة إلى ذلك،", DiscourseMarker::Continuation),
    ("علاوة على ذلك،", DiscourseMarker::Continuation),
    ("فضلاً عن ذلك،", DiscourseMarker::Continuation),
    ("كذلك ", DiscourseMarker::Continuation),
    ("أيضاً ", DiscourseMarker::Continuation),
    // Contrast
    ("على العكس من ذلك،", DiscourseMarker::Contrast),
    ("في المقابل،", DiscourseMarker::Contrast),
    ("مع ذلك،", DiscourseMarker::Contrast),
    ("ومع ذلك،", DiscourseMarker::Contrast),
    ("لكن ", DiscourseMarker::Contrast),
    ("إلا أن ", DiscourseMarker::Contrast),
    ("غير أن ", DiscourseMarker::Contrast),
    // Causation
    ("نتيجة لذلك،", DiscourseMarker::Causation),
    ("بالتالي،", DiscourseMarker::Causation),
    ("لذلك،", DiscourseMarker::Causation),
    ("لذلك ", DiscourseMarker::Causation),
    ("لأن ", DiscourseMarker::Causation),
    ("بسبب ", DiscourseMarker::Causation),
    // Exemplification
    ("على سبيل المثال،", DiscourseMarker::Exemplification),
    ("مثلاً،", DiscourseMarker::Exemplification),
    ("مثلاً ", DiscourseMarker::Exemplification),
    // Elaboration
    ("بشكل خاص،", DiscourseMarker::Elaboration),
    ("تحديداً،", DiscourseMarker::Elaboration),
    ("أي أن ", DiscourseMarker::Elaboration),
    // Conclusion
    ("في الختام،", DiscourseMarker::Conclusion),
    ("خلاصة القول،", DiscourseMarker::Conclusion),
    ("باختصار،", DiscourseMarker::Conclusion),
    ("أخيراً،", DiscourseMarker::Conclusion),
    ("أخيراً ", DiscourseMarker::Conclusion),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_english_discourse() {
        let markers = detect_discourse_markers_multilingual(
            "Furthermore, the system...",
            LanguageGroup::English,
        );
        assert_eq!(markers, vec![DiscourseMarker::Continuation]);
    }

    #[test]
    fn test_german_discourse() {
        let markers = detect_discourse_markers_multilingual(
            "Darüber hinaus unterstützt das System...",
            LanguageGroup::German,
        );
        assert_eq!(markers, vec![DiscourseMarker::Continuation]);
    }

    #[test]
    fn test_french_contrast() {
        let markers = detect_discourse_markers_multilingual(
            "Cependant, cette approche a des limites.",
            LanguageGroup::French,
        );
        assert_eq!(markers, vec![DiscourseMarker::Contrast]);
    }

    #[test]
    fn test_spanish_causation() {
        let markers = detect_discourse_markers_multilingual(
            "Por lo tanto, elegimos una estrategia diferente.",
            LanguageGroup::Spanish,
        );
        assert_eq!(markers, vec![DiscourseMarker::Causation]);
    }

    #[test]
    fn test_russian_exemplification() {
        let markers = detect_discourse_markers_multilingual(
            "Например, CogniGraph Chunker обрабатывает...",
            LanguageGroup::Russian,
        );
        assert_eq!(markers, vec![DiscourseMarker::Exemplification]);
    }

    #[test]
    fn test_japanese_continuation() {
        let markers = detect_discourse_markers_multilingual(
            "さらに、このシステムは...",
            LanguageGroup::Japanese,
        );
        assert_eq!(markers, vec![DiscourseMarker::Continuation]);
    }

    #[test]
    fn test_chinese_contrast() {
        let markers = detect_discourse_markers_multilingual(
            "然而，这种方法有局限性。",
            LanguageGroup::Chinese,
        );
        assert_eq!(markers, vec![DiscourseMarker::Contrast]);
    }

    #[test]
    fn test_korean_causation() {
        let markers = detect_discourse_markers_multilingual(
            "따라서 우리는 다른 전략을 선택했다.",
            LanguageGroup::Korean,
        );
        assert_eq!(markers, vec![DiscourseMarker::Causation]);
    }

    #[test]
    fn test_arabic_continuation() {
        let markers = detect_discourse_markers_multilingual(
            "بالإضافة إلى ذلك، يدعم النظام...",
            LanguageGroup::Arabic,
        );
        assert_eq!(markers, vec![DiscourseMarker::Continuation]);
    }

    #[test]
    fn test_unknown_language_no_markers() {
        let markers =
            detect_discourse_markers_multilingual("Xyz abc def ghi.", LanguageGroup::Other);
        assert!(markers.is_empty());
    }
}
