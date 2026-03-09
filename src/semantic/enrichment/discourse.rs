//! Discourse marker detection.
//!
//! Classifies leading phrases that signal rhetorical relationships
//! between adjacent blocks: continuation, contrast, causation, etc.

use super::super::cognitive_types::DiscourseMarker;

/// Match table: (prefix, marker type).
/// Ordered longest-first within each group to avoid partial matches.
const DISCOURSE_PATTERNS: &[(&str, DiscourseMarker)] = &[
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

/// Detect discourse markers at the start of a text block.
///
/// Returns all matching markers (usually zero or one).
pub fn detect_discourse_markers(text: &str) -> Vec<DiscourseMarker> {
    let lower = text.trim_start().to_lowercase();
    let mut markers = Vec::new();

    for &(pattern, marker) in DISCOURSE_PATTERNS {
        if lower.starts_with(pattern) {
            markers.push(marker);
            break; // One match is sufficient — they're mutually exclusive at block start
        }
    }

    markers
}

/// Score how strongly a discourse marker signals continuation (0.0–1.0).
///
/// Higher = stronger signal to keep blocks together.
pub fn discourse_continuation_score(markers: &[DiscourseMarker]) -> f64 {
    if markers.is_empty() {
        return 0.0;
    }

    markers
        .iter()
        .map(|m| match m {
            // Strong continuation signals
            DiscourseMarker::Continuation => 0.9,
            DiscourseMarker::Elaboration => 0.85,
            DiscourseMarker::Exemplification => 0.8,
            DiscourseMarker::Causation => 0.7,
            // Contrast can signal either continuation or break
            DiscourseMarker::Contrast => 0.4,
            // Conclusion signals a winding-down
            DiscourseMarker::Conclusion => 0.2,
        })
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_continuation() {
        let markers = detect_discourse_markers("Furthermore, the system also supports...");
        assert_eq!(markers, vec![DiscourseMarker::Continuation]);
    }

    #[test]
    fn test_detect_contrast() {
        let markers = detect_discourse_markers("However, this approach has limitations.");
        assert_eq!(markers, vec![DiscourseMarker::Contrast]);
    }

    #[test]
    fn test_detect_causation() {
        let markers = detect_discourse_markers("Therefore, we chose a different strategy.");
        assert_eq!(markers, vec![DiscourseMarker::Causation]);
    }

    #[test]
    fn test_detect_exemplification() {
        let markers = detect_discourse_markers("For example, the CogniGraph Chunker...");
        assert_eq!(markers, vec![DiscourseMarker::Exemplification]);
    }

    #[test]
    fn test_detect_conclusion() {
        let markers = detect_discourse_markers("In summary, the results demonstrate...");
        assert_eq!(markers, vec![DiscourseMarker::Conclusion]);
    }

    #[test]
    fn test_no_marker() {
        let markers = detect_discourse_markers("The CogniGraph Chunker processes text.");
        assert!(markers.is_empty());
    }

    #[test]
    fn test_continuation_score() {
        assert!(discourse_continuation_score(&[DiscourseMarker::Continuation]) > 0.8);
        assert!(discourse_continuation_score(&[DiscourseMarker::Contrast]) < 0.5);
        assert_eq!(discourse_continuation_score(&[]), 0.0);
    }
}
