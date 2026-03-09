//! Diagnostic output for cognitive chunking signals.

use super::cognitive_types::BoundarySignal;

/// Emit boundary signals to stderr in TSV format for debugging.
pub fn emit_signals_tsv(signals: &[BoundarySignal]) {
    eprintln!("--- cognitive boundary signals ---");
    eprintln!(
        "idx\tsem_sim\tent_cont\trel_cont\tdisc_cont\thead_cont\tstruct_aff\ttopic_shift\torphan\tbudget\tjoin\tbreak\treasons"
    );
    for s in signals {
        eprintln!(
            "{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}",
            s.index,
            s.semantic_similarity,
            s.entity_continuity,
            s.relation_continuity,
            s.discourse_continuation,
            s.heading_continuity,
            s.structural_affinity,
            s.topic_shift_penalty,
            s.orphan_risk,
            s.budget_pressure,
            s.join_score,
            if s.is_break { "BREAK" } else { "join" },
            s.reasons.join("; "),
        );
    }
    eprintln!("--- end ---");
}

/// Serialize boundary signals as JSON array for API output.
pub fn signals_to_json(signals: &[BoundarySignal]) -> Vec<serde_json::Value> {
    signals
        .iter()
        .map(|s| {
            serde_json::json!({
                "boundary_index": s.index,
                "scores": {
                    "semantic_similarity": round4(s.semantic_similarity),
                    "entity_continuity": round4(s.entity_continuity),
                    "relation_continuity": round4(s.relation_continuity),
                    "discourse_continuation": round4(s.discourse_continuation),
                    "heading_continuity": round4(s.heading_continuity),
                    "structural_affinity": round4(s.structural_affinity),
                    "topic_shift_penalty": round4(s.topic_shift_penalty),
                    "orphan_risk": round4(s.orphan_risk),
                    "budget_pressure": round4(s.budget_pressure),
                },
                "join_score": round4(s.join_score),
                "decision": if s.is_break { "break" } else { "join" },
                "reasons": s.reasons,
            })
        })
        .collect()
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}
