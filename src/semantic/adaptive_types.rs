//! Data types for adaptive chunking.

use serde::Serialize;

use super::quality_metrics::{MetricWeights, QualityMetrics};

/// Result of the adaptive chunking meta-router.
#[derive(Debug, Serialize)]
pub struct AdaptiveResult {
    /// Name of the winning method.
    pub winner: String,
    /// The winner's chunks serialized as JSON values (polymorphic across method types).
    pub chunks: Vec<serde_json::Value>,
    /// Quality evaluation report for all candidates.
    pub report: AdaptiveReport,
    /// Number of chunks produced by the winner.
    pub count: usize,
}

/// Full report of the adaptive selection process.
#[derive(Debug, Serialize)]
pub struct AdaptiveReport {
    /// Scores for each candidate that was evaluated.
    pub candidates: Vec<CandidateScore>,
    /// Pre-screening decisions for all considered methods.
    pub pre_screening: Vec<ScreeningDecision>,
    /// Metric weights used for composite scoring.
    pub metric_weights: MetricWeights,
}

/// Quality metrics and metadata for a single candidate method.
#[derive(Debug, Serialize)]
pub struct CandidateScore {
    /// Method name (e.g. "semantic", "cognitive").
    pub method: String,
    /// Quality metrics computed by evaluate_chunks.
    pub metrics: QualityMetrics,
    /// Number of chunks produced.
    pub chunk_count: usize,
    /// Sum of estimated token counts across all chunks.
    pub total_tokens: usize,
}

/// Pre-screening decision for a candidate method.
#[derive(Debug, Serialize)]
pub struct ScreeningDecision {
    /// Method name.
    pub method: String,
    /// Whether the method was included as a candidate.
    pub included: bool,
    /// Human-readable reason for inclusion or exclusion.
    pub reason: String,
}
