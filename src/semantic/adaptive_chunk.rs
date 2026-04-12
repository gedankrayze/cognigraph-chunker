//! Adaptive chunking meta-router.
//!
//! Evaluates multiple candidate chunking methods via intrinsic quality metrics
//! and returns the winner's chunks along with a comparative report.

use anyhow::Result;

use crate::embeddings::EmbeddingProvider;
use crate::llm::CompletionClient;

use super::adaptive_types::{AdaptiveReport, AdaptiveResult, CandidateScore, ScreeningDecision};
use super::blocks::{BlockKind, split_blocks};
use super::cognitive_types::{CognitiveConfig, CognitiveWeights};
use super::enriched_chunk::{EnrichedConfig, enriched_chunk};
use super::intent_chunk::{IntentConfig, intent_chunk};
use super::quality_metrics::{ChunkForEval, MetricConfig, MetricWeights, evaluate_chunks};
use super::topo_chunk::{TopoConfig, topo_chunk};
use super::{SemanticConfig, cognitive_chunk, semantic_chunk};

/// Configuration for the adaptive chunking router.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Method names to try (e.g. ["semantic", "cognitive", "intent"]).
    pub candidates: Vec<String>,
    /// If true, bypass pre-screening heuristics and run all listed candidates.
    pub force_candidates: bool,
    /// Soft token budget per chunk.
    pub soft_budget: usize,
    /// Hard token ceiling per chunk.
    pub hard_budget: usize,
    /// Weights for the quality metric composite score.
    pub metric_weights: MetricWeights,
    /// Window size for cross-similarity (semantic/cognitive).
    pub sim_window: usize,
    /// Savitzky-Golay smoothing window (semantic/cognitive).
    pub sg_window: usize,
    /// Savitzky-Golay polynomial order (semantic/cognitive).
    pub poly_order: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            candidates: vec!["semantic".to_string(), "cognitive".to_string()],
            force_candidates: false,
            soft_budget: 512,
            hard_budget: 768,
            metric_weights: MetricWeights::default(),
            sim_window: 3,
            sg_window: 11,
            poly_order: 3,
        }
    }
}

/// Estimate token count using whitespace splitting.
fn token_estimate(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Count distinct heading levels in a document (using split_blocks).
fn count_heading_levels(text: &str) -> usize {
    let blocks = split_blocks(text);
    let mut levels = std::collections::HashSet::new();
    for block in &blocks {
        if block.kind == BlockKind::Heading {
            let trimmed = block.text.trim();
            let level = trimmed.chars().take_while(|&c| c == '#').count();
            if level > 0 {
                levels.insert(level);
            }
        }
    }
    levels.len()
}

/// Check whether a document has any markdown structure (headings, lists, tables, code blocks).
fn has_markdown_structure(text: &str) -> bool {
    let blocks = split_blocks(text);
    blocks.iter().any(|b| {
        matches!(
            b.kind,
            BlockKind::Heading | BlockKind::Table | BlockKind::CodeBlock | BlockKind::List
        )
    })
}

/// Pre-screen candidates based on document characteristics.
///
/// Returns a list of screening decisions and the filtered candidate names.
pub fn pre_screen(
    text: &str,
    candidates: &[String],
    force: bool,
) -> (Vec<ScreeningDecision>, Vec<String>) {
    let doc_tokens = token_estimate(text);
    let heading_levels = count_heading_levels(text);
    let has_structure = has_markdown_structure(text);

    let mut decisions = Vec::new();
    let mut included = Vec::new();

    for method in candidates {
        let (include, reason) = if force {
            (true, "force_candidates enabled".to_string())
        } else {
            match method.as_str() {
                "semantic" => (true, "always included".to_string()),
                "cognitive" => (true, "always included".to_string()),
                "topo" => {
                    if heading_levels < 2 {
                        (
                            false,
                            format!(
                                "skipped: document has {} heading level(s), topo needs >= 2",
                                heading_levels
                            ),
                        )
                    } else {
                        (true, format!("{heading_levels} heading levels detected"))
                    }
                }
                "intent" => {
                    if doc_tokens < 500 {
                        (
                            false,
                            format!(
                                "skipped: document has {doc_tokens} tokens, intent needs >= 500"
                            ),
                        )
                    } else {
                        (true, format!("{doc_tokens} tokens, sufficient for intent"))
                    }
                }
                "enriched" => {
                    if !has_structure && doc_tokens < 1000 {
                        (
                            false,
                            format!(
                                "skipped: no markdown structure and {doc_tokens} < 1000 tokens"
                            ),
                        )
                    } else {
                        (
                            true,
                            "document has structure or sufficient length".to_string(),
                        )
                    }
                }
                other => (false, format!("unknown method: {other}")),
            }
        };

        decisions.push(ScreeningDecision {
            method: method.clone(),
            included: include,
            reason,
        });

        if include {
            included.push(method.clone());
        }
    }

    (decisions, included)
}

/// Run the adaptive chunking meta-router.
///
/// Evaluates each screened candidate method, scores their output using quality
/// metrics, and returns the winner's chunks with a full comparison report.
///
/// If `llm_client` is None, LLM-based methods (intent, enriched, topo) are
/// automatically excluded even if listed in `config.candidates`.
pub async fn adaptive_chunk<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    llm_client: Option<&CompletionClient>,
    config: &AdaptiveConfig,
) -> Result<AdaptiveResult> {
    // Filter out LLM methods if no client is available
    let effective_candidates: Vec<String> = config
        .candidates
        .iter()
        .filter(|m| {
            if llm_client.is_none() {
                !matches!(m.as_str(), "intent" | "enriched" | "topo")
            } else {
                true
            }
        })
        .cloned()
        .collect();

    // Pre-screen
    let (screening_decisions, screened) =
        pre_screen(text, &effective_candidates, config.force_candidates);

    if screened.is_empty() {
        anyhow::bail!("No candidate methods passed pre-screening");
    }

    let metric_config = MetricConfig {
        soft_budget: config.soft_budget,
        hard_budget: config.hard_budget,
        weights: config.metric_weights.clone(),
    };

    // Run each candidate and evaluate
    let mut scored: Vec<(
        String,
        Vec<ChunkForEval>,
        Vec<serde_json::Value>,
        CandidateScore,
    )> = Vec::new();

    for method in &screened {
        let result = run_candidate(text, method, provider, llm_client, config).await;

        match result {
            Ok((evals, json_chunks)) => {
                let total_tokens: usize = evals.iter().map(|c| token_estimate(&c.text)).sum();
                let chunk_count = evals.len();

                match evaluate_chunks(text, &evals, provider, &metric_config).await {
                    Ok(metrics) => {
                        scored.push((
                            method.clone(),
                            evals,
                            json_chunks,
                            CandidateScore {
                                method: method.clone(),
                                metrics,
                                chunk_count,
                                total_tokens,
                            },
                        ));
                    }
                    Err(e) => {
                        eprintln!("[adaptive] evaluation failed for {method}: {e}");
                    }
                }
            }
            Err(e) => {
                eprintln!("[adaptive] candidate {method} failed: {e}");
            }
        }
    }

    if scored.is_empty() {
        anyhow::bail!("All candidate methods failed");
    }

    // Pick winner: highest composite score, ties broken by fewer chunks
    scored.sort_by(|a, b| {
        b.3.metrics
            .composite
            .partial_cmp(&a.3.metrics.composite)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.3.chunk_count.cmp(&b.3.chunk_count))
    });

    let (winner_name, _evals, winner_chunks, _winner_score) = scored.remove(0);

    // Build candidate scores list (winner + others)
    let mut candidate_scores: Vec<CandidateScore> = Vec::new();
    candidate_scores.push(_winner_score);
    for (_, _, _, score) in scored {
        candidate_scores.push(score);
    }

    let count = winner_chunks.len();

    Ok(AdaptiveResult {
        winner: winner_name,
        chunks: winner_chunks,
        report: AdaptiveReport {
            candidates: candidate_scores,
            pre_screening: screening_decisions,
            metric_weights: config.metric_weights.clone(),
        },
        count,
    })
}

/// Run a single candidate method and return its chunks as both evaluation structs and JSON.
async fn run_candidate<P: EmbeddingProvider>(
    text: &str,
    method: &str,
    provider: &P,
    llm_client: Option<&CompletionClient>,
    config: &AdaptiveConfig,
) -> Result<(Vec<ChunkForEval>, Vec<serde_json::Value>)> {
    match method {
        "semantic" => {
            let sem_config = SemanticConfig {
                sim_window: config.sim_window,
                sg_window: config.sg_window,
                poly_order: config.poly_order,
                ..SemanticConfig::default()
            };
            let result = semantic_chunk(text, provider, &sem_config).await?;
            let evals: Vec<ChunkForEval> = result
                .chunks
                .iter()
                .map(|(chunk_text, offset)| ChunkForEval {
                    text: chunk_text.clone(),
                    offset_start: *offset,
                    offset_end: offset + chunk_text.len(),
                })
                .collect();
            let json_chunks: Vec<serde_json::Value> = result
                .chunks
                .iter()
                .enumerate()
                .map(|(i, (chunk_text, offset))| {
                    serde_json::json!({
                        "index": i,
                        "text": chunk_text,
                        "offset": offset,
                        "length": chunk_text.len(),
                    })
                })
                .collect();
            Ok((evals, json_chunks))
        }
        "cognitive" => {
            let cog_config = CognitiveConfig {
                weights: CognitiveWeights::default(),
                soft_budget: config.soft_budget,
                hard_budget: config.hard_budget,
                sim_window: config.sim_window,
                sg_window: config.sg_window,
                poly_order: config.poly_order,
                max_blocks: 10_000,
                emit_signals: false,
                language: None,
            };
            let result = cognitive_chunk(text, provider, &cog_config).await?;
            let evals: Vec<ChunkForEval> = result
                .chunks
                .iter()
                .map(|c| ChunkForEval {
                    text: c.text.clone(),
                    offset_start: c.offset_start,
                    offset_end: c.offset_end,
                })
                .collect();
            let json_chunks: Vec<serde_json::Value> = result
                .chunks
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    serde_json::json!({
                        "index": i,
                        "text": c.text,
                        "offset_start": c.offset_start,
                        "offset_end": c.offset_end,
                        "token_estimate": c.token_estimate,
                        "heading_path": c.heading_path,
                        "dominant_entities": c.dominant_entities,
                    })
                })
                .collect();
            Ok((evals, json_chunks))
        }
        "intent" => {
            let client =
                llm_client.ok_or_else(|| anyhow::anyhow!("intent method requires LLM client"))?;
            let intent_config = IntentConfig {
                soft_budget: config.soft_budget,
                hard_budget: config.hard_budget,
                ..IntentConfig::default()
            };
            let result = intent_chunk(text, provider, client, &intent_config).await?;
            let evals: Vec<ChunkForEval> = result
                .chunks
                .iter()
                .map(|c| ChunkForEval {
                    text: c.text.clone(),
                    offset_start: c.offset_start,
                    offset_end: c.offset_end,
                })
                .collect();
            let json_chunks: Vec<serde_json::Value> = result
                .chunks
                .iter()
                .map(|c| serde_json::to_value(c).unwrap_or_default())
                .collect();
            Ok((evals, json_chunks))
        }
        "enriched" => {
            let client =
                llm_client.ok_or_else(|| anyhow::anyhow!("enriched method requires LLM client"))?;
            let enriched_config = EnrichedConfig {
                soft_budget: config.soft_budget,
                hard_budget: config.hard_budget,
                ..EnrichedConfig::default()
            };
            let result = enriched_chunk(text, client, &enriched_config).await?;
            let evals: Vec<ChunkForEval> = result
                .chunks
                .iter()
                .map(|c| ChunkForEval {
                    text: c.text.clone(),
                    offset_start: c.offset_start,
                    offset_end: c.offset_end,
                })
                .collect();
            let json_chunks: Vec<serde_json::Value> = result
                .chunks
                .iter()
                .map(|c| serde_json::to_value(c).unwrap_or_default())
                .collect();
            Ok((evals, json_chunks))
        }
        "topo" => {
            let client =
                llm_client.ok_or_else(|| anyhow::anyhow!("topo method requires LLM client"))?;
            let topo_config = TopoConfig {
                soft_budget: config.soft_budget,
                hard_budget: config.hard_budget,
                ..TopoConfig::default()
            };
            let result = topo_chunk(text, client, &topo_config).await?;
            let evals: Vec<ChunkForEval> = result
                .chunks
                .iter()
                .map(|c| ChunkForEval {
                    text: c.text.clone(),
                    offset_start: c.offset_start,
                    offset_end: c.offset_end,
                })
                .collect();
            let json_chunks: Vec<serde_json::Value> = result
                .chunks
                .iter()
                .map(|c| serde_json::to_value(c).unwrap_or_default())
                .collect();
            Ok((evals, json_chunks))
        }
        other => anyhow::bail!("unknown chunking method: {other}"),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_screening_flat_doc() {
        // A document with no headings should skip topo
        let doc = "Just a flat paragraph without any headings.\nAnother sentence here.\nMore content follows.";
        let candidates = vec![
            "semantic".to_string(),
            "cognitive".to_string(),
            "topo".to_string(),
        ];

        let (decisions, included) = pre_screen(doc, &candidates, false);

        // topo should be excluded
        assert!(!included.contains(&"topo".to_string()));
        assert!(included.contains(&"semantic".to_string()));
        assert!(included.contains(&"cognitive".to_string()));

        // Verify the screening decision for topo says excluded
        let topo_decision = decisions.iter().find(|d| d.method == "topo").unwrap();
        assert!(!topo_decision.included);
        assert!(topo_decision.reason.contains("heading level"));
    }

    #[test]
    fn test_pre_screening_short_doc() {
        // A document < 500 tokens should skip intent
        let doc = "Short document. Only a few words here.";
        let candidates = vec![
            "semantic".to_string(),
            "cognitive".to_string(),
            "intent".to_string(),
        ];

        let (decisions, included) = pre_screen(doc, &candidates, false);

        assert!(!included.contains(&"intent".to_string()));
        assert!(included.contains(&"semantic".to_string()));

        let intent_decision = decisions.iter().find(|d| d.method == "intent").unwrap();
        assert!(!intent_decision.included);
        assert!(intent_decision.reason.contains("tokens"));
    }

    #[test]
    fn test_pre_screening_force_candidates() {
        // With force_candidates, everything should be included
        let doc = "Short doc.";
        let candidates = vec![
            "semantic".to_string(),
            "cognitive".to_string(),
            "intent".to_string(),
            "topo".to_string(),
            "enriched".to_string(),
        ];

        let (_decisions, included) = pre_screen(doc, &candidates, true);

        assert_eq!(included.len(), 5);
        for method in &candidates {
            assert!(
                included.contains(method),
                "Expected {method} to be included"
            );
        }
    }

    #[test]
    fn test_pre_screening_enriched_no_structure_short() {
        // A short document with no markdown structure should skip enriched
        let doc = "Plain text with no markdown. Just sentences.";
        let candidates = vec!["enriched".to_string()];

        let (_decisions, included) = pre_screen(doc, &candidates, false);
        assert!(!included.contains(&"enriched".to_string()));
    }

    #[test]
    fn test_pre_screening_enriched_with_structure() {
        // A document with markdown structure should include enriched
        let doc = "# Heading\n\nSome content here.\n\n## Subheading\n\nMore content.";
        let candidates = vec!["enriched".to_string()];

        let (_decisions, included) = pre_screen(doc, &candidates, false);
        assert!(included.contains(&"enriched".to_string()));
    }

    #[test]
    fn test_pre_screening_topo_with_multiple_levels() {
        // A document with >= 2 heading levels should include topo
        let doc = "# Top\n\nIntro.\n\n## Sub\n\nContent.";
        let candidates = vec!["topo".to_string()];

        let (_decisions, included) = pre_screen(doc, &candidates, false);
        assert!(included.contains(&"topo".to_string()));
    }

    #[test]
    fn test_count_heading_levels() {
        let doc = "# H1\n\nText.\n\n## H2\n\nMore.\n\n### H3\n\nDeep.";
        assert_eq!(count_heading_levels(doc), 3);

        let flat = "No headings at all.";
        assert_eq!(count_heading_levels(flat), 0);

        let single = "# Only one level\n\nContent.";
        assert_eq!(count_heading_levels(single), 1);
    }
}
