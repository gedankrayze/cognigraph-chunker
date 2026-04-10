# New Chunking Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 new chunking methods (intent, topo, enriched, adaptive) + a standalone quality metrics module as first-class CLI subcommands and API endpoints.

**Architecture:** Flat peer methods following the existing pattern — each method gets a CLI subcommand (`src/cli/{method}_cmd.rs`), API handler (`src/api/{method}.rs`), core logic (`src/semantic/{method}_chunk.rs`), types file (`src/semantic/{method}_types.rs`), and LLM prompts (`src/llm/{method}.rs`). Registration in `src/main.rs` (Commands enum + match), `src/cli/mod.rs`, `src/api/mod.rs` (router + module), and `src/semantic/mod.rs` (module declaration).

**Tech Stack:** Rust 2024, clap 4.5 (derive), serde/serde_json, tokio, reqwest, axum, existing `EmbeddingProvider` trait, existing `CompletionClient` for LLM calls.

**Spec:** `docs/superpowers/specs/2026-04-10-new-chunking-methods-design.md`

---

## File Map

### Quality Metrics Module (Task 1)
- Create: `src/semantic/quality_metrics.rs` — 5 metric implementations + `evaluate_chunks` function
- Create: `src/api/evaluate.rs` — `POST /api/v1/evaluate` handler
- Modify: `src/semantic/mod.rs` — add `pub mod quality_metrics;`
- Modify: `src/api/mod.rs` — add `pub mod evaluate;` + route
- Test: unit tests inline in `quality_metrics.rs`

### Intent-Driven Chunking (Task 2)
- Create: `src/semantic/intent_types.rs` — IntentResult, IntentChunk, PredictedIntent, IntentType
- Create: `src/semantic/intent_chunk.rs` — DP algorithm, alignment scoring, pipeline orchestration
- Create: `src/llm/intents.rs` — intent generation prompt + JSON schema
- Create: `src/cli/intent_cmd.rs` — CLI subcommand
- Create: `src/api/intent.rs` — API handler
- Modify: `src/semantic/mod.rs` — add module declarations + re-export `intent_chunk`
- Modify: `src/llm/mod.rs` — add `pub mod intents;`
- Modify: `src/cli/mod.rs` — add `pub mod intent_cmd;`
- Modify: `src/api/mod.rs` — add `pub mod intent;` + route
- Modify: `src/main.rs` — add `Intent` variant + match arm
- Test: unit tests inline in `intent_chunk.rs`

### Enriched Chunking (Task 3)
- Create: `src/semantic/enriched_types.rs` — EnrichedResult, EnrichedChunk, TypedEntity, MergeRecord
- Create: `src/semantic/enriched_chunk.rs` — pipeline orchestration, initial grouping, key-based recombination
- Create: `src/llm/enrichment.rs` — 7-field enrichment prompt, JSON schema, rolling key logic
- Create: `src/cli/enriched_cmd.rs` — CLI subcommand
- Create: `src/api/enriched.rs` — API handler
- Modify: `src/semantic/mod.rs` — add module declarations
- Modify: `src/llm/mod.rs` — add `pub mod enrichment;`
- Modify: `src/cli/mod.rs` — add `pub mod enriched_cmd;`
- Modify: `src/api/mod.rs` — add `pub mod enriched;` + route
- Modify: `src/main.rs` — add `Enriched` variant + match arm
- Test: unit tests inline in `enriched_chunk.rs`

### Topology-Aware Chunking (Task 4)
- Create: `src/semantic/sir.rs` — SIR data structures (SirNode, SirEdge, Sir, SirNodeType, SirEdgeType)
- Create: `src/semantic/topo_types.rs` — TopoResult, TopoChunk, SectionClassification, SectionClass
- Create: `src/semantic/topo_chunk.rs` — SIR builder, assembly, pipeline orchestration
- Create: `src/llm/topo_agents.rs` — Inspector + Refiner prompts and JSON schemas
- Create: `src/cli/topo_cmd.rs` — CLI subcommand
- Create: `src/api/topo.rs` — API handler
- Modify: `src/semantic/mod.rs` — add module declarations
- Modify: `src/llm/mod.rs` — add `pub mod topo_agents;`
- Modify: `src/cli/mod.rs` — add `pub mod topo_cmd;`
- Modify: `src/api/mod.rs` — add `pub mod topo;` + route
- Modify: `src/main.rs` — add `Topo` variant + match arm
- Test: unit tests inline in `topo_chunk.rs`

### Adaptive Chunking (Task 5)
- Create: `src/semantic/adaptive_types.rs` — AdaptiveResult, AdaptiveReport, CandidateScore, ScreeningDecision
- Create: `src/semantic/adaptive_chunk.rs` — orchestrator: pre-screening, candidate dispatch, scoring, selection
- Create: `src/cli/adaptive_cmd.rs` — CLI subcommand
- Create: `src/api/adaptive.rs` — API handler
- Modify: `src/semantic/mod.rs` — add module declarations
- Modify: `src/cli/mod.rs` — add `pub mod adaptive_cmd;`
- Modify: `src/api/mod.rs` — add `pub mod adaptive;` + route
- Modify: `src/main.rs` — add `Adaptive` variant + match arm
- Test: unit tests inline in `adaptive_chunk.rs`

---

## Task 1: Quality Metrics Module

The standalone module that scores any chunking output on 5 intrinsic dimensions. Used by Adaptive and available via API for benchmarking.

**Files:**
- Create: `src/semantic/quality_metrics.rs`
- Create: `src/api/evaluate.rs`
- Modify: `src/semantic/mod.rs`
- Modify: `src/api/mod.rs`

### Step 1.1: Write types and size compliance metric

- [ ] **Create `src/semantic/quality_metrics.rs` with core types + SC metric**

```rust
//! Intrinsic quality metrics for evaluating chunking output.
//!
//! Five metrics scored 0.0–1.0:
//! - Size Compliance (SC): fraction of chunks within target token bounds
//! - Intrachunk Cohesion (ICC): semantic unity within each chunk
//! - Contextual Coherence (DCC): smooth transitions between adjacent chunks
//! - Block Integrity (BI): structural elements preserved intact
//! - Reference Completeness (RC): entity-pronoun chains not severed

use std::collections::HashSet;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::embeddings::EmbeddingProvider;

/// Quality scores for a set of chunks (each 0.0–1.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub size_compliance: f64,
    pub intrachunk_cohesion: f64,
    pub contextual_coherence: f64,
    pub block_integrity: f64,
    pub reference_completeness: f64,
    pub composite: f64,
}

/// Weights for combining the five metrics into a composite score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricWeights {
    pub sc: f64,
    pub icc: f64,
    pub dcc: f64,
    pub bi: f64,
    pub rc: f64,
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self {
            sc: 0.20,
            icc: 0.20,
            dcc: 0.20,
            bi: 0.20,
            rc: 0.20,
        }
    }
}

/// A chunk presented for evaluation (minimal interface — works with any method's output).
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkForEval {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
}

/// Configuration for quality evaluation.
#[derive(Debug, Clone)]
pub struct MetricConfig {
    pub soft_budget: usize,
    pub hard_budget: usize,
    pub weights: MetricWeights,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            soft_budget: 512,
            hard_budget: 768,
            weights: MetricWeights::default(),
        }
    }
}

/// Estimate token count for a text (whitespace splitting, same as core::merge).
fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Compute composite score from individual metrics.
pub fn composite_score(metrics: &QualityMetrics, weights: &MetricWeights) -> f64 {
    weights.sc * metrics.size_compliance
        + weights.icc * metrics.intrachunk_cohesion
        + weights.dcc * metrics.contextual_coherence
        + weights.bi * metrics.block_integrity
        + weights.rc * metrics.reference_completeness
}

// ── Size Compliance ────────────────────────────────────────────────

/// SC = count(chunks where soft_budget * 0.5 <= tokens <= hard_budget) / total.
pub fn size_compliance(chunks: &[ChunkForEval], soft_budget: usize, hard_budget: usize) -> f64 {
    if chunks.is_empty() {
        return 1.0;
    }
    let min_tokens = soft_budget / 2;
    let compliant = chunks
        .iter()
        .filter(|c| {
            let tokens = estimate_tokens(&c.text);
            tokens >= min_tokens && tokens <= hard_budget
        })
        .count();
    compliant as f64 / chunks.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_compliance_all_compliant() {
        let chunks = vec![
            ChunkForEval { text: "word ".repeat(300), offset_start: 0, offset_end: 100 },
            ChunkForEval { text: "word ".repeat(500), offset_start: 100, offset_end: 200 },
        ];
        let sc = size_compliance(&chunks, 512, 768);
        assert!((sc - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_size_compliance_too_small() {
        let chunks = vec![
            ChunkForEval { text: "short text".to_string(), offset_start: 0, offset_end: 10 },
            ChunkForEval { text: "word ".repeat(400), offset_start: 10, offset_end: 200 },
        ];
        let sc = size_compliance(&chunks, 512, 768);
        assert!((sc - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_size_compliance_empty() {
        let sc = size_compliance(&[], 512, 768);
        assert!((sc - 1.0).abs() < f64::EPSILON);
    }
}
```

- [ ] **Register module in `src/semantic/mod.rs`**

Add after the existing module declarations (line ~18 in `src/semantic/mod.rs`):

```rust
pub mod quality_metrics;
```

- [ ] **Run `cargo check` to verify compilation**

Run: `cargo check 2>&1 | head -20`
Expected: no errors (warnings about unused imports are OK at this stage)

- [ ] **Run tests**

Run: `cargo test quality_metrics -- --nocapture 2>&1 | tail -20`
Expected: 3 tests pass

- [ ] **Commit**

```bash
git add src/semantic/quality_metrics.rs src/semantic/mod.rs
git commit -m "feat(quality-metrics): add types + size compliance metric"
```

### Step 1.2: Add block integrity metric

- [ ] **Add BI metric to `src/semantic/quality_metrics.rs`**

Append before the `#[cfg(test)]` module:

```rust
// ── Block Integrity ────────────────────────────────────────────────

/// BI = count(structural_elements_fully_contained) / total_structural_elements.
///
/// Structural elements are identified by re-parsing the original text.
/// A structural element is "fully contained" if its byte range falls entirely
/// within a single chunk's [offset_start, offset_end) range.
pub fn block_integrity(
    original_text: &str,
    chunks: &[ChunkForEval],
) -> f64 {
    use super::blocks::{split_blocks, BlockKind};

    let blocks = split_blocks(original_text);
    let structural: Vec<_> = blocks
        .iter()
        .filter(|b| matches!(b.kind, BlockKind::Table | BlockKind::CodeBlock | BlockKind::List | BlockKind::BlockQuote))
        .collect();

    if structural.is_empty() {
        return 1.0; // No structural elements = perfect integrity
    }

    let contained = structural
        .iter()
        .filter(|block| {
            let block_start = block.offset;
            let block_end = block.offset + block.text.len();
            chunks.iter().any(|c| c.offset_start <= block_start && c.offset_end >= block_end)
        })
        .count();

    contained as f64 / structural.len() as f64
}
```

Add test inside the `#[cfg(test)]` module:

```rust
    #[test]
    fn test_block_integrity_no_structural() {
        let text = "Just a paragraph of plain text without any tables or code.";
        let chunks = vec![
            ChunkForEval { text: text.to_string(), offset_start: 0, offset_end: text.len() },
        ];
        let bi = block_integrity(text, &chunks);
        assert!((bi - 1.0).abs() < f64::EPSILON);
    }
```

- [ ] **Run tests**

Run: `cargo test quality_metrics -- --nocapture 2>&1 | tail -20`
Expected: 4 tests pass

- [ ] **Commit**

```bash
git add src/semantic/quality_metrics.rs
git commit -m "feat(quality-metrics): add block integrity metric"
```

### Step 1.3: Add reference completeness metric

- [ ] **Add RC metric to `src/semantic/quality_metrics.rs`**

Append before the `#[cfg(test)]` module:

```rust
// ── Reference Completeness ─────────────────────────────────────────

/// RC = 1.0 - (orphan_count / total_boundary_count).
///
/// An "orphan" is a chunk that starts with a pronoun or demonstrative
/// whose antecedent is likely in the previous chunk (i.e., the split
/// severed a coreference chain).
pub fn reference_completeness(chunks: &[ChunkForEval]) -> f64 {
    if chunks.len() <= 1 {
        return 1.0;
    }

    let boundary_count = chunks.len() - 1;
    let mut orphan_count = 0usize;

    // English pronouns and demonstratives that signal orphan risk
    let orphan_starters: &[&str] = &[
        "it ", "its ", "they ", "them ", "their ", "he ", "him ", "his ",
        "she ", "her ", "this ", "that ", "these ", "those ", "such ",
    ];

    for chunk in chunks.iter().skip(1) {
        let lower = chunk.text.trim_start().to_lowercase();
        if orphan_starters.iter().any(|s| lower.starts_with(s)) {
            orphan_count += 1;
        }
    }

    1.0 - (orphan_count as f64 / boundary_count as f64)
}
```

Add tests:

```rust
    #[test]
    fn test_reference_completeness_no_orphans() {
        let chunks = vec![
            ChunkForEval { text: "Compound XR-7742 inhibits serotonin.".to_string(), offset_start: 0, offset_end: 36 },
            ChunkForEval { text: "The dosing protocol requires titration.".to_string(), offset_start: 36, offset_end: 75 },
        ];
        let rc = reference_completeness(&chunks);
        assert!((rc - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reference_completeness_with_orphan() {
        let chunks = vec![
            ChunkForEval { text: "Compound XR-7742 inhibits serotonin.".to_string(), offset_start: 0, offset_end: 36 },
            ChunkForEval { text: "It also blocks dopamine reuptake.".to_string(), offset_start: 36, offset_end: 68 },
        ];
        let rc = reference_completeness(&chunks);
        assert!((rc - 0.0).abs() < f64::EPSILON); // 1 orphan / 1 boundary = 0
    }
```

- [ ] **Run tests**

Run: `cargo test quality_metrics -- --nocapture 2>&1 | tail -20`
Expected: 6 tests pass

- [ ] **Commit**

```bash
git add src/semantic/quality_metrics.rs
git commit -m "feat(quality-metrics): add reference completeness metric"
```

### Step 1.4: Add embedding-dependent metrics (ICC + DCC) and evaluate_chunks

- [ ] **Add ICC, DCC, and the main `evaluate_chunks` function**

Append before the `#[cfg(test)]` module:

```rust
// ── Intrachunk Cohesion (ICC) ──────────────────────────────────────

/// ICC: mean cosine similarity of sentence embeddings to chunk centroid.
/// Requires an embedding provider.
pub async fn intrachunk_cohesion<P: EmbeddingProvider>(
    chunks: &[ChunkForEval],
    provider: &P,
) -> Result<f64> {
    use super::sentence::split_sentences;

    if chunks.is_empty() {
        return Ok(1.0);
    }

    let mut cohesion_scores = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let sentences: Vec<String> = split_sentences(&chunk.text)
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        if sentences.len() <= 1 {
            cohesion_scores.push(1.0); // Single-sentence chunk is perfectly cohesive
            continue;
        }

        // Embed all sentences + the full chunk text
        let mut texts: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
        texts.push(&chunk.text);

        let embeddings = provider.embed(&texts).await?;
        let chunk_emb = embeddings.last().unwrap();

        let mut sim_sum = 0.0;
        for emb in &embeddings[..embeddings.len() - 1] {
            sim_sum += cosine_similarity(emb, chunk_emb);
        }
        cohesion_scores.push(sim_sum / sentences.len() as f64);
    }

    Ok(cohesion_scores.iter().sum::<f64>() / cohesion_scores.len() as f64)
}

// ── Contextual Coherence (DCC) ─────────────────────────────────────

/// DCC: mean cosine similarity between adjacent chunk embeddings.
pub async fn contextual_coherence<P: EmbeddingProvider>(
    chunks: &[ChunkForEval],
    provider: &P,
) -> Result<f64> {
    if chunks.len() <= 1 {
        return Ok(1.0);
    }

    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let embeddings = provider.embed(&texts).await?;

    let mut sim_sum = 0.0;
    for i in 0..embeddings.len() - 1 {
        sim_sum += cosine_similarity(&embeddings[i], &embeddings[i + 1]);
    }

    Ok(sim_sum / (embeddings.len() - 1) as f64)
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ── Main evaluation function ───────────────────────────────────────

/// Evaluate a set of chunks against the original text using all 5 metrics.
pub async fn evaluate_chunks<P: EmbeddingProvider>(
    original_text: &str,
    chunks: &[ChunkForEval],
    provider: &P,
    config: &MetricConfig,
) -> Result<QualityMetrics> {
    let sc = size_compliance(chunks, config.soft_budget, config.hard_budget);
    let bi = block_integrity(original_text, chunks);
    let rc = reference_completeness(chunks);
    let icc = intrachunk_cohesion(chunks, provider).await?;
    let dcc = contextual_coherence(chunks, provider).await?;

    let mut metrics = QualityMetrics {
        size_compliance: sc,
        intrachunk_cohesion: icc,
        contextual_coherence: dcc,
        block_integrity: bi,
        reference_completeness: rc,
        composite: 0.0,
    };
    metrics.composite = composite_score(&metrics, &config.weights);
    Ok(metrics)
}
```

- [ ] **Run `cargo check`**

Run: `cargo check 2>&1 | head -20`
Expected: compiles (ICC/DCC are async, tested via integration tests with mock provider)

- [ ] **Commit**

```bash
git add src/semantic/quality_metrics.rs
git commit -m "feat(quality-metrics): add ICC, DCC metrics + evaluate_chunks function"
```

### Step 1.5: Add /api/v1/evaluate endpoint

- [ ] **Create `src/api/evaluate.rs`**

```rust
//! POST /api/v1/evaluate handler.
//!
//! Accepts pre-chunked output and returns quality metrics.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use crate::embeddings::ollama::OllamaProvider;
use crate::embeddings::openai::OpenAiProvider;
use crate::embeddings::onnx::OnnxProvider;
use crate::embeddings::cloudflare::{CloudflareProvider, resolve_cloudflare_credentials};
use crate::embeddings::oauth::{OAuthProvider, resolve_oauth_credentials};
use crate::embeddings::EmbeddingProvider;
use crate::semantic::quality_metrics::{
    ChunkForEval, MetricConfig, MetricWeights, QualityMetrics, evaluate_chunks,
};

use super::AppState;
use super::errors::ApiError;
use super::semantic::{ProviderParam, validate_base_url};

fn default_provider() -> ProviderParam {
    ProviderParam::Ollama
}
fn default_soft_budget() -> usize {
    512
}
fn default_hard_budget() -> usize {
    768
}

#[derive(Debug, Deserialize)]
pub struct EvaluateRequest {
    /// The original document text.
    pub text: String,
    /// Pre-chunked output to evaluate.
    pub chunks: Vec<ChunkForEval>,
    #[serde(default = "default_provider")]
    pub provider: ProviderParam,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model_path: Option<String>,
    pub cf_auth_token: Option<String>,
    pub cf_account_id: Option<String>,
    pub cf_ai_gateway: Option<String>,
    pub oauth_token_url: Option<String>,
    pub oauth_client_id: Option<String>,
    pub oauth_client_secret: Option<String>,
    pub oauth_scope: Option<String>,
    pub oauth_base_url: Option<String>,
    #[serde(default)]
    pub danger_accept_invalid_certs: bool,
    #[serde(default = "default_soft_budget")]
    pub soft_budget: usize,
    #[serde(default = "default_hard_budget")]
    pub hard_budget: usize,
    pub metric_weights: Option<MetricWeights>,
}

#[derive(Serialize)]
pub struct EvaluateResponse {
    pub metrics: QualityMetrics,
    pub chunk_count: usize,
}

pub async fn evaluate_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EvaluateRequest>,
) -> Result<Json<EvaluateResponse>, ApiError> {
    if let Some(ref base_url) = req.base_url {
        validate_base_url(base_url, state.allow_private_urls)?;
    }

    let config = MetricConfig {
        soft_budget: req.soft_budget,
        hard_budget: req.hard_budget,
        weights: req.metric_weights.unwrap_or_default(),
    };

    let metrics = match req.provider {
        ProviderParam::Ollama => {
            let provider = OllamaProvider::new(req.base_url.clone(), req.model.clone())?;
            evaluate_chunks(&req.text, &req.chunks, &provider, &config).await?
        }
        ProviderParam::Openai => {
            let api_key = resolve_openai_key(&req.api_key)?;
            let provider = OpenAiProvider::new(api_key, req.base_url.clone(), req.model.clone())?;
            evaluate_chunks(&req.text, &req.chunks, &provider, &config).await?
        }
        ProviderParam::Onnx => {
            let model_path = req
                .model_path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("model_path is required for onnx provider"))?;
            let provider = OnnxProvider::new(model_path)?;
            evaluate_chunks(&req.text, &req.chunks, &provider, &config).await?
        }
        ProviderParam::Cloudflare => {
            let (token, account_id, gateway) = resolve_cloudflare_credentials(
                &req.cf_auth_token,
                &req.cf_account_id,
                &req.cf_ai_gateway,
            )?;
            let provider = CloudflareProvider::new(token, account_id, req.model.clone(), gateway)?;
            provider.verify_token().await?;
            evaluate_chunks(&req.text, &req.chunks, &provider, &config).await?
        }
        ProviderParam::Oauth => {
            let creds = resolve_oauth_credentials(
                &req.oauth_token_url,
                &req.oauth_client_id,
                &req.oauth_client_secret,
                &req.oauth_scope,
                &req.oauth_base_url,
                &req.model,
            )?;
            let provider = OAuthProvider::new(
                creds.token_url,
                creds.client_id,
                creds.client_secret,
                creds.scope,
                creds.base_url,
                creds.model,
                req.danger_accept_invalid_certs,
            )?;
            provider.verify_credentials().await?;
            evaluate_chunks(&req.text, &req.chunks, &provider, &config).await?
        }
    };

    let chunk_count = req.chunks.len();
    Ok(Json(EvaluateResponse {
        metrics,
        chunk_count,
    }))
}

fn resolve_openai_key(flag: &Option<String>) -> anyhow::Result<String> {
    if let Some(key) = flag {
        return Ok(key.clone());
    }
    if let Ok(key) = std::env::var("OPENAI_API_KEY")
        && !key.is_empty()
    {
        return Ok(key);
    }
    if let Ok(content) = std::fs::read_to_string(".env.openai") {
        for line in content.lines() {
            let line = line.trim();
            if let Some(val) = line.strip_prefix("OPENAI_API_KEY=") {
                let val = val.trim();
                if !val.is_empty() {
                    return Ok(val.to_string());
                }
            }
        }
    }
    anyhow::bail!("OpenAI API key not found.")
}
```

- [ ] **Register in `src/api/mod.rs`**

Add `pub mod evaluate;` after the existing module declarations (after line 9 `pub mod types;`).

Add route to the router (after the `/api/v1/merge` route, before `.layer(...)`):

```rust
        .route(
            "/api/v1/evaluate",
            axum::routing::post(evaluate::evaluate_handler),
        )
```

- [ ] **Run `cargo check`**

Run: `cargo check 2>&1 | head -20`
Expected: compiles

- [ ] **Commit**

```bash
git add src/api/evaluate.rs src/api/mod.rs
git commit -m "feat(quality-metrics): add POST /api/v1/evaluate endpoint"
```

---

## Task 2: Intent-Driven Chunking

LLM predicts user queries → DP finds globally optimal chunk boundaries maximizing query-chunk alignment.

**Files:**
- Create: `src/semantic/intent_types.rs`
- Create: `src/llm/intents.rs`
- Create: `src/semantic/intent_chunk.rs`
- Create: `src/cli/intent_cmd.rs`
- Create: `src/api/intent.rs`
- Modify: `src/semantic/mod.rs`, `src/llm/mod.rs`, `src/cli/mod.rs`, `src/api/mod.rs`, `src/main.rs`

### Step 2.1: Create intent types

- [ ] **Create `src/semantic/intent_types.rs`**

```rust
//! Data types for intent-driven chunking.

use serde::{Deserialize, Serialize};

/// The type of information need a query represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntentType {
    Factual,
    Procedural,
    Conceptual,
    Comparative,
}

/// A predicted user query generated by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedIntent {
    pub query: String,
    pub intent_type: IntentType,
    /// Chunk indices that best align with this intent (populated after assembly).
    #[serde(default)]
    pub matched_chunks: Vec<usize>,
}

/// A single chunk produced by intent-driven chunking.
#[derive(Debug, Clone, Serialize)]
pub struct IntentChunk {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub token_estimate: usize,
    /// Index into the intents vector for the best-matching intent.
    pub best_intent: usize,
    /// Cosine similarity between chunk centroid and best intent embedding.
    pub alignment_score: f64,
    /// Heading ancestry path at the start of this chunk.
    pub heading_path: Vec<String>,
}

/// Result of intent-driven chunking.
#[derive(Debug)]
pub struct IntentResult {
    pub chunks: Vec<IntentChunk>,
    pub intents: Vec<PredictedIntent>,
    /// Mean alignment score across all chunks.
    pub partition_score: f64,
    /// Number of blocks processed.
    pub block_count: usize,
}
```

- [ ] **Register in `src/semantic/mod.rs`**

Add after the existing module declarations:

```rust
pub mod intent_types;
```

- [ ] **Run `cargo check`**

Run: `cargo check 2>&1 | head -20`
Expected: compiles

- [ ] **Commit**

```bash
git add src/semantic/intent_types.rs src/semantic/mod.rs
git commit -m "feat(intent): add intent chunking data types"
```

### Step 2.2: Create LLM intent generation

- [ ] **Create `src/llm/intents.rs`**

```rust
//! LLM-based intent generation for intent-driven chunking.
//!
//! Generates predicted user queries that represent likely information needs
//! for a given document.

use anyhow::{Context, Result};
use serde::Deserialize;

use super::CompletionClient;
use crate::semantic::intent_types::{IntentType, PredictedIntent};

#[derive(Deserialize)]
struct IntentResponse {
    intents: Vec<IntentEntry>,
}

#[derive(Deserialize)]
struct IntentEntry {
    query: String,
    intent_type: IntentType,
}

const SYSTEM_PROMPT: &str = "\
You are an information needs prediction engine. Given a document, generate diverse \
user queries that someone might search for when looking for information in this document.

Rules:
- Generate queries that represent realistic information needs
- Cover different aspects of the document (not just the introduction)
- Include a mix of factual, procedural, conceptual, and comparative queries
- Queries should be specific enough to match a single section or paragraph
- Each query should be self-contained (understandable without the document)
- Avoid duplicate or near-duplicate queries
- Prioritize queries that would benefit from precise chunk retrieval";

fn json_schema(max_intents: usize) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "intents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A realistic user query about this document"
                        },
                        "intent_type": {
                            "type": "string",
                            "enum": ["factual", "procedural", "conceptual", "comparative"],
                            "description": "The type of information need"
                        }
                    },
                    "required": ["query", "intent_type"],
                    "additionalProperties": false
                },
                "maxItems": max_intents
            }
        },
        "required": ["intents"],
        "additionalProperties": false
    })
}

/// Generate predicted user intents for a document.
pub async fn generate_intents(
    client: &CompletionClient,
    text: &str,
    max_intents: usize,
) -> Result<Vec<PredictedIntent>> {
    let user_prompt = format!(
        "Generate up to {max_intents} diverse user queries for this document:\n\n{text}"
    );

    let response = client
        .complete_json(SYSTEM_PROMPT, &user_prompt, json_schema(max_intents))
        .await
        .context("Intent generation failed")?;

    let parsed: IntentResponse =
        serde_json::from_str(&response).context("Failed to parse intent response")?;

    Ok(parsed
        .intents
        .into_iter()
        .map(|e| PredictedIntent {
            query: e.query,
            intent_type: e.intent_type,
            matched_chunks: vec![],
        })
        .collect())
}
```

- [ ] **Register in `src/llm/mod.rs`**

Add after the existing module declarations (line 7):

```rust
pub mod intents;
```

- [ ] **Run `cargo check`**

Run: `cargo check 2>&1 | head -20`
Expected: compiles

- [ ] **Commit**

```bash
git add src/llm/intents.rs src/llm/mod.rs
git commit -m "feat(intent): add LLM intent generation"
```

### Step 2.3: Create intent chunking pipeline (DP + alignment scoring)

- [ ] **Create `src/semantic/intent_chunk.rs`**

```rust
//! Intent-driven chunking pipeline.
//!
//! Pipeline: blocks → LLM intent generation → embed blocks + intents →
//!           DP alignment optimization → IntentResult

use anyhow::{Result, bail};

use crate::embeddings::EmbeddingProvider;
use crate::llm::CompletionClient;
use crate::llm::intents::generate_intents;

use super::blocks::{split_blocks, BlockKind};
use super::enrichment::heading_context::compute_heading_paths;
use super::intent_types::{IntentChunk, IntentResult, PredictedIntent};
use super::sentence::split_sentences;

/// Configuration for intent-driven chunking.
#[derive(Debug, Clone)]
pub struct IntentConfig {
    /// Maximum number of intents to generate.
    pub max_intents: usize,
    /// Soft token budget per chunk (preferred minimum).
    pub soft_budget: usize,
    /// Hard token ceiling per chunk.
    pub hard_budget: usize,
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            max_intents: 20,
            soft_budget: 512,
            hard_budget: 768,
        }
    }
}

/// Estimate token count (whitespace splitting).
fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Compute centroid embedding (mean of block embeddings in range).
fn centroid(embeddings: &[Vec<f64>], start: usize, end: usize) -> Vec<f64> {
    let count = end - start;
    if count == 0 {
        return vec![];
    }
    let dim = embeddings[start].len();
    let mut result = vec![0.0; dim];
    for emb in &embeddings[start..end] {
        for (i, val) in emb.iter().enumerate() {
            result[i] += val;
        }
    }
    for val in &mut result {
        *val /= count as f64;
    }
    result
}

/// Best alignment score for a chunk (max cosine sim to any intent embedding).
fn chunk_alignment(
    block_embeddings: &[Vec<f64>],
    intent_embeddings: &[Vec<f64>],
    start: usize,
    end: usize,
) -> (f64, usize) {
    let cent = centroid(block_embeddings, start, end);
    if cent.is_empty() || intent_embeddings.is_empty() {
        return (0.0, 0);
    }
    let mut best_score = f64::NEG_INFINITY;
    let mut best_idx = 0;
    for (i, intent_emb) in intent_embeddings.iter().enumerate() {
        let sim = cosine_similarity(&cent, intent_emb);
        if sim > best_score {
            best_score = sim;
            best_idx = i;
        }
    }
    (best_score, best_idx)
}

/// Run the intent-driven chunking pipeline on markdown text.
pub async fn intent_chunk<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    llm_client: &CompletionClient,
    config: &IntentConfig,
) -> Result<IntentResult> {
    let blocks = split_blocks(text);
    intent_chunk_from_blocks(text, &blocks, provider, llm_client, config).await
}

/// Run the intent-driven chunking pipeline on plain text.
pub async fn intent_chunk_plain<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    llm_client: &CompletionClient,
    config: &IntentConfig,
) -> Result<IntentResult> {
    let sentences = split_sentences(text);
    let blocks: Vec<super::blocks::Block<'_>> = sentences
        .iter()
        .map(|s| super::blocks::Block {
            text: s,
            offset: s.as_ptr() as usize - text.as_ptr() as usize,
            kind: BlockKind::Sentence,
        })
        .collect();
    intent_chunk_from_blocks(text, &blocks, provider, llm_client, config).await
}

async fn intent_chunk_from_blocks<'a, P: EmbeddingProvider>(
    _text: &str,
    blocks: &[super::blocks::Block<'a>],
    provider: &P,
    llm_client: &CompletionClient,
    config: &IntentConfig,
) -> Result<IntentResult> {
    if blocks.is_empty() {
        return Ok(IntentResult {
            chunks: vec![],
            intents: vec![],
            partition_score: 0.0,
            block_count: 0,
        });
    }

    let block_count = blocks.len();

    // Step 1: Compute heading paths
    let (heading_paths, _heading_terms) = compute_heading_paths(blocks);

    // Step 2: Generate intents from full text
    let full_text: String = blocks.iter().map(|b| b.text).collect::<Vec<_>>().join("\n");
    let mut intents = generate_intents(llm_client, &full_text, config.max_intents).await?;

    if intents.is_empty() {
        bail!("LLM generated no intents for the document");
    }

    // Step 3: Embed blocks
    let block_texts: Vec<&str> = blocks.iter().map(|b| b.text).collect();
    let block_embeddings = provider.embed(&block_texts).await?;

    // Step 4: Embed intents
    let intent_texts: Vec<&str> = intents.iter().map(|i| i.query.as_str()).collect();
    let intent_embeddings = provider.embed(&intent_texts).await?;

    // Step 5: Compute token estimates per block
    let block_tokens: Vec<usize> = blocks.iter().map(|b| estimate_tokens(b.text)).collect();
    let avg_block_tokens = block_tokens.iter().sum::<usize>() as f64 / block_tokens.len() as f64;

    // Compute min/max blocks per chunk from budgets
    let min_blocks = ((config.soft_budget as f64 * 0.5 / avg_block_tokens).ceil() as usize).max(1);
    let max_blocks = ((config.hard_budget as f64 / avg_block_tokens).ceil() as usize)
        .max(min_blocks + 1)
        .min(block_count);

    // Step 6: Dynamic programming
    // dp[i] = (best_score, backtrack_index) for blocks 0..i
    let n = block_count;
    let mut dp = vec![(f64::NEG_INFINITY, 0usize); n + 1];
    dp[0] = (0.0, 0);

    for i in 1..=n {
        for chunk_size in min_blocks..=max_blocks {
            if chunk_size > i {
                break;
            }
            let start = i - chunk_size;
            if dp[start].0 == f64::NEG_INFINITY {
                continue;
            }
            let (alignment, _) = chunk_alignment(
                &block_embeddings,
                &intent_embeddings,
                start,
                i,
            );
            let score = dp[start].0 + alignment;
            if score > dp[i].0 {
                dp[i] = (score, start);
            }
        }
    }

    // Backtrack to recover partition
    let mut boundaries = Vec::new();
    let mut pos = n;
    while pos > 0 {
        let start = dp[pos].1;
        boundaries.push((start, pos));
        pos = start;
    }
    boundaries.reverse();

    // Step 7: Build chunks
    let mut chunks = Vec::with_capacity(boundaries.len());
    for (start, end) in &boundaries {
        let chunk_text: String = blocks[*start..*end]
            .iter()
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("\n");

        let offset_start = blocks[*start].offset;
        let offset_end = blocks[end - 1].offset + blocks[end - 1].text.len();
        let token_est = estimate_tokens(&chunk_text);

        let (alignment, best_intent_idx) = chunk_alignment(
            &block_embeddings,
            &intent_embeddings,
            *start,
            *end,
        );

        let heading_path = heading_paths[*start].clone();

        chunks.push(IntentChunk {
            text: chunk_text,
            offset_start,
            offset_end,
            token_estimate: token_est,
            best_intent: best_intent_idx,
            alignment_score: alignment,
            heading_path,
        });
    }

    // Populate intent -> chunk mappings
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        if chunk.best_intent < intents.len() {
            intents[chunk_idx % intents.len()].matched_chunks.push(chunk_idx);
        }
    }
    // Fix: map each chunk's best_intent properly
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        intents[chunk.best_intent].matched_chunks.push(chunk_idx);
    }
    // Deduplicate matched_chunks
    for intent in &mut intents {
        intent.matched_chunks.sort_unstable();
        intent.matched_chunks.dedup();
    }

    let partition_score = if chunks.is_empty() {
        0.0
    } else {
        chunks.iter().map(|c| c.alignment_score).sum::<f64>() / chunks.len() as f64
    };

    Ok(IntentResult {
        chunks,
        intents,
        partition_score,
        block_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centroid_basic() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let c = centroid(&embeddings, 0, 2);
        assert!((c[0] - 0.5).abs() < f64::EPSILON);
        assert!((c[1] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world foo bar"), 4);
        assert_eq!(estimate_tokens(""), 0);
    }
}
```

- [ ] **Register in `src/semantic/mod.rs`**

Add:

```rust
pub mod intent_chunk;
```

And add public re-exports at the bottom of the file (following the pattern of existing `cognitive_chunk` re-exports):

```rust
pub use intent_chunk::{intent_chunk, intent_chunk_plain};
```

- [ ] **Run `cargo check` and tests**

Run: `cargo check 2>&1 | head -20`
Run: `cargo test intent_chunk -- --nocapture 2>&1 | tail -20`
Expected: compiles, 4 unit tests pass

- [ ] **Commit**

```bash
git add src/semantic/intent_chunk.rs src/semantic/mod.rs
git commit -m "feat(intent): add DP alignment pipeline"
```

### Step 2.4: Create CLI subcommand

- [ ] **Create `src/cli/intent_cmd.rs`**

This follows the exact pattern of `cognitive_cmd.rs`. The file should contain:
- `IntentArgs` struct with clap derive (input, provider, model, api_key, base_url, model_path, cf_* OAuth flags, intent-model, max-intents, soft-budget, hard-budget, format, merge, chunk-size, no-markdown)
- `pub async fn run(args: &IntentArgs, global: &GlobalOpts) -> anyhow::Result<()>` that:
  1. Reads input via `read_input` (same helper as cognitive)
  2. Resolves LLM config via `LlmConfig::resolve`
  3. Creates `CompletionClient`
  4. Constructs `IntentConfig`
  5. Dispatches to provider (same match pattern as cognitive)
  6. Calls `intent_chunk` or `intent_chunk_plain`
  7. Optionally applies merge post-processing
  8. Writes output in requested format

The CLI args struct and run function are structurally identical to `cognitive_cmd.rs` but with fewer flags (no reranker, no relations, no synopsis, no emit-signals, no graph) and with `--intent-model` and `--max-intents` added.

- [ ] **Register in `src/cli/mod.rs`**

Add:

```rust
pub mod intent_cmd;
```

- [ ] **Register in `src/main.rs`**

Add to Commands enum (after `Cognitive` variant):

```rust
    /// Intent-driven chunking optimized for predicted user queries
    #[command(after_help = "\
EXAMPLES:
  cognigraph-chunker intent -i doc.md -p openai --api-key $KEY
  cognigraph-chunker intent -i doc.md --intent-model gpt-4.1-mini --max-intents 30
  cognigraph-chunker intent -i doc.md -p ollama -f json
")]
    Intent(Box<cli::intent_cmd::IntentArgs>),
```

Add match arm:

```rust
        Commands::Intent(args) => cli::intent_cmd::run(args, &cli.global).await,
```

- [ ] **Run `cargo check`**

Run: `cargo check 2>&1 | head -20`
Expected: compiles

- [ ] **Commit**

```bash
git add src/cli/intent_cmd.rs src/cli/mod.rs src/main.rs
git commit -m "feat(intent): add CLI subcommand"
```

### Step 2.5: Create API handler

- [ ] **Create `src/api/intent.rs`**

Follows the pattern of `src/api/cognitive.rs`. Defines:
- `IntentRequest` (text, provider, model, api_key, base_url, model_path, cf_* OAuth flags, intent_model, max_intents, soft_budget, hard_budget, no_markdown)
- `IntentResponse` (chunks, intents, partition_score, count)
- `intent_handler` function matching the cognitive pattern (provider dispatch → `intent_chunk` / `intent_chunk_plain`)

- [ ] **Register in `src/api/mod.rs`**

Add `pub mod intent;` and route:

```rust
        .route(
            "/api/v1/intent",
            axum::routing::post(intent::intent_handler),
        )
```

- [ ] **Run `cargo check`**

Run: `cargo check 2>&1 | head -20`
Expected: compiles

- [ ] **Commit**

```bash
git add src/api/intent.rs src/api/mod.rs
git commit -m "feat(intent): add POST /api/v1/intent endpoint"
```

---

## Task 3: Enriched Chunking

Structure-preserving chunking + single-call LLM enrichment with 7 metadata fields + semantic-key recombination.

**Files:**
- Create: `src/semantic/enriched_types.rs`
- Create: `src/llm/enrichment.rs`
- Create: `src/semantic/enriched_chunk.rs`
- Create: `src/cli/enriched_cmd.rs`
- Create: `src/api/enriched.rs`
- Modify: `src/semantic/mod.rs`, `src/llm/mod.rs`, `src/cli/mod.rs`, `src/api/mod.rs`, `src/main.rs`

### Step 3.1: Create enriched types

- [ ] **Create `src/semantic/enriched_types.rs`**

```rust
//! Data types for enriched chunking.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A typed entity extracted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedEntity {
    pub name: String,
    pub entity_type: String,
}

/// A record of chunks that were merged during key-based recombination.
#[derive(Debug, Clone, Serialize)]
pub struct MergeRecord {
    pub result_chunk: usize,
    pub source_chunks: Vec<usize>,
    pub shared_key: String,
}

/// A single chunk produced by enriched chunking, with full metadata.
#[derive(Debug, Clone, Serialize)]
pub struct EnrichedChunk {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub token_estimate: usize,
    pub title: String,
    pub summary: String,
    pub keywords: Vec<String>,
    pub typed_entities: Vec<TypedEntity>,
    pub hypothetical_questions: Vec<String>,
    pub semantic_keys: Vec<String>,
    pub category: String,
    pub heading_path: Vec<String>,
}

/// Result of enriched chunking.
#[derive(Debug)]
pub struct EnrichedResult {
    pub chunks: Vec<EnrichedChunk>,
    /// Semantic key → list of chunk indices.
    pub key_dictionary: HashMap<String, Vec<usize>>,
    pub merge_history: Vec<MergeRecord>,
    pub block_count: usize,
}
```

- [ ] **Register in `src/semantic/mod.rs`**

Add:

```rust
pub mod enriched_types;
```

- [ ] **Run `cargo check`, commit**

```bash
git add src/semantic/enriched_types.rs src/semantic/mod.rs
git commit -m "feat(enriched): add enriched chunking data types"
```

### Step 3.2: Create LLM enrichment prompt

- [ ] **Create `src/llm/enrichment.rs`**

```rust
//! Single-call LLM enrichment for enriched chunking.
//!
//! Extracts 7 metadata fields per chunk in one structured JSON call.
//! Maintains a rolling semantic key dictionary across chunks.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::CompletionClient;
use crate::semantic::enriched_types::TypedEntity;

/// Raw enrichment response from the LLM.
#[derive(Debug, Deserialize)]
pub struct EnrichmentResponse {
    pub title: String,
    pub summary: String,
    pub keywords: Vec<String>,
    pub typed_entities: Vec<TypedEntity>,
    pub hypothetical_questions: Vec<String>,
    pub semantic_keys: Vec<String>,
    pub category: String,
}

const SYSTEM_PROMPT: &str = "\
You are a document enrichment engine. For each text chunk, extract structured metadata \
to maximize its usefulness for information retrieval.

Rules:
- title: a descriptive title (5-15 words) capturing the chunk's main topic
- summary: 1-2 sentence synopsis of the key content
- keywords: 5-10 relevant terms or phrases (include domain-specific terminology)
- typed_entities: named entities with types (person, organization, concept, location, etc.)
- hypothetical_questions: 3-5 questions this chunk could answer (for search matching)
- semantic_keys: 2-4 lowercase hyphenated topic keys (e.g. \"protein-folding\", \"clinical-dosing\")
- category: one of: background, methodology, results, discussion, configuration, reference, definition, procedure, example, other

For semantic_keys, reuse existing keys from the dictionary when the content matches. \
Create new keys only for genuinely new topics.";

fn json_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "summary": { "type": "string" },
            "keywords": { "type": "array", "items": { "type": "string" } },
            "typed_entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "entity_type": { "type": "string" }
                    },
                    "required": ["name", "entity_type"],
                    "additionalProperties": false
                }
            },
            "hypothetical_questions": { "type": "array", "items": { "type": "string" } },
            "semantic_keys": { "type": "array", "items": { "type": "string" } },
            "category": { "type": "string" }
        },
        "required": ["title", "summary", "keywords", "typed_entities", "hypothetical_questions", "semantic_keys", "category"],
        "additionalProperties": false
    })
}

/// Enrich a single chunk with 7 metadata fields.
///
/// The `existing_keys` parameter provides the rolling key dictionary
/// so the LLM can reuse existing semantic keys.
pub async fn enrich_chunk(
    client: &CompletionClient,
    text: &str,
    existing_keys: &HashMap<String, Vec<usize>>,
) -> Result<EnrichmentResponse> {
    let key_list: String = if existing_keys.is_empty() {
        "No existing keys yet.".to_string()
    } else {
        let keys: Vec<&str> = existing_keys.keys().map(|k| k.as_str()).collect();
        format!("Existing semantic keys: {}", keys.join(", "))
    };

    let user_prompt = format!(
        "{key_list}\n\nChunk text:\n{text}"
    );

    let response = client
        .complete_json(SYSTEM_PROMPT, &user_prompt, json_schema())
        .await
        .context("Chunk enrichment failed")?;

    serde_json::from_str(&response).context("Failed to parse enrichment response")
}

/// Re-enrich a merged chunk (title + summary only).
pub async fn re_enrich_merged(
    client: &CompletionClient,
    text: &str,
) -> Result<(String, String)> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "summary": { "type": "string" }
        },
        "required": ["title", "summary"],
        "additionalProperties": false
    });

    let response = client
        .complete_json(
            "You are a summarization engine. Generate a title and 1-2 sentence summary for this merged chunk.",
            text,
            schema,
        )
        .await
        .context("Re-enrichment failed")?;

    #[derive(Deserialize)]
    struct ReEnrich {
        title: String,
        summary: String,
    }

    let parsed: ReEnrich =
        serde_json::from_str(&response).context("Failed to parse re-enrichment response")?;

    Ok((parsed.title, parsed.summary))
}
```

- [ ] **Register in `src/llm/mod.rs`**

Add:

```rust
pub mod enrichment;
```

- [ ] **Run `cargo check`, commit**

```bash
git add src/llm/enrichment.rs src/llm/mod.rs
git commit -m "feat(enriched): add LLM enrichment prompt with rolling keys"
```

### Step 3.3: Create enriched chunking pipeline

- [ ] **Create `src/semantic/enriched_chunk.rs`**

This file implements:
- `EnrichedConfig` (soft_budget, hard_budget, recombine, re_enrich booleans)
- `enriched_chunk()` — main pipeline: split_blocks → initial grouping → LLM enrichment → key-based recombination → optional re-enrichment
- `enriched_chunk_plain()` — plain text variant
- `initial_grouping()` — greedy accumulator respecting atomic blocks and heading starts
- `recombine_by_keys()` — bin-packing merge of same-key chunks within hard budget

Tests: `test_initial_grouping_respects_budget`, `test_initial_grouping_heading_starts_new_chunk`

- [ ] **Register in `src/semantic/mod.rs`**

Add `pub mod enriched_chunk;` and re-export `enriched_chunk::{enriched_chunk, enriched_chunk_plain}`.

- [ ] **Run `cargo check` and tests, commit**

```bash
git add src/semantic/enriched_chunk.rs src/semantic/mod.rs
git commit -m "feat(enriched): add enriched chunking pipeline with key recombination"
```

### Step 3.4: Create CLI subcommand and API handler

- [ ] **Create `src/cli/enriched_cmd.rs`**

Pattern: same as `intent_cmd.rs` but with:
- No embedding provider flags (enriched doesn't need embeddings)
- `--enrichment-model` instead of `--intent-model`
- `--no-recombine`, `--no-re-enrich` boolean flags
- Only needs LLM config (api-key, llm-base-url)

- [ ] **Create `src/api/enriched.rs`**

Pattern: same as `intent.rs` but without embedding provider dispatch — only needs `CompletionClient`.

- [ ] **Register in `src/cli/mod.rs`, `src/api/mod.rs`, `src/main.rs`**

Add module declarations, route, Commands variant + match arm.

- [ ] **Run `cargo check`, commit**

```bash
git add src/cli/enriched_cmd.rs src/api/enriched.rs src/cli/mod.rs src/api/mod.rs src/main.rs
git commit -m "feat(enriched): add CLI subcommand + API endpoint"
```

---

## Task 4: Topology-Aware Chunking

Builds a Structured Intermediate Representation (SIR) from the document, then uses two LLM agents (Inspector + Refiner) to produce topology-preserving chunks.

**Files:**
- Create: `src/semantic/sir.rs`
- Create: `src/semantic/topo_types.rs`
- Create: `src/llm/topo_agents.rs`
- Create: `src/semantic/topo_chunk.rs`
- Create: `src/cli/topo_cmd.rs`
- Create: `src/api/topo.rs`
- Modify: `src/semantic/mod.rs`, `src/llm/mod.rs`, `src/cli/mod.rs`, `src/api/mod.rs`, `src/main.rs`

### Step 4.1: Create SIR data structures

- [ ] **Create `src/semantic/sir.rs`**

```rust
//! Structured Intermediate Representation (SIR) for topology-aware chunking.
//!
//! The SIR is a tree built from the heading hierarchy with content blocks as leaves.
//! Cross-reference edges link blocks that share entities or discourse continuations.

use serde::{Deserialize, Serialize};

use super::blocks::BlockKind;

/// Type of node in the SIR tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SirNodeType {
    /// A section defined by a heading.
    Section,
    /// A content block (sentence, table, code, list, etc.).
    ContentBlock,
}

/// Type of cross-reference edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SirEdgeType {
    /// Shared entity mentions between blocks.
    EntityCoref,
    /// Discourse continuation marker ("As described above").
    DiscourseContinuation,
}

/// A node in the SIR tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SirNode {
    pub id: usize,
    pub node_type: SirNodeType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heading: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heading_level: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_type: Option<BlockKind>,
    /// Start and end block indices (inclusive range).
    pub block_range: (usize, usize),
    pub children: Vec<usize>,
    /// First 200 chars of text for LLM context.
    pub text_preview: String,
    pub token_estimate: usize,
}

/// A cross-reference edge between two SIR nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SirEdge {
    pub from: usize,
    pub to: usize,
    pub edge_type: SirEdgeType,
}

/// The complete Structured Intermediate Representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sir {
    pub nodes: Vec<SirNode>,
    pub edges: Vec<SirEdge>,
    pub root: usize,
}
```

Note: `BlockKind` needs `Serialize, Deserialize` derives. The existing `BlockKind` in `src/semantic/blocks.rs` only has `Debug, Clone, Copy, PartialEq, Eq`. We need to add serde derives.

- [ ] **Add serde derives to `BlockKind` in `src/semantic/blocks.rs`**

Change line 14 from:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
```

to:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
```

- [ ] **Register in `src/semantic/mod.rs`**

Add:

```rust
pub mod sir;
```

- [ ] **Run `cargo check`, commit**

```bash
git add src/semantic/sir.rs src/semantic/blocks.rs src/semantic/mod.rs
git commit -m "feat(topo): add SIR data structures"
```

### Step 4.2: Create topo types

- [ ] **Create `src/semantic/topo_types.rs`**

```rust
//! Data types for topology-aware chunking output.

use serde::{Deserialize, Serialize};

use super::sir::Sir;

/// Section classification assigned by the Inspector agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SectionClass {
    Atomic,
    Splittable,
    MergeCandidate,
}

/// Classification of a section by the Inspector agent.
#[derive(Debug, Clone, Serialize)]
pub struct SectionClassification {
    pub section_id: usize,
    pub class: SectionClass,
    pub reason: String,
}

/// A single chunk produced by topology-aware chunking.
#[derive(Debug, Clone, Serialize)]
pub struct TopoChunk {
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub token_estimate: usize,
    pub heading_path: Vec<String>,
    pub section_classification: String,
    /// Indices of other chunks this one depends on.
    pub cross_references: Vec<usize>,
}

/// Result of topology-aware chunking.
#[derive(Debug)]
pub struct TopoResult {
    pub chunks: Vec<TopoChunk>,
    pub sir: Sir,
    pub classifications: Vec<SectionClassification>,
    pub block_count: usize,
}
```

- [ ] **Register in `src/semantic/mod.rs`**, run `cargo check`, commit.

```bash
git add src/semantic/topo_types.rs src/semantic/mod.rs
git commit -m "feat(topo): add topo chunking result types"
```

### Step 4.3: Create LLM topo agents

- [ ] **Create `src/llm/topo_agents.rs`**

Contains:
- `InspectorResponse`, `RefinerResponse` deserialization types
- `INSPECTOR_SYSTEM_PROMPT` — receives SIR JSON, classifies sections
- `REFINER_SYSTEM_PROMPT` — receives Inspector output + SIR + text of splittable sections
- `inspector_schema()`, `refiner_schema()` — JSON schemas
- `pub async fn inspect_sir(client, sir_json) -> Result<InspectorResponse>`
- `pub async fn refine_partition(client, inspector_output, sir_json, section_texts) -> Result<RefinerResponse>`
- Context window handling: if SIR JSON > 80% of 128k tokens (~100k chars), truncate text_preview fields

- [ ] **Register in `src/llm/mod.rs`**, run `cargo check`, commit.

```bash
git add src/llm/topo_agents.rs src/llm/mod.rs
git commit -m "feat(topo): add Inspector + Refiner LLM agents"
```

### Step 4.4: Create topo chunking pipeline

- [ ] **Create `src/semantic/topo_chunk.rs`**

Implements:
- `TopoConfig` (soft_budget, hard_budget, emit_sir boolean)
- `topo_chunk(text, llm_client, config) -> Result<TopoResult>` — main pipeline
- `build_sir(blocks, heading_paths, entities)` — heuristic SIR construction from blocks
- `assemble_from_partition(blocks, partition, heading_paths)` — map LLM partition back to text spans

Tests: `test_build_sir_basic`, `test_build_sir_flat_document`

- [ ] **Register in `src/semantic/mod.rs`**, run tests, commit.

### Step 4.5: Create CLI subcommand and API handler

- [ ] **Create `src/cli/topo_cmd.rs`**

Same pattern as enriched — no embedding provider, only LLM config. Adds `--topo-model`, `--emit-sir`.

- [ ] **Create `src/api/topo.rs`**

Same pattern as enriched API handler.

- [ ] **Register everywhere, run `cargo check`, commit.**

```bash
git add src/cli/topo_cmd.rs src/api/topo.rs src/cli/mod.rs src/api/mod.rs src/main.rs
git commit -m "feat(topo): add CLI subcommand + API endpoint"
```

---

## Task 5: Adaptive Chunking

Meta-router that scores candidates with quality metrics and picks the best.

**Files:**
- Create: `src/semantic/adaptive_types.rs`
- Create: `src/semantic/adaptive_chunk.rs`
- Create: `src/cli/adaptive_cmd.rs`
- Create: `src/api/adaptive.rs`
- Modify: `src/semantic/mod.rs`, `src/cli/mod.rs`, `src/api/mod.rs`, `src/main.rs`

### Step 5.1: Create adaptive types

- [ ] **Create `src/semantic/adaptive_types.rs`**

```rust
//! Data types for adaptive chunking.

use serde::Serialize;

use super::quality_metrics::{MetricWeights, QualityMetrics};

/// Result of adaptive chunking.
#[derive(Debug, Serialize)]
pub struct AdaptiveResult {
    /// Name of the winning method.
    pub winner: String,
    /// Chunks from the winning method (polymorphic JSON).
    pub chunks: Vec<serde_json::Value>,
    /// Quality evaluation report for all candidates.
    pub report: AdaptiveReport,
    /// Number of chunks in the winning result.
    pub count: usize,
}

/// Full quality evaluation report.
#[derive(Debug, Serialize)]
pub struct AdaptiveReport {
    pub candidates: Vec<CandidateScore>,
    pub pre_screening: Vec<ScreeningDecision>,
    pub metric_weights: MetricWeights,
}

/// Scores for a single candidate method.
#[derive(Debug, Serialize)]
pub struct CandidateScore {
    pub method: String,
    pub metrics: QualityMetrics,
    pub chunk_count: usize,
    pub total_tokens: usize,
}

/// Pre-screening decision for a candidate method.
#[derive(Debug, Serialize)]
pub struct ScreeningDecision {
    pub method: String,
    pub included: bool,
    pub reason: String,
}
```

- [ ] **Register in `src/semantic/mod.rs`**, run `cargo check`, commit.

### Step 5.2: Create adaptive orchestrator

- [ ] **Create `src/semantic/adaptive_chunk.rs`**

Implements:
- `AdaptiveConfig` (candidates list, force_candidates, metric_weights, soft/hard budget, provider configs)
- `adaptive_chunk()` — main orchestrator:
  1. Pre-screen candidates (heading level check for topo, token count for intent, structure check for enriched)
  2. Run each candidate method
  3. Convert each result to `Vec<ChunkForEval>` (generic text + offsets)
  4. Score via `evaluate_chunks`
  5. Pick winner by composite score (ties broken by fewer chunks)
  6. Serialize winner's chunks as `Vec<serde_json::Value>`
  7. Return `AdaptiveResult`

Tests: `test_pre_screening_flat_doc`, `test_composite_score_selection`

- [ ] **Register in `src/semantic/mod.rs`**, run tests, commit.

### Step 5.3: Create CLI subcommand and API handler

- [ ] **Create `src/cli/adaptive_cmd.rs`**

Needs ALL provider flags (embedding + LLM) since it delegates to any method. Adds:
- `--candidates <LIST>` (comma-separated)
- `--force-candidates`
- `--metric-weights <KEY=VALUE,...>`
- `--report` (include quality report in JSON output)

- [ ] **Create `src/api/adaptive.rs`**

Accepts all provider credentials + `candidates`, `force_candidates`, `metric_weights`, `include_report`.

- [ ] **Register everywhere, run `cargo check`, commit.**

```bash
git add src/semantic/adaptive_types.rs src/semantic/adaptive_chunk.rs src/cli/adaptive_cmd.rs src/api/adaptive.rs src/semantic/mod.rs src/cli/mod.rs src/api/mod.rs src/main.rs
git commit -m "feat(adaptive): add meta-router with quality metrics scoring"
```

---

## Task 6: Final Integration

### Step 6.1: Run full test suite

- [ ] **Run all tests**

Run: `cargo test 2>&1 | tail -30`
Expected: all existing 108 tests + new tests pass

- [ ] **Run clippy**

Run: `cargo clippy -- -W clippy::all 2>&1 | tail -20`
Expected: no errors (warnings are acceptable)

### Step 6.2: Verify all CLI subcommands are registered

- [ ] **Check help output**

Run: `cargo run -- --help 2>&1`
Expected: `intent`, `topo`, `enriched`, `adaptive` appear in subcommand list alongside `chunk`, `split`, `semantic`, `cognitive`, `serve`

### Step 6.3: Commit final integration

```bash
git add -A
git commit -m "feat: integrate all 4 new chunking methods + quality metrics"
```

---

## Task 7: Documentation

### Step 7.1: Create doc articles

- [ ] **Create `docs/10-intent-driven-chunking.md`** — describe the intent-driven method, pipeline, CLI flags, API endpoint, use cases (follows the style of existing `docs/07-cognition-aware-chunking.md`)

- [ ] **Create `docs/11-topology-aware-chunking.md`** — describe the topo method, SIR construction, dual-agent architecture, CLI/API

- [ ] **Create `docs/12-enriched-chunking.md`** — describe the enriched method, 7-field metadata, rolling keys, recombination, CLI/API

- [ ] **Create `docs/13-adaptive-chunking.md`** — describe the adaptive method, 5 quality metrics, pre-screening, CLI/API

- [ ] **Update `README.md`** — add new modes to the CLI reference table and method comparison

- [ ] **Commit**

```bash
git add docs/10-intent-driven-chunking.md docs/11-topology-aware-chunking.md docs/12-enriched-chunking.md docs/13-adaptive-chunking.md README.md
git commit -m "docs: add articles for 4 new chunking methods"
```
