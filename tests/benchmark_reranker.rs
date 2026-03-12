//! Benchmark: reranker impact on cognitive chunking quality and throughput.
//!
//! Compares cognitive chunking with vs without a real ONNX cross-encoder
//! reranker, using OpenAI embeddings for boundary scoring.
//!
//! Requires:
//! - `.env.openai` with `OPENAI_API_KEY=sk-...`
//! - ONNX reranker model at `models/ms-marco-MiniLM-L-6-v2/`
//!
//! Run with: cargo test --test benchmark_reranker -- --ignored --nocapture

use std::time::Instant;

use cognigraph_chunker::embeddings::openai::OpenAiProvider;
use cognigraph_chunker::embeddings::reranker::{OnnxReranker, RerankerProvider};
use cognigraph_chunker::semantic::cognitive_types::CognitiveConfig;
use cognigraph_chunker::semantic::evaluation::EvaluationMetrics;
use cognigraph_chunker::semantic::{cognitive_chunk, cognitive_chunk_with_reranker};

// ── Environment helpers ────────────────────────────────────────────

fn load_openai_key() -> String {
    // Try env var first, then .env.openai file
    if let Ok(key) = std::env::var("OPENAI_API_KEY")
        && !key.is_empty()
    {
        return key;
    }
    let content = std::fs::read_to_string(".env.openai")
        .expect("Missing .env.openai file with OPENAI_API_KEY");
    for line in content.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("OPENAI_API_KEY=") {
            let val = val.trim();
            if !val.is_empty() {
                return val.to_string();
            }
        }
    }
    panic!("OPENAI_API_KEY not found in .env.openai");
}

const RERANKER_MODEL_PATH: &str = "models/ms-marco-MiniLM-L-6-v2";

// ── Sample documents ───────────────────────────────────────────────

const DOCS: &[(&str, &str)] = &[
    (
        "mixed_headings",
        r#"# Architecture Overview

The system uses a modular pipeline for text processing. Each module handles a specific stage of the pipeline.

## Block Extraction

Markdown documents are parsed into an AST. Tables and code blocks are kept as atomic units. Paragraphs are split into individual sentences for fine-grained boundary detection.

This approach ensures that structured content like tables and code examples are never broken mid-structure. It also allows the system to detect topic boundaries at the sentence level within prose sections.

## Embedding Generation

Each block is embedded using a neural network model. The embedding captures the semantic meaning of the text in a high-dimensional vector space.

Models like `nomic-embed-text` produce 768-dimensional vectors. OpenAI's `text-embedding-3-small` produces 1536-dimensional vectors. The choice of model affects both quality and performance.

## Boundary Detection

The cosine similarity between consecutive embeddings reveals topic continuity. Low similarity indicates a potential topic change.

A Savitzky-Golay filter smooths the similarity curve. Local minima in the smoothed curve mark candidate split points. A percentile threshold filters out weak boundaries.

## Chunk Assembly

Blocks between boundaries are joined into chunks. Each chunk maintains coherent meaning and context for downstream tasks like RAG and search indexing.
"#,
    ),
    (
        "table_heavy",
        r#"# Quarterly Financial Report

## Revenue Summary

| Quarter | Revenue ($M) | Growth (%) | Region |
|---------|-------------|------------|--------|
| Q1 2025 | 142.3 | 12.1 | North America |
| Q2 2025 | 156.8 | 10.2 | North America |
| Q3 2025 | 148.9 | -5.0 | North America |
| Q4 2025 | 171.2 | 15.0 | North America |

The revenue figures show strong growth in Q1 and Q2, followed by a seasonal dip in Q3. However, Q4 recovery exceeded expectations with 15% quarter-over-quarter growth driven by enterprise contracts.

## Operating Expenses

| Category | Q3 2025 ($M) | Q4 2025 ($M) | Change (%) |
|----------|-------------|-------------|------------|
| Personnel | 45.2 | 48.1 | 6.4 |
| Infrastructure | 22.8 | 24.3 | 6.6 |
| Marketing | 15.6 | 18.2 | 16.7 |
| R&D | 31.4 | 33.7 | 7.3 |

Marketing spend increased significantly in Q4 to support the product launch. Furthermore, R&D investment continued to grow as the team expanded. This trend is expected to continue into Q1 2026.

## Profit Margins

Net profit margin improved from 18.2% in Q3 to 21.4% in Q4. This improvement was driven by revenue growth outpacing expense increases. The operating leverage in the business model is becoming more pronounced as scale increases.
"#,
    ),
    (
        "code_blocks",
        r#"# API Integration Guide

## Authentication

All API requests require a Bearer token in the Authorization header. Tokens expire after 24 hours and must be refreshed.

```bash
curl -X POST https://api.example.com/auth/token \
  -H "Content-Type: application/json" \
  -d '{"client_id": "YOUR_ID", "client_secret": "YOUR_SECRET"}'
```

The response contains an access token and refresh token. Store the refresh token securely for obtaining new access tokens.

## Chunking Endpoint

The chunking endpoint accepts markdown text and returns semantically coherent chunks.

```json
{
  "text": "Your markdown document here...",
  "provider": "ollama",
  "model": "nomic-embed-text",
  "config": {
    "sim_window": 3,
    "threshold": 0.5
  }
}
```

## Error Handling

The API returns standard HTTP status codes. Common errors include:

- `400 Bad Request` — Invalid input or missing required fields
- `401 Unauthorized` — Invalid or expired token
- `413 Payload Too Large` — Input exceeds maximum size
- `500 Internal Server Error` — Embedding provider failure

```rust
match response.status() {
    StatusCode::OK => parse_chunks(response),
    StatusCode::UNAUTHORIZED => refresh_token_and_retry(),
    status => Err(ApiError::unexpected(status)),
}
```

Always implement exponential backoff for retries. The API enforces rate limits of 100 requests per minute per token.
"#,
    ),
    (
        "long_narrative",
        r#"# Clinical Trial Protocol: Phase III Study of Compound XR-7742

## Background and Rationale

Compound XR-7742 is a selective serotonin reuptake inhibitor developed for the treatment of major depressive disorder. Phase II trials demonstrated statistically significant improvement in Hamilton Depression Rating Scale scores compared to placebo, with a favorable safety profile. The most common adverse events were mild nausea and headache, both resolving within the first two weeks of treatment.

The mechanism of action involves selective inhibition of the serotonin transporter protein. Unlike earlier SSRIs, XR-7742 exhibits minimal activity at muscarinic, histaminergic, and adrenergic receptors. This selectivity is hypothesized to account for the reduced side effect burden observed in Phase II.

## Study Design

This is a randomized, double-blind, placebo-controlled, parallel-group study. Approximately 800 patients will be enrolled across 45 sites in North America and Europe. The study duration is 12 weeks, with a 4-week screening period and an optional 52-week open-label extension.

Patients will be randomized 1:1:1 to receive XR-7742 20mg, XR-7742 40mg, or matching placebo. Randomization will be stratified by baseline severity and geographic region. The primary endpoint is change from baseline in MADRS total score at Week 8.

## Inclusion Criteria

Eligible patients must meet the following criteria:

- Age 18-65 years at screening
- Diagnosis of major depressive disorder per DSM-5 criteria
- Current depressive episode of at least 8 weeks duration
- MADRS total score ≥ 26 at both screening and baseline visits
- Body mass index between 18 and 35 kg/m²

## Exclusion Criteria

Patients with the following conditions will be excluded:

- History of bipolar disorder, schizophrenia, or schizoaffective disorder
- Current substance use disorder within the past 6 months
- Active suicidal ideation with intent or plan
- Treatment-resistant depression defined as failure of ≥ 3 adequate antidepressant trials
- Significant hepatic or renal impairment

## Statistical Analysis

The primary analysis will use a mixed-model repeated measures approach. The model will include treatment group, visit, treatment-by-visit interaction, baseline MADRS score, and stratification factors as covariates. Missing data will be handled under the missing-at-random assumption.

Sample size calculations assume a treatment difference of 3 points on the MADRS, a standard deviation of 9 points, and a two-sided significance level of 0.025 to account for multiplicity. With 250 evaluable patients per group, the study has approximately 90% power to detect this difference.
"#,
    ),
];

// ── Result types ───────────────────────────────────────────────────

struct RerankerComparison {
    name: String,
    bytes: usize,
    base_chunks: usize,
    base_ms: f64,
    base_eval: EvaluationMetrics,
    rerank_chunks: usize,
    rerank_ms: f64,
    rerank_eval: EvaluationMetrics,
    reranked_boundaries: usize,
    overhead_ms: f64,
    overhead_pct: f64,
}

// ── Benchmark ──────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
#[ignore] // requires OpenAI API key + ONNX reranker model
async fn benchmark_reranker_impact() {
    eprintln!("Loading OpenAI provider...");
    let api_key = load_openai_key();
    let provider =
        OpenAiProvider::new(api_key, None, None).expect("Failed to create OpenAI provider");
    eprintln!("OpenAI provider ready.");

    eprintln!("Loading ONNX reranker from {RERANKER_MODEL_PATH}...");
    let reranker =
        OnnxReranker::new(RERANKER_MODEL_PATH).expect("Failed to load ONNX reranker model");
    eprintln!("Reranker loaded: {}", reranker.model_name());

    println!();
    println!("Embedding provider: OpenAI text-embedding-3-small");
    println!(
        "Reranker model:     {} ({})",
        reranker.model_name(),
        RERANKER_MODEL_PATH
    );
    println!();

    let config = CognitiveConfig {
        emit_signals: true,
        ..CognitiveConfig::default()
    };

    let mut results = Vec::new();

    for (name, text) in DOCS {
        eprintln!("  [{name}] Starting baseline...");

        // Baseline: cognitive chunking without reranker
        let t0 = Instant::now();
        let base_result = cognitive_chunk(text, &provider, &config)
            .await
            .expect("cognitive_chunk failed");
        let base_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  [{name}] Baseline done in {base_ms:.0}ms");

        // With reranker
        eprintln!("  [{name}] Starting reranker run...");
        let t0 = Instant::now();
        let rerank_result = cognitive_chunk_with_reranker(text, &provider, &config, &reranker)
            .await
            .expect("cognitive_chunk_with_reranker failed");
        let rerank_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  [{name}] Reranker done in {rerank_ms:.0}ms");

        // Count how many boundaries were actually reranked (have "reranked:" in reasons)
        let reranked_boundaries = rerank_result
            .signals
            .iter()
            .filter(|s| s.reasons.iter().any(|r| r.starts_with("reranked:")))
            .count();

        let overhead_ms = rerank_ms - base_ms;
        let overhead_pct = if base_ms > 0.0 {
            (overhead_ms / base_ms) * 100.0
        } else {
            0.0
        };

        println!(
            " done ({} blocks, {} base chunks, {} reranked chunks, {} boundaries refined)",
            base_result.block_count,
            base_result.chunks.len(),
            rerank_result.chunks.len(),
            reranked_boundaries,
        );

        results.push(RerankerComparison {
            name: name.to_string(),
            bytes: text.len(),
            base_chunks: base_result.chunks.len(),
            base_ms,
            base_eval: base_result.evaluation,
            rerank_chunks: rerank_result.chunks.len(),
            rerank_ms,
            rerank_eval: rerank_result.evaluation,
            reranked_boundaries,
            overhead_ms,
            overhead_pct,
        });
    }

    // ── Print results ──────────────────────────────────────────────

    println!();
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                     Reranker Impact on Cognitive Chunking (Real Providers)                  ║"
    );
    println!(
        "╠══════════════════╦═══════╦═════════════════════╦══════════════════════════════╦═════════════╣"
    );
    println!(
        "║ Document         ║ Bytes ║ Chunks (base/rerank)║ Time ms (base / rerank)      ║ Refined     ║"
    );
    println!(
        "╠══════════════════╬═══════╬═════════════════════╬══════════════════════════════╬═════════════╣"
    );

    for r in &results {
        println!(
            "║ {:<16} ║ {:>5} ║ {:>5} / {:>5}        ║ {:>7.1} / {:>7.1} ({:>+5.0}%) ║ {:>5} bnd   ║",
            r.name,
            r.bytes,
            r.base_chunks,
            r.rerank_chunks,
            r.base_ms,
            r.rerank_ms,
            r.overhead_pct,
            r.reranked_boundaries,
        );
    }

    println!(
        "╠══════════════════╩═══════╩═════════════════════╩══════════════════════════════╩═════════════╣"
    );
    println!(
        "║                                                                                            ║"
    );
    println!(
        "║  Evaluation Metrics Comparison (lower is better except heading attachment)                  ║"
    );
    println!(
        "╠══════════════════╦══════════════════════════════════╦══════════════════════════════════════╣"
    );
    println!(
        "║ Document         ║ Base                             ║ With Reranker                       ║"
    );
    println!(
        "║                  ║ Orph   Pron   Head   Disc        ║ Orph   Pron   Head   Disc           ║"
    );
    println!(
        "╠══════════════════╬══════════════════════════════════╬══════════════════════════════════════╣"
    );

    for r in &results {
        println!(
            "║ {:<16} ║ {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}%    ║ {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}%        ║",
            r.name,
            r.base_eval.entity_orphan_rate * 100.0,
            r.base_eval.pronoun_boundary_rate * 100.0,
            r.base_eval.heading_attachment_rate * 100.0,
            r.base_eval.discourse_break_rate * 100.0,
            r.rerank_eval.entity_orphan_rate * 100.0,
            r.rerank_eval.pronoun_boundary_rate * 100.0,
            r.rerank_eval.heading_attachment_rate * 100.0,
            r.rerank_eval.discourse_break_rate * 100.0,
        );
    }

    println!(
        "╚══════════════════╩══════════════════════════════════╩══════════════════════════════════════╝"
    );

    // ── Summary statistics ─────────────────────────────────────────

    let avg_overhead_pct =
        results.iter().map(|r| r.overhead_pct).sum::<f64>() / results.len() as f64;
    let avg_overhead_ms = results.iter().map(|r| r.overhead_ms).sum::<f64>() / results.len() as f64;
    let total_reranked: usize = results.iter().map(|r| r.reranked_boundaries).sum();
    let boundary_changes: usize = results
        .iter()
        .map(|r| (r.base_chunks as isize - r.rerank_chunks as isize).unsigned_abs())
        .sum();

    println!();
    println!("Summary:");
    println!("  Embedding provider:      OpenAI text-embedding-3-small");
    println!("  Reranker model:          {}", reranker.model_name());
    println!("  Average overhead:        {avg_overhead_ms:>+.1} ms ({avg_overhead_pct:>+.1}%)");
    println!(
        "  Total boundaries refined: {total_reranked} (across {} docs)",
        results.len()
    );
    println!("  Total chunk count changes: {boundary_changes}");
    println!();

    // Sanity assertions
    for r in &results {
        assert!(r.base_chunks > 0, "{}: no base chunks", r.name);
        assert!(r.rerank_chunks > 0, "{}: no reranked chunks", r.name);
    }
}
