//! Benchmark comparison: cognitive vs semantic chunking on sample documents.
//!
//! Uses a deterministic hash-based mock embedding provider so the test
//! runs without any external services. This compares chunk structure,
//! not embedding quality — it validates that the cognitive pipeline
//! produces reasonable chunks and evaluation metrics.
//!
//! Run with: cargo test --test benchmark_comparison -- --nocapture

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use anyhow::Result;

use cognigraph_chunker::embeddings::EmbeddingProvider;
use cognigraph_chunker::semantic::cognitive_types::CognitiveConfig;
use cognigraph_chunker::semantic::evaluation::format_metrics;
use cognigraph_chunker::semantic::{SemanticConfig, cognitive_chunk, semantic_chunk};

// ── Mock embedding provider ────────────────────────────────────────

const EMBED_DIM: usize = 64;

/// Deterministic embedding provider that hashes text into a unit vector.
/// Produces consistent embeddings without external services.
struct HashEmbedder;

impl EmbeddingProvider for HashEmbedder {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
        Ok(texts.iter().map(|t| hash_embed(t)).collect())
    }

    fn dimension(&self) -> Option<usize> {
        Some(EMBED_DIM)
    }
}

fn hash_embed(text: &str) -> Vec<f64> {
    let mut vec = Vec::with_capacity(EMBED_DIM);
    for i in 0..EMBED_DIM {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        i.hash(&mut hasher);
        let h = hasher.finish();
        // Map hash to [-1, 1] range
        vec.push((h as f64 / u64::MAX as f64) * 2.0 - 1.0);
    }
    // Normalize to unit vector
    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in &mut vec {
            *x /= norm;
        }
    }
    vec
}

// ── Sample documents ───────────────────────────────────────────────

/// Inline test documents covering different structural patterns.
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
        "short_prose",
        r#"# Summary

Machine learning models require high-quality training data. The data must be cleaned, normalized, and validated before use. This process is often the most time-consuming part of any ML project.

Transfer learning reduces the amount of training data needed. Pre-trained models can be fine-tuned on domain-specific data. This approach is especially effective when labeled data is scarce.
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

// ── Benchmark runner ───────────────────────────────────────────────

struct DocResult {
    name: String,
    bytes: usize,
    semantic_chunks: usize,
    semantic_avg_len: usize,
    semantic_ms: f64,
    cognitive_chunks: usize,
    cognitive_avg_len: usize,
    cognitive_ms: f64,
    eval_entity_orphan: f64,
    eval_pronoun_boundary: f64,
    eval_heading_attachment: f64,
    eval_discourse_break: f64,
}

#[tokio::test]
async fn benchmark_semantic_vs_cognitive() {
    let provider = HashEmbedder;
    let sem_config = SemanticConfig::default();
    let cog_config = CognitiveConfig {
        emit_signals: true, // need signals for evaluation
        ..CognitiveConfig::default()
    };

    let mut results = Vec::new();

    for (name, text) in DOCS {
        // Semantic chunking
        let t0 = Instant::now();
        let sem_result = semantic_chunk(text, &provider, &sem_config)
            .await
            .expect("semantic_chunk failed");
        let semantic_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Cognitive chunking
        let t0 = Instant::now();
        let cog_result = cognitive_chunk(text, &provider, &cog_config)
            .await
            .expect("cognitive_chunk failed");
        let cognitive_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let sem_total_len: usize = sem_result.chunks.iter().map(|(t, _)| t.len()).sum();
        let cog_total_len: usize = cog_result.chunks.iter().map(|c| c.text.len()).sum();

        results.push(DocResult {
            name: name.to_string(),
            bytes: text.len(),
            semantic_chunks: sem_result.chunks.len(),
            semantic_avg_len: if sem_result.chunks.is_empty() {
                0
            } else {
                sem_total_len / sem_result.chunks.len()
            },
            semantic_ms,
            cognitive_chunks: cog_result.chunks.len(),
            cognitive_avg_len: if cog_result.chunks.is_empty() {
                0
            } else {
                cog_total_len / cog_result.chunks.len()
            },
            cognitive_ms,
            eval_entity_orphan: cog_result.evaluation.entity_orphan_rate,
            eval_pronoun_boundary: cog_result.evaluation.pronoun_boundary_rate,
            eval_heading_attachment: cog_result.evaluation.heading_attachment_rate,
            eval_discourse_break: cog_result.evaluation.discourse_break_rate,
        });
    }

    // Print comparison table
    println!();
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                    Semantic vs Cognitive Chunking Comparison                            ║"
    );
    println!(
        "╠══════════════════╦═══════╦═══════════════════════╦═══════════════════════╦══════════════╣"
    );
    println!(
        "║ Document         ║ Bytes ║ Semantic (chunks/avg) ║ Cognitive (chunks/avg)║ Time (ms)    ║"
    );
    println!(
        "╠══════════════════╬═══════╬═══════════════════════╬═══════════════════════╬══════════════╣"
    );

    for r in &results {
        println!(
            "║ {:<16} ║ {:>5} ║ {:>3} chunks / {:>4} avg ║ {:>3} chunks / {:>4} avg ║ {:.1} / {:.1} ║",
            r.name,
            r.bytes,
            r.semantic_chunks,
            r.semantic_avg_len,
            r.cognitive_chunks,
            r.cognitive_avg_len,
            r.semantic_ms,
            r.cognitive_ms,
        );
    }

    println!(
        "╠══════════════════╩═══════╩═══════════════════════╩═══════════════════════╩══════════════╣"
    );
    println!(
        "║                                                                                        ║"
    );
    println!(
        "║  Cognitive Evaluation Metrics                                                          ║"
    );
    println!(
        "╠══════════════════╦═════════════╦═════════════╦═════════════╦════════════════════════════╣"
    );
    println!(
        "║ Document         ║ Entity Orph ║ Pronoun Bnd ║ Head Attach ║ Discourse Break            ║"
    );
    println!(
        "╠══════════════════╬═════════════╬═════════════╬═════════════╬════════════════════════════╣"
    );

    for r in &results {
        println!(
            "║ {:<16} ║ {:>9.1}%  ║ {:>9.1}%  ║ {:>9.1}%  ║ {:>9.1}%                   ║",
            r.name,
            r.eval_entity_orphan * 100.0,
            r.eval_pronoun_boundary * 100.0,
            r.eval_heading_attachment * 100.0,
            r.eval_discourse_break * 100.0,
        );
    }

    println!(
        "╚══════════════════╩═════════════╩═════════════╩═════════════╩════════════════════════════╝"
    );
    println!();

    // Print detailed cognitive evaluation for each doc
    for r in &results {
        let metrics = cognigraph_chunker::semantic::evaluation::EvaluationMetrics {
            entity_orphan_rate: r.eval_entity_orphan,
            pronoun_boundary_rate: r.eval_pronoun_boundary,
            heading_attachment_rate: r.eval_heading_attachment,
            discourse_break_rate: r.eval_discourse_break,
            triple_severance_rate: 0.0,
            chunk_count: r.cognitive_chunks,
            block_count: 0, // not tracked in DocResult
        };
        println!("[{}] {}", r.name, format_metrics(&metrics));
        println!();
    }

    // Basic sanity assertions (mock embeddings produce random boundaries,
    // so we only assert structural invariants, not quality metrics)
    for r in &results {
        assert!(r.semantic_chunks > 0, "{}: no semantic chunks", r.name);
        assert!(r.cognitive_chunks > 0, "{}: no cognitive chunks", r.name);
    }
}

/// Test with the external sample documents if available.
#[tokio::test]
async fn benchmark_sample_documents() {
    let samples_dir = std::path::Path::new("/Users/skitsanos/FTP/Products/CogniGraph/samples");
    if !samples_dir.exists() {
        println!(
            "Skipping: sample documents not found at {}",
            samples_dir.display()
        );
        return;
    }

    let provider = HashEmbedder;
    let sem_config = SemanticConfig::default();
    let cog_config = CognitiveConfig {
        emit_signals: true,
        ..CognitiveConfig::default()
    };

    // Pick a representative subset (one per domain)
    let sample_files = [
        "forensics-incident-report-001.md",
        "healthcare-clinical-trial-008.md",
        "finance-audit-report-020.md",
        "pharma-batch-record-011.md",
        "manufacturing-deviation-report-024.md",
    ];

    println!();
    println!("Sample Document Benchmark (external files)");
    println!("{:-<90}", "");
    println!(
        "{:<42} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8}",
        "Document", "Bytes", "S.chk", "C.chk", "S.avg", "C.avg", "Eval"
    );
    println!("{:-<90}", "");

    for filename in &sample_files {
        let path = samples_dir.join(filename);
        if !path.exists() {
            println!("{:<42} SKIPPED (not found)", filename);
            continue;
        }

        let text = std::fs::read_to_string(&path).expect("failed to read sample");

        let sem_result = semantic_chunk(&text, &provider, &sem_config)
            .await
            .expect("semantic_chunk failed");

        let cog_result = cognitive_chunk(&text, &provider, &cog_config)
            .await
            .expect("cognitive_chunk failed");

        let sem_total: usize = sem_result.chunks.iter().map(|(t, _)| t.len()).sum();
        let sem_avg = if sem_result.chunks.is_empty() {
            0
        } else {
            sem_total / sem_result.chunks.len()
        };

        let cog_total: usize = cog_result.chunks.iter().map(|c| c.text.len()).sum();
        let cog_avg = if cog_result.chunks.is_empty() {
            0
        } else {
            cog_total / cog_result.chunks.len()
        };

        let eval = &cog_result.evaluation;
        let eval_summary = format!(
            "orph={:.0}% pro={:.0}% head={:.0}% disc={:.0}%",
            eval.entity_orphan_rate * 100.0,
            eval.pronoun_boundary_rate * 100.0,
            eval.heading_attachment_rate * 100.0,
            eval.discourse_break_rate * 100.0,
        );

        println!(
            "{:<42} {:>6} {:>6} {:>6} {:>6} {:>6}  {}",
            filename,
            text.len(),
            sem_result.chunks.len(),
            cog_result.chunks.len(),
            sem_avg,
            cog_avg,
            eval_summary,
        );
    }

    println!("{:-<90}", "");
    println!();
}
