use criterion::{Criterion, black_box, criterion_group, criterion_main};

use cognigraph_chunker::core::chunk::chunk;
use cognigraph_chunker::core::merge::{find_merge_indices, merge_splits};
use cognigraph_chunker::core::savgol::{
    filter_split_indices, find_local_minima_interpolated, savgol_filter, windowed_cross_similarity,
};
use cognigraph_chunker::core::split::{
    IncludeDelim, PatternSplitter, split_at_delimiters, split_at_patterns,
};
use cognigraph_chunker::semantic::blocks::split_blocks;
use cognigraph_chunker::semantic::sentence::split_sentences;

// ---------------------------------------------------------------------------
// Sample data
// ---------------------------------------------------------------------------

const SAMPLE_TEXT: &str = "\
# Introduction to Semantic Chunking

Semantic chunking is a technique for splitting documents into meaningful segments.
Unlike fixed-size chunking, it considers the meaning of the text.
This approach uses embeddings to find natural topic boundaries.

## How It Works

The pipeline starts by extracting blocks from a markdown document.
Tables and code blocks are kept as atomic units.
Paragraphs are split into individual sentences for fine-grained analysis.

Each sentence is then embedded using a neural network model.
The cosine similarity between consecutive embeddings reveals topic continuity.
Low similarity indicates a potential topic change.

## Signal Processing

A Savitzky-Golay filter smooths the similarity curve.
Local minima in the smoothed curve mark candidate split points.
A percentile threshold filters out weak boundaries.
Minimum distance constraints prevent overly short chunks.

## Applications

- Retrieval-Augmented Generation (RAG) pipelines
- Document indexing for vector search
- Summarization pre-processing
- Knowledge base construction

## Code Example

```rust
let chunks = semantic_chunk(text, &provider, &config).await?;
for (chunk_text, offset) in &chunks {
    println!(\"Chunk at {}: {}\", offset, &chunk_text[..50]);
}
```

| Method | Precision | Recall |
|--------|-----------|--------|
| Fixed  | 0.72      | 0.68   |
| Semantic | 0.89   | 0.85   |
| Hybrid | 0.91      | 0.88   |

The hybrid approach combines fixed-size pre-chunking with semantic boundary refinement.
This gives the best results while keeping computational costs manageable.
Each chunk maintains coherent meaning and context for downstream tasks.
";

fn make_large_text(repetitions: usize) -> Vec<u8> {
    SAMPLE_TEXT.repeat(repetitions).into_bytes()
}

fn make_synthetic_similarities(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64 / n as f64 * std::f64::consts::PI * 6.0;
            0.5 + 0.4 * x.cos() + 0.1 * (x * 3.7).sin()
        })
        .collect()
}

fn make_synthetic_embeddings(n: usize, d: usize) -> Vec<f64> {
    (0..n * d)
        .map(|i| {
            let x = i as f64 * 0.1;
            x.sin() * 0.5 + 0.5
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Group 1: Fixed-size chunking
// ---------------------------------------------------------------------------

fn bench_chunking(c: &mut Criterion) {
    let text = make_large_text(10);

    let mut group = c.benchmark_group("fixed_size_chunking");

    for size in [256, 1024, 4096, 8192] {
        group.bench_function(format!("default_delims/size_{size}"), |b| {
            b.iter(|| {
                let chunks: Vec<_> = chunk(black_box(&text)).size(size).collect();
                black_box(chunks);
            });
        });
    }

    for size in [256, 1024, 4096] {
        group.bench_function(format!("custom_delims/size_{size}"), |b| {
            b.iter(|| {
                let chunks: Vec<_> = chunk(black_box(&text))
                    .size(size)
                    .delimiters(b".!?\n")
                    .collect();
                black_box(chunks);
            });
        });
    }

    group.bench_function("prefix_mode/size_1024", |b| {
        b.iter(|| {
            let chunks: Vec<_> = chunk(black_box(&text)).size(1024).prefix().collect();
            black_box(chunks);
        });
    });

    group.bench_function("consecutive/size_1024", |b| {
        b.iter(|| {
            let chunks: Vec<_> = chunk(black_box(&text))
                .size(1024)
                .prefix()
                .consecutive()
                .collect();
            black_box(chunks);
        });
    });

    group.bench_function("forward_fallback/size_1024", |b| {
        b.iter(|| {
            let chunks: Vec<_> = chunk(black_box(&text))
                .size(1024)
                .forward_fallback()
                .collect();
            black_box(chunks);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: Delimiter splitting
// ---------------------------------------------------------------------------

fn bench_splitting(c: &mut Criterion) {
    let text = make_large_text(10);

    let mut group = c.benchmark_group("delimiter_splitting");

    // Single delimiter
    group.bench_function("split_at_delimiters/1_delim", |b| {
        b.iter(|| {
            let offsets = split_at_delimiters(black_box(&text), b".", IncludeDelim::Prev, 0);
            black_box(offsets);
        });
    });

    // Three delimiters
    group.bench_function("split_at_delimiters/3_delims", |b| {
        b.iter(|| {
            let offsets = split_at_delimiters(black_box(&text), b".!?", IncludeDelim::Prev, 0);
            black_box(offsets);
        });
    });

    // Five+ delimiters (triggers lookup table path)
    group.bench_function("split_at_delimiters/5_delims", |b| {
        b.iter(|| {
            let offsets = split_at_delimiters(black_box(&text), b".!?\n;", IncludeDelim::Prev, 0);
            black_box(offsets);
        });
    });

    // IncludeDelim modes
    for mode in [
        ("prev", IncludeDelim::Prev),
        ("next", IncludeDelim::Next),
        ("none", IncludeDelim::None),
    ] {
        group.bench_function(format!("include_delim/{}", mode.0), |b| {
            b.iter(|| {
                let offsets = split_at_delimiters(black_box(&text), b".", mode.1, 0);
                black_box(offsets);
            });
        });
    }

    // Multi-byte pattern splitting
    group.bench_function("split_at_patterns/1_pattern", |b| {
        let patterns: &[&[u8]] = &[b". "];
        b.iter(|| {
            let offsets = split_at_patterns(black_box(&text), patterns, IncludeDelim::Prev, 0);
            black_box(offsets);
        });
    });

    group.bench_function("split_at_patterns/3_patterns", |b| {
        let patterns: &[&[u8]] = &[b". ", b"? ", b"! "];
        b.iter(|| {
            let offsets = split_at_patterns(black_box(&text), patterns, IncludeDelim::Prev, 0);
            black_box(offsets);
        });
    });

    // Compiled PatternSplitter
    group.bench_function("pattern_splitter/compiled_3_patterns", |b| {
        let splitter = PatternSplitter::new(&[b". ", b"? ", b"! "]);
        b.iter(|| {
            let offsets = splitter.split(black_box(&text), IncludeDelim::Prev, 0);
            black_box(offsets);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: Token-aware merging
// ---------------------------------------------------------------------------

fn bench_merging(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_aware_merging");

    // find_merge_indices with varying segment counts
    for n in [100, 1000, 10000] {
        let token_counts: Vec<usize> = (0..n).map(|i| 10 + (i % 50)).collect();

        group.bench_function(format!("find_merge_indices/n_{n}"), |b| {
            b.iter(|| {
                let indices = find_merge_indices(black_box(&token_counts), 200);
                black_box(indices);
            });
        });
    }

    // merge_splits with synthetic data
    for n in [50, 200, 1000] {
        let splits: Vec<&str> = (0..n).map(|_| "Hello world. ").collect();
        let token_counts: Vec<usize> = (0..n).map(|i| 5 + (i % 20)).collect();

        group.bench_function(format!("merge_splits/n_{n}"), |b| {
            b.iter(|| {
                let result = merge_splits(black_box(&splits), black_box(&token_counts), 100);
                black_box(result);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: Signal processing (savgol)
// ---------------------------------------------------------------------------

fn bench_signal_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_processing");

    // savgol_filter with different window sizes
    let data_500 = make_synthetic_similarities(500);

    for window in [5, 11, 21] {
        group.bench_function(format!("savgol_filter/w_{window}_n_500"), |b| {
            b.iter(|| {
                let result = savgol_filter(black_box(&data_500), window, 3, 0);
                black_box(result);
            });
        });
    }

    // savgol_filter with derivative
    group.bench_function("savgol_filter/deriv_1/w_11_n_500", |b| {
        b.iter(|| {
            let result = savgol_filter(black_box(&data_500), 11, 3, 1);
            black_box(result);
        });
    });

    // windowed_cross_similarity with varying n
    for n in [100, 500, 1000] {
        let d = 128;
        let embeddings = make_synthetic_embeddings(n, d);

        group.bench_function(format!("windowed_cross_similarity/n_{n}_d_{d}"), |b| {
            b.iter(|| {
                let result = windowed_cross_similarity(black_box(&embeddings), n, d, 3);
                black_box(result);
            });
        });
    }

    // windowed_cross_similarity with different window sizes
    let n = 500;
    let d = 128;
    let embeddings = make_synthetic_embeddings(n, d);
    for window in [3, 5, 11] {
        group.bench_function(format!("windowed_cross_similarity/n_{n}_w_{window}"), |b| {
            b.iter(|| {
                let result = windowed_cross_similarity(black_box(&embeddings), n, d, window);
                black_box(result);
            });
        });
    }

    // find_local_minima_interpolated
    let data_200 = make_synthetic_similarities(200);
    group.bench_function("find_local_minima/n_200", |b| {
        b.iter(|| {
            let result = find_local_minima_interpolated(black_box(&data_200), 11, 3, 0.1);
            black_box(result);
        });
    });

    let data_1000 = make_synthetic_similarities(1000);
    group.bench_function("find_local_minima/n_1000", |b| {
        b.iter(|| {
            let result = find_local_minima_interpolated(black_box(&data_1000), 11, 3, 0.1);
            black_box(result);
        });
    });

    // filter_split_indices
    let indices: Vec<usize> = (0..100).map(|i| i * 5).collect();
    let values: Vec<f64> = (0..100)
        .map(|i| 0.5 + 0.4 * (i as f64 * 0.3).sin())
        .collect();

    group.bench_function("filter_split_indices/n_100", |b| {
        b.iter(|| {
            let result = filter_split_indices(black_box(&indices), black_box(&values), 0.5, 3);
            black_box(result);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 5: Markdown block splitting
// ---------------------------------------------------------------------------

fn bench_block_splitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("markdown_blocks");

    group.bench_function("split_blocks/sample", |b| {
        b.iter(|| {
            let blocks = split_blocks(black_box(SAMPLE_TEXT));
            black_box(blocks);
        });
    });

    let large_md = SAMPLE_TEXT.repeat(20);
    group.bench_function("split_blocks/large_20x", |b| {
        b.iter(|| {
            let blocks = split_blocks(black_box(&large_md));
            black_box(blocks);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 6: Sentence splitting
// ---------------------------------------------------------------------------

fn bench_sentence_splitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentence_splitting");

    let short = "Hello world. This is a test. Another sentence here.";
    group.bench_function("split_sentences/short", |b| {
        b.iter(|| {
            let sentences = split_sentences(black_box(short));
            black_box(sentences);
        });
    });

    let medium = "First sentence. Second sentence. Third one. Fourth here. Fifth now. \
                   Sixth comes next. Seventh is here. Eighth follows. Ninth appears. Tenth ends."
        .repeat(10);
    group.bench_function("split_sentences/medium", |b| {
        b.iter(|| {
            let sentences = split_sentences(black_box(&medium));
            black_box(sentences);
        });
    });

    let large = SAMPLE_TEXT.repeat(50);
    group.bench_function("split_sentences/large", |b| {
        b.iter(|| {
            let sentences = split_sentences(black_box(&large));
            black_box(sentences);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_chunking,
    bench_splitting,
    bench_merging,
    bench_signal_processing,
    bench_block_splitting,
    bench_sentence_splitting,
);
criterion_main!(benches);
