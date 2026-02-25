# Semantic Chunking: Finding Topic Boundaries with Embeddings

Fixed-size and delimiter-based chunking operate on the surface structure of text: byte positions, punctuation marks, whitespace patterns. They work well for many use cases, but they can't tell the difference between a paragraph that continues the same topic and one that shifts to something entirely new. Semantic chunking bridges that gap by using the meaning of the text itself to decide where to split.

The core insight is that adjacent text blocks about the same topic produce similar embeddings, while a topic shift causes a drop in similarity. By measuring this similarity across the document and looking for valleys in the signal, we can identify natural topic boundaries.

## The pipeline

Semantic chunking is not a single operation but a signal processing pipeline with six stages. Each stage transforms the output of the previous one, progressively refining the raw text into a set of topic-aligned chunks.

### Stage 1: Block extraction

The first step breaks the document into small, meaningful units called blocks. In a markdown document, the parser walks the AST and extracts headings, table rows, code blocks, list items, and individual sentences from paragraphs. In plain text, Unicode sentence segmentation splits the text at sentence boundaries.

Each block is the smallest unit that will be embedded. The granularity matters: too coarse (whole paragraphs), and topic shifts within a paragraph go undetected. Too fine (individual words), and the embeddings lack enough context to be meaningful. Sentence-level granularity with atomic treatment of structural elements (tables, code) strikes a practical balance.

### Stage 2: Embedding

Every block gets passed through an embedding model that converts it into a dense vector — a list of floating-point numbers representing its meaning in a high-dimensional space. Blocks about similar topics will produce vectors that point in roughly the same direction; blocks about different topics will point in different directions.

The choice of embedding model affects both quality and performance. Small local models (like all-MiniLM-L6-v2 running in ONNX Runtime) produce 384-dimensional vectors quickly with no network calls. Cloud APIs (like OpenAI's text-embedding-3-small) produce higher-dimensional vectors that may capture subtler distinctions but add latency and cost. For most chunking purposes, a lightweight local model is sufficient — we're measuring relative similarity between adjacent blocks, not absolute semantic quality.

### Stage 3: Windowed cross-similarity

With embeddings in hand, the next step measures how similar each pair of adjacent blocks is. The simplest approach would compute cosine similarity between block N and block N+1 for every N. But this point-to-point measurement is noisy: a single unusual sentence can create a false dip in similarity even when the surrounding text is about the same topic.

Windowed cross-similarity addresses this by averaging the similarity over a local window. For a window of size 3 centered on the boundary between blocks 5 and 6, it computes the average cosine similarity between all consecutive pairs in blocks 4 through 7. This smooths out sentence-level noise while preserving genuine topic transitions.

The output is a distance curve: one value for each boundary between adjacent blocks. High values mean the blocks on either side are similar (same topic). Low values mean they're dissimilar (potential topic shift). The curve has N-1 values for N blocks.

### Stage 4: Savitzky-Golay smoothing

Even with windowed averaging, the distance curve can be noisy. Small fluctuations in embedding similarity create spurious dips that don't correspond to real topic changes. The Savitzky-Golay filter addresses this by fitting a local polynomial to the curve and replacing each point with the polynomial's value.

The filter has two key parameters: **window size** (how many neighbors contribute to each smoothed value) and **polynomial order** (the degree of the fitted polynomial). A larger window produces more aggressive smoothing, filtering out smaller fluctuations but potentially blurring genuine boundaries. A higher polynomial order preserves sharper features but follows noise more closely.

The default parameters (window 11, polynomial order 3) work well for typical documents of 20 to 200 blocks. For very short documents, the window is automatically clamped to the data length to prevent the filter from operating on padded values.

What makes the Savitzky-Golay filter particularly well-suited here is that it preserves the location and shape of real minima while flattening noise. Unlike a simple moving average, which shifts peak positions and broadens them, the polynomial fitting maintains the true positions of topic boundaries.

### Stage 5: Local minima detection

Topic boundaries correspond to valleys (local minima) in the smoothed distance curve — points where similarity is locally at its lowest, meaning the text on either side is about different things.

Finding these minima uses the first and second derivatives of the smoothed curve, both computed analytically via the Savitzky-Golay filter itself (which can compute derivatives of any order). A point is a local minimum when the first derivative is approximately zero (the curve is flat) and the second derivative is positive (the curve is concave up — it's a valley, not a peak).

The tolerance parameter controls how close to zero the first derivative needs to be. A tight tolerance finds only sharp, well-defined minima. A loose tolerance also captures broad, shallow valleys that might represent gradual topic transitions.

### Stage 6: Filtering and thresholding

Not every local minimum is a meaningful topic boundary. Some are shallow dips — the similarity decreased slightly but the topic didn't really change. The filtering stage uses two criteria to select which minima become actual split points.

**Percentile threshold**: The filter computes the value at a given percentile of all detected minima. Only minima below this threshold are kept. A threshold of 0.5 means only the deeper-than-median minima survive. Lower thresholds (0.3, 0.2) are more selective, producing fewer, higher-confidence splits. Higher thresholds (0.7, 0.8) produce more splits, capturing subtler topic shifts.

**Minimum distance**: Adjacent split points must be separated by at least this many blocks. A minimum distance of 2 prevents the chunker from creating single-block chunks. A larger minimum distance (5, 10) produces longer chunks with fewer boundaries, which can be useful when downstream processing prefers larger context windows.

## Tuning the parameters

The pipeline has several knobs, and the right settings depend on your documents and your use case.

**sim_window** (cross-similarity window, default 3): Controls how much local averaging happens before the Savitzky-Golay filter. Larger values (5, 7) smooth more aggressively at the embedding level. Should always be odd and at least 3.

**sg_window** (Savitzky-Golay window, default 11): Controls the smoothing of the distance curve. Larger values smooth more. Must be odd and greater than poly_order.

**poly_order** (polynomial order, default 3): Higher values follow local variations more closely. Should be less than sg_window. Order 2 or 3 works for most cases.

**threshold** (percentile threshold, default 0.5): Controls how many minima become split points. Lower = fewer, more confident splits. Higher = more splits.

**min_distance** (minimum block gap, default 2): Prevents tiny chunks. Increase this if your chunks are too small.

For a first pass, the defaults work reasonably well. If you're getting too many chunks, lower the threshold or increase min_distance. If you're getting too few, raise the threshold.

## When to use semantic chunking

Semantic chunking is the most computationally expensive strategy. It requires an embedding model (either local or remote), and the cost scales linearly with the number of blocks. For a 1000-sentence document with a local ONNX model, this takes seconds rather than microseconds.

Use it when topic alignment matters — when your retrieval system needs chunks that correspond to coherent topics rather than arbitrary size buckets. This is especially valuable for long-form content with multiple topics (research papers, documentation, reports) where fixed-size chunking would blend unrelated content into single chunks.

For short, topically focused documents, or for high-throughput pipelines where latency matters more than chunk quality, fixed-size or delimiter-based chunking is the better choice. You can always combine strategies: use delimiter splitting for initial segmentation and then run semantic analysis on the resulting segments to merge related ones.
