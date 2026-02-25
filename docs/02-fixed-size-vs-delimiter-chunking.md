# Fixed-Size vs. Delimiter-Based Chunking: Choosing the Right Strategy

Text chunking strategies fall into two structural families: those that target a specific chunk size and those that split at every occurrence of a delimiter. Both operate on the raw bytes of the text without understanding meaning, but they make fundamentally different tradeoffs between uniformity and coherence. Understanding those tradeoffs helps you pick the right one for your use case.

## Fixed-size chunking with boundary awareness

The idea is simple: walk through the text in steps of N bytes, and each step produces one chunk. A naive implementation just cuts at byte N, 2N, 3N, and so on. The result is predictably sized chunks, but those cuts land in arbitrary places — often in the middle of words or sentences.

Boundary-aware chunking improves on this by searching backward from the cut point to find a nearby delimiter. Instead of cutting at exactly byte 4096, it looks for the last newline, period, or question mark before that position and splits there instead. The chunk ends up slightly smaller than the target, but it ends at a natural text boundary.

This backward search is the core mechanism. When the text contains delimiters — and most text contains plenty of sentence-ending punctuation and newlines — the algorithm almost always finds a good split point within the target window. Only when a stretch of text contains no delimiters at all (an extremely long word or a binary blob) does the algorithm resort to a hard split at the target size.

### Delimiter search performance

The efficiency of this backward search matters when chunking large documents. For one to three single-byte delimiters, SIMD-accelerated byte search (via `memchr`) scans memory at near-hardware speed. For four or more delimiters, a 256-entry lookup table maps every possible byte value to a boolean, and a simple backward scan checks each byte against the table. Both approaches are branch-light and cache-friendly, so even multi-megabyte documents chunk in microseconds.

### Prefix vs. suffix mode

When the chunker finds a delimiter, it has a choice: does the delimiter belong to the current chunk or the next one? In **suffix mode** (the default), the delimiter stays at the end of the current chunk. "Hello." becomes one chunk, " World." becomes the next. In **prefix mode**, the delimiter moves to the start of the next chunk. "Hello" becomes the first chunk, ". World" becomes the next.

This matters for downstream processing. If you're feeding chunks into a sentence classifier that expects each chunk to end with punctuation, suffix mode is the right choice. If you're splitting on section-starting markers like headings or bullet points, prefix mode ensures those markers stay with the content they introduce.

### Consecutive delimiter handling

Text often contains runs of the same delimiter: double newlines between paragraphs, chains of dots in a table of contents, sequences of dashes in a separator. Without special handling, the chunker might split in the middle of `\n\n\n`, producing a chunk that ends with `\n` and another that starts with `\n\n`. Neither is ideal.

Consecutive mode addresses this by finding the start of a delimiter run rather than splitting within it. When it encounters `\n\n\n`, it identifies the position before the first `\n` as the split point, keeping the entire run intact on one side of the boundary. Combined with prefix mode, this means paragraph breaks and section separators stay attached to the content that follows them.

### Forward fallback

Sometimes the backward search fails: the target window contains no delimiters. The default behavior is a hard split at the target size. But with forward fallback enabled, the chunker searches forward from the target point instead, looking for the next delimiter after the boundary. This produces a chunk larger than the target size but ensures it ends at a natural boundary.

Forward fallback is useful when your text has long stretches without delimiters (technical identifiers, URLs, base64-encoded data) interspersed with normal prose. The occasional oversized chunk is a better tradeoff than splitting in the middle of a URL.

## Delimiter-based splitting

Delimiter splitting takes a different approach entirely. Instead of targeting a size and adjusting the boundary, it splits at every occurrence of the specified delimiters. Every period, every newline, every question mark — each one produces a new segment.

The result is chunks that are maximally coherent: each one corresponds to a sentence, a paragraph, or whatever unit the delimiters define. But the sizes are unpredictable. A sentence might be 10 bytes or 500 bytes. A paragraph might be 50 bytes or 5000.

### Single-byte vs. multi-byte patterns

Single-byte delimiters (like `.`, `?`, `\n`) use the same SIMD-accelerated search as the fixed-size chunker, scanning forward through the text to find every occurrence. Multi-byte patterns (like `". "`, `"? "`, or even `"\n\n"`) require a different approach: an Aho-Corasick automaton built from all specified patterns. This automaton scans the text once and finds all pattern occurrences in a single pass, regardless of how many patterns you specify.

The Aho-Corasick approach is particularly useful for splitting on sentence boundaries in natural text, where the delimiter isn't just a period but a period followed by a space. Splitting on `.` alone would break abbreviations like "Dr." and "U.S.", while splitting on `". "` correctly identifies sentence boundaries.

### Where does the delimiter go?

Just like fixed-size chunking has prefix and suffix modes, delimiter splitting has three options for what happens to the delimiter itself:

**Previous** (default): the delimiter stays at the end of the preceding segment. `"Hello. World."` splits into `"Hello."` and `" World."`.

**Next**: the delimiter moves to the start of the following segment. `"Hello. World."` splits into `"Hello"` and `". World"` and `"."`.

**None**: the delimiter is dropped from both segments. `"Hello. World."` splits into `"Hello"` and `" World"`.

### Minimum character threshold

Because delimiter splitting produces segments of arbitrary size, some of them can be very small — a one-word sentence, a single-character line, an empty string between consecutive delimiters. The minimum character threshold addresses this by accumulating segments until the accumulated length reaches the threshold. A `min_chars` of 100 means no segment will be shorter than 100 characters (except possibly the last one).

This is a lightweight alternative to the full token-aware merging discussed later in the series. It doesn't count tokens or respect model limits, but it prevents the production of degenerate tiny segments that would produce poor embeddings.

## When to use which

**Fixed-size chunking** is the right choice when you need predictable chunk sizes. Embedding models often have fixed token limits (512 tokens is common), and fixed-size chunking with an appropriate byte target ensures every chunk fits. It's also the fastest strategy — the backward search is a single pass with no memory allocation beyond the output list.

**Delimiter splitting** is the right choice when semantic coherence matters more than size uniformity. If you're building a search index where each chunk should represent one complete thought, splitting on sentence boundaries gives you that guarantee. The variable sizes can be addressed with merging (combining small chunks into token-budget groups) as a post-processing step.

For plain text without markup, these two strategies cover most needs. When your documents have rich structure — headings, tables, code blocks — or when you want the chunking to reflect actual topic boundaries rather than punctuation, you'll want to look at semantic chunking, which we cover in the next article.
