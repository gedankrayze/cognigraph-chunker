# Token-Aware Merging: Right-Sizing Chunks for Your Model

Chunking strategies produce segments of varying sizes. Delimiter splitting creates one segment per sentence — some are 10 tokens, others are 200. Semantic chunking groups blocks at topic boundaries, but a topic that spans two sentences produces a small chunk while a topic that spans twenty produces a large one. Even fixed-size chunking with delimiter awareness produces the occasional short final chunk.

These size variations create a practical problem. Embedding models have optimal input ranges: too short, and the embedding is sparse and uninformative; too long, and the model truncates or the embedding averages over too much content. Most retrieval systems work best when chunks are roughly similar in size, somewhere in the 100-to-500 token range depending on the model and use case.

Token-aware merging is a post-processing step that takes the output of any chunking strategy and combines adjacent small chunks into groups that fit within a specified token budget. It doesn't change the order of the text or modify the content — it only decides where to draw the boundaries between output chunks.

## The merging algorithm

The algorithm is greedy and runs in a single forward pass. It maintains a running count of tokens in the current group. For each input chunk, it checks whether adding that chunk's token count would exceed the budget. If it fits, the chunk joins the current group. If it doesn't, the current group is closed and a new group begins with this chunk.

The key insight that makes this efficient is the use of cumulative token counts with binary search. Instead of scanning forward one chunk at a time, the algorithm precomputes a prefix sum of all token counts and uses binary search to find the furthest chunk boundary that stays within the budget from the current position. This reduces the worst case from O(n^2) to O(n log n), though in practice the constant factors are small enough that the difference is negligible for typical document sizes.

Here's how it works conceptually. Given chunks with token counts [2, 3, 1, 4, 2, 3] and a budget of 6:

The cumulative sums are [0, 2, 5, 6, 10, 12, 15]. Starting from position 0, binary search finds that the cumulative sum at position 3 (value 6) is the last position within budget. So the first group contains chunks 0 through 2 (tokens: 2+3+1 = 6). Starting from position 3, the cumulative sum at position 4 minus position 3 is 4, position 5 minus position 3 is 6. So the second group contains chunks 3 and 4 (tokens: 4+2 = 6). The last chunk (3 tokens) forms its own group.

## Token counting

A precise token count requires running the text through the actual tokenizer of the target model. Different models tokenize differently: GPT-4's tokenizer produces different counts than BERT's, which differs from Llama's.

For the purpose of chunk merging, a precise count is rarely necessary. The goal is to produce groups that are roughly within budget, not to hit an exact number. Whitespace-based token counting — splitting on spaces and counting the pieces — provides a fast approximation that's typically within 10-20% of the true token count for English text. It overestimates for languages with long compound words and underestimates for text with many short words, but the error is consistent enough that the merge boundaries end up in reasonable places.

This approximation runs in microseconds with no model dependency, making merging a lightweight post-processing step that doesn't require loading a tokenizer or calling an API.

## How merging interacts with each strategy

**Fixed-size chunking + merge**: This combination is useful when the fixed-size target (in bytes) doesn't map cleanly to a token count. You might chunk at 2048 bytes to get manageable pieces, then merge into 512-token groups for your embedding model. The fixed-size step handles boundary detection, and the merge step handles token budgeting.

**Delimiter splitting + merge**: This is the most natural combination. Delimiter splitting produces maximally coherent chunks (one per sentence), and merging groups them into token-appropriate sizes. The result is chunks where every boundary falls on a sentence boundary and every chunk is close to the target token count. This gives you the best of both worlds: semantic coherence and size uniformity.

**Semantic chunking + merge**: When the semantic pipeline produces very small topic-aligned chunks (a topic that spans only two sentences), merging can combine adjacent chunks that are topically similar anyway. This is most useful when the semantic chunker's threshold is aggressive (low), producing many fine-grained splits that are individually too small for good embeddings.

## The merge result

Merging produces two outputs: the merged text segments and the token count for each merged group. The text segments are the concatenation of the original chunks in each group, preserving the original order and whitespace. The token counts let you verify that the merging achieved the desired budget and identify any groups that exceeded it (which happens when a single input chunk is already larger than the budget — in that case, it becomes its own group unchanged).

## When not to merge

Merging assumes that adjacent chunks can be meaningfully combined. This is true when the chunks come from the same document and are in order. It's less appropriate when chunks have been shuffled, when they come from different documents, or when the boundaries between them carry semantic weight that would be lost by combining them.

If your semantic chunker has already produced well-sized chunks that fit your token budget, merging adds no value and slightly obscures the topic boundaries the pipeline worked to find. In general, apply merging when chunk sizes are too variable or too small for your downstream needs, and skip it when the chunking strategy already produces appropriately sized output.

## The complete picture

With all five concepts covered in this series — the fundamental need for chunking, fixed-size and delimiter strategies for structural splitting, semantic analysis for topic-aligned boundaries, markdown-aware parsing for preserving document structure, and token-aware merging for size normalization — you have the building blocks for a chunking pipeline tailored to your use case.

The practical recipe for most RAG applications: parse your documents with markdown awareness, apply semantic chunking to find topic boundaries, merge small chunks into token-appropriate groups, and embed the result. Tune the parameters (similarity window, smoothing window, threshold, merge budget) on a sample of your actual documents until retrieval quality meets your needs. The defaults are a reasonable starting point; your data will tell you which direction to adjust.
