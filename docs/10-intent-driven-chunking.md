# Intent-Driven Chunking: Optimizing for Retrieval

Most chunking methods ask where to split based on properties of the text itself: topic boundaries, structural markers, entity chains. The resulting chunks reflect the author's organization of the document. But the people searching the document have their own questions, and those questions rarely align perfectly with how the author structured the content.

A clinical trial protocol might organize information by study phase: background, design, endpoints, analysis plan. A regulatory reviewer searching for "what are the primary efficacy endpoints?" needs a chunk that contains the endpoint definition, the measurement schedule, and the analysis method -- information that spans parts of two or three author-defined sections. A semantic chunker produces chunks that follow the author's structure. An intent-driven chunker produces chunks that follow the reader's questions.

Intent-driven chunking optimizes boundaries for predicted user queries rather than topic transitions. It asks: given the likely information needs of people who will search this document, which partition maximizes the chance that each query retrieves a single, self-contained chunk?

## How it works

The pipeline has three stages beyond the standard block extraction: intent prediction, alignment scoring, and dynamic programming.

**Intent prediction** sends the document (or a summary of it, for long documents) to an LLM with a structured output schema. The LLM generates 10-30 hypothetical queries that users might ask about this document. Each query is classified by type: factual ("What is the recommended dose?"), procedural ("How do I configure the pipeline?"), conceptual ("Why does the system use valley detection?"), or comparative ("How does cognitive chunking differ from semantic chunking?"). The diversity of query types ensures that the resulting partition serves different retrieval patterns, not just factual lookups.

**Alignment scoring** measures how well a candidate chunk answers a predicted query. For each candidate chunk (a contiguous range of blocks), the pipeline computes a centroid embedding -- the mean of the block embeddings in that range -- and calculates the cosine similarity between the centroid and every intent embedding. The alignment score for that chunk is the maximum similarity to any intent. A high score means the chunk is tightly focused on at least one predicted query. A low score means the chunk is a grab bag that doesn't cleanly answer anything.

**Dynamic programming** finds the globally optimal partition. Unlike greedy approaches that make local decisions at each boundary, the DP explores all valid partitions (subject to minimum and maximum chunk sizes derived from the token budgets) and selects the one that maximizes the total alignment score across all chunks, normalized by chunk count. The time complexity is O(n * max_blocks) where n is the number of blocks, making it practical for documents up to tens of thousands of blocks.

The DP formulation is the key differentiator. Greedy chunking can get trapped: a locally optimal boundary at position 50 might force a poor boundary at position 80. The DP avoids this by evaluating the downstream consequences of every boundary decision.

## What the output contains

Each intent chunk carries its text, byte offsets, token estimate, heading path, and two fields specific to this method: the index of the best-matching intent and the alignment score. The result also includes the full list of predicted intents with their matched chunk indices, plus the overall partition score.

The partition score is a single number summarizing retrieval quality. Higher is better. When comparing different parameter settings (more or fewer intents, different token budgets), the partition score provides a direct measure of which configuration produces more retrieval-friendly chunks.

## Configuration

**`--max-intents`** controls how many hypothetical queries the LLM generates. The default of 20 works well for most documents. Very short documents may benefit from fewer (10), and very long documents with many distinct topics may benefit from more (30). More intents increase the chance that every chunk aligns well with at least one query, but they also increase the embedding cost.

**`--soft-budget`** and **`--hard-budget`** control chunk sizes. The DP uses these to derive the minimum and maximum number of blocks per chunk. The soft budget (default 512 tokens) is the target; the hard budget (default 768 tokens) is the ceiling.

**`--intent-model`** selects the LLM used for intent generation. The default is `gpt-4.1-mini`, which balances quality and cost. Any OpenAI-compatible model works.

## CLI usage

```sh
# Intent-driven chunking with OpenAI embeddings
cognigraph-chunker intent -i document.md -p openai --api-key $OPENAI_API_KEY

# Custom intent count and token budgets
cognigraph-chunker intent -i doc.md -p openai --api-key $KEY \
  --max-intents 30 --soft-budget 256 --hard-budget 512

# Use a different LLM for intent generation
cognigraph-chunker intent -i doc.md -p openai --api-key $KEY \
  --intent-model gpt-4.1-mini

# JSON output with post-merge
cognigraph-chunker intent -i doc.md -p openai --api-key $KEY -f json --merge
```

## API usage

```
POST /api/v1/intent

{
  "text": "...",
  "provider": "openai",
  "model": "text-embedding-3-small",
  "intent_model": "gpt-4.1-mini",
  "max_intents": 20,
  "soft_budget": 512,
  "hard_budget": 768
}
```

The response includes chunks with alignment scores, the predicted intents with their matched chunk indices, and the overall partition score.

## When to use intent-driven chunking

Use it when retrieval quality is the primary objective and you can tolerate the cost of an LLM call plus embedding the generated intents. It works best for documents where users have diverse, specific information needs -- reference manuals, knowledge bases, FAQ compilations, compliance documents.

It is less useful for documents where the structure itself is the information (deeply nested specifications, legal contracts with numbered clauses) or where the document is short enough that a single chunk suffices.

Intent-driven chunking requires both an LLM (for intent generation) and an embedding provider (for alignment scoring). This makes it the most expensive method per document, but the DP optimization means the actual chunking step is fast -- the cost is dominated by the LLM and embedding calls.

## Intent-driven chunking in regulated industries

Pharmaceutical regulatory submissions are dense documents where reviewers search with specific questions: "What is the primary endpoint?", "What were the inclusion criteria?", "How was the DSMB constituted?" These are precisely the kinds of queries the intent predictor generates. The resulting chunks align with regulatory review patterns rather than the document's organizational structure, which means a RAG system built on intent-driven chunks returns more complete answers to reviewer queries.

The alignment scores provide an auditable quality signal. A chunk with a low alignment score is a warning: it doesn't cleanly answer any predicted query, which means it may need manual review or restructuring. In validated environments, this metric can be logged alongside the chunking output as part of the processing record.
