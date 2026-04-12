# Enriched Chunking: Self-Describing Chunks for RAG

Standard chunking produces text fragments. A retrieval system embeds them, indexes them, and hopes that the embedding captures enough of what the chunk is about. But embeddings are lossy. A chunk about dosing protocols for a specific compound might embed close to a chunk about clinical trial phases in general -- both are "pharmaceutical" in embedding space, but they answer very different questions.

Enriched chunking produces chunks that describe themselves. Each chunk carries a title, a summary, a keyword list, typed entities, hypothetical questions the chunk can answer, semantic keys for cross-chunk linking, and a category label. These metadata fields enable retrieval strategies that go beyond dense vector similarity: BM25 over keywords and titles, HyDE-style matching against hypothetical questions, entity-type filtering, and category-based routing.

The key insight is that all seven metadata fields can be extracted in a single LLM call per chunk, and the semantic keys enable a recombination step that merges chunks sharing the same conceptual thread.

## How the pipeline works

**Initial grouping** divides the document into preliminary chunks using simple structural rules. Blocks accumulate until the soft token budget is reached. Headings start new chunks. Atomic blocks (tables, code blocks, complete lists) are never split. No embeddings are needed -- this step is purely structural.

**Single-call LLM enrichment** sends each preliminary chunk to the LLM with a structured output schema requesting all seven fields at once. The prompt includes the chunk text and a rolling semantic key dictionary -- the set of keys assigned to previously processed chunks. This context lets the LLM reuse existing keys when a new chunk covers the same concept, creating explicit links between chunks.

The seven metadata fields are:

- **title**: a concise descriptive title for the chunk content
- **summary**: a 1-2 sentence summary of what the chunk covers
- **keywords**: terms that a user might search for when looking for this content
- **typed_entities**: named entities with type labels (person, organization, compound, location, etc.)
- **hypothetical_questions**: 2-4 questions that this chunk can answer, suitable for HyDE retrieval
- **semantic_keys**: normalized concept identifiers (e.g., "xr-7742-dosing", "clinical-protocol") that link chunks covering the same topic
- **category**: a single label classifying the chunk's role (background, methodology, results, discussion, etc.)

**Key-based recombination** examines the semantic key dictionary after all chunks are enriched. Chunks sharing identical keys are candidates for merging. The recombination uses a bin-packing strategy: same-key chunks that are also adjacent in document order are merged first, then non-adjacent same-key chunks are merged if the combined size fits within the hard budget. Chunks with unique keys remain untouched.

**Re-enrichment** runs only on chunks that were actually merged. Since the merged chunk has new content, its title and summary are updated via a lightweight LLM call. Keywords, entities, questions, and keys from the constituent chunks are preserved as a union -- no information is lost.

## The rolling key dictionary

The rolling key dictionary is what makes enriched chunking more than just "chunking plus metadata." As the LLM processes chunks sequentially, it sees which concepts have already been named. When chunk 5 discusses the same dosing protocol as chunk 1, the LLM reuses the key "xr-7742-dosing" rather than inventing a new one. This creates an explicit link between the two chunks without requiring embeddings or entity matching.

The dictionary is a map from key names to lists of chunk indices. After processing, it provides a concept-level index of the document: "xr-7742-dosing" appears in chunks [0, 3, 7], "clinical-protocol" appears in chunks [0, 1, 3]. This is directly usable for graph-based retrieval or for augmenting a vector index with concept links.

## Configuration

**`--enrichment-model`** selects the LLM for enrichment calls. The default is `gpt-4.1-mini`.

**`--soft-budget`** and **`--hard-budget`** control initial grouping sizes and the hard ceiling for recombined chunks.

**`--no-recombine`** skips the key-based recombination step, producing chunks with metadata but no merging.

**`--no-re-enrich`** skips the re-enrichment of merged chunks, keeping the original titles and summaries from the constituent chunks.

## CLI usage

```sh
# Enriched chunking with metadata extraction
cognigraph-chunker enriched -i document.md --api-key $OPENAI_API_KEY

# Custom model and budgets
cognigraph-chunker enriched -i doc.md --api-key $KEY \
  --enrichment-model gpt-4.1-mini --soft-budget 256 --hard-budget 512

# Skip recombination (metadata only, no merging)
cognigraph-chunker enriched -i doc.md --api-key $KEY --no-recombine

# JSON output
cognigraph-chunker enriched -i doc.md --api-key $KEY -f json
```

## API usage

```
POST /api/v1/enriched

{
  "text": "...",
  "enrichment_model": "gpt-4.1-mini",
  "soft_budget": 512,
  "hard_budget": 768,
  "recombine": true,
  "re_enrich": true
}
```

The response includes chunks with all seven metadata fields plus the semantic key dictionary.

## When to use enriched chunking

Use it when your retrieval pipeline supports hybrid search (BM25 + dense vectors), when you need HyDE-style retrieval (matching queries against hypothetical questions), or when chunks need to be self-describing for downstream consumers that cannot access the original document.

Enriched chunking is especially valuable for knowledge bases where chunks are stored independently of their source documents. The title, summary, and category provide enough context for a human to understand what a chunk is about without reading it. The hypothetical questions provide alternative query surfaces for retrieval. The typed entities enable faceted filtering.

It is less useful when retrieval is purely dense-vector-based and metadata fields would go unused, or when LLM cost per chunk is a concern. Each chunk requires one LLM call for enrichment (and potentially a second for re-enrichment after merging), so the cost scales linearly with chunk count.

Enriched chunking does not require an embedding provider. It relies on the LLM for metadata extraction and on structural heuristics for initial grouping. This makes it a good choice when you want rich metadata without the cost or complexity of running an embedding model.

## Enriched chunking for enterprise search

Enterprise knowledge bases accumulate documents from many sources: wikis, runbooks, policy documents, incident reports. These documents use different terminology for the same concepts. A runbook might call it "the primary database," a policy document might call it "the production data store," and an incident report might call it "prod-db-01."

The semantic key dictionary normalizes these references. When the LLM enriches a runbook chunk about failover procedures, it assigns the key "production-database-failover." When it encounters an incident report chunk describing a database outage, it reuses the same key. The result is a concept-level index that bridges vocabulary differences across document sources -- something that embedding similarity alone cannot reliably achieve.

The hypothetical questions field is particularly useful for enterprise search. Users searching an internal knowledge base often phrase queries as questions: "How do I reset the database password?" or "What is the escalation procedure for a P1 incident?" The hypothetical questions generated by the LLM match these natural query patterns, providing a retrieval surface that embeddings of the chunk text alone may not capture.
