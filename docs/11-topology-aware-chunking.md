# Topology-Aware Chunking: Preserving Document Structure

Some documents are flat sequences of paragraphs. Others have deep hierarchical structure: nested sections, subsections, cross-references between distant parts, tables that belong to the section that introduces them. Research papers, technical specifications, regulatory filings, and API documentation all share this property. Their structure carries meaning that flat chunking methods discard.

Semantic chunking treats the document as a stream of blocks and measures embedding similarity between neighbors. It has no concept of section boundaries, heading hierarchy, or cross-section dependencies. A subsection of three paragraphs under "Methods > Data Collection" might get merged with the first paragraph of "Methods > Analysis" because the embedding similarity is high -- they are both about methods. But structurally, they belong to different units, and a reader would never confuse them.

Topology-aware chunking builds an explicit representation of the document's hierarchical structure before making any boundary decisions. It then uses two LLM agents to classify sections and produce a final partition that respects the document's topology.

## The Structured Intermediate Representation

The first step after block extraction is constructing a Structured Intermediate Representation (SIR) -- a tree that mirrors the heading hierarchy with content blocks as leaves.

A document with headings like "Introduction," "Methods > Data Collection," "Methods > Analysis," and "Results" becomes a tree where "Methods" is a parent node with two children. Each leaf node corresponds to a content block (paragraph, table, code block, list). The SIR also includes cross-reference edges: if two blocks in different sections mention the same entity, an entity co-reference edge connects them. If a block starts with a discourse continuation marker ("As described above," "Building on this"), a discourse edge links it to the preceding section.

This construction reuses the heading context propagation, entity detection, and discourse marker detection from the cognitive chunking pipeline. No LLM calls are needed for SIR construction -- it is purely heuristic.

## Two-agent refinement

The SIR provides the structure. Two LLM agents make the boundary decisions.

**The Inspector Agent** receives the SIR as a JSON tree -- section titles, block types, block lengths, and cross-reference edges, but not the full text. It classifies each section node as one of three types:

- **Atomic**: the section must stay together as a single chunk. Typical for short definition sections, single-paragraph introductions, or sections that are already within the token budget.
- **Splittable**: the section is large enough to be divided at block boundaries. The Inspector identifies which blocks are potential split points.
- **Merge candidate**: the section is too small to stand alone and should be merged with an adjacent section. A single-paragraph subsection with no heading of its own is a common case.

The Inspector also identifies cross-section dependencies: pairs of sections that reference each other and should ideally end up in the same chunk or carry explicit cross-references.

**The Refiner Agent** receives the Inspector's classifications plus the SIR and the full text of sections classified as splittable. For splittable sections, it determines the optimal split points within the section. For merge candidate pairs, it decides the merge direction. For cross-section dependencies, it ensures that dependent content either stays together or gets explicit cross-reference annotations.

The Refiner outputs a partition: a list of chunk groups, each specifying which section IDs and block ranges it contains. The assembly stage maps these groups back to byte ranges in the original document.

## Context window handling

Long documents may produce a SIR that exceeds the LLM's context window. When the SIR JSON exceeds 80% of the model's context window, large content blocks are summarized to their first and last 100 characters plus token counts. Cross-reference edges are preserved even when block text is truncated. The Refiner receives full text only for sections classified as splittable, not the entire document.

## What the output contains

Each topology-aware chunk carries its text, byte offsets, token estimate, heading path, the Inspector's classification (atomic, splittable, or merged), and a list of cross-reference indices pointing to other chunks that share dependencies.

When `--emit-sir` is enabled (JSON output only), the response includes the full SIR structure -- useful for debugging or for downstream systems that want to reason about document topology.

## CLI usage

```sh
# Topology-aware chunking
cognigraph-chunker topo -i document.md --api-key $OPENAI_API_KEY

# Custom model and budgets
cognigraph-chunker topo -i doc.md --api-key $KEY \
  --topo-model gpt-4.1-mini --soft-budget 256 --hard-budget 512

# Include SIR in JSON output
cognigraph-chunker topo -i doc.md --api-key $KEY -f json --emit-sir
```

## API usage

```
POST /api/v1/topo

{
  "text": "...",
  "topo_model": "gpt-4.1-mini",
  "soft_budget": 512,
  "hard_budget": 768,
  "emit_sir": false
}
```

The response includes chunks with section classifications and cross-reference indices.

## When to use topology-aware chunking

Use it for deeply nested documents where the heading hierarchy carries structural meaning: research papers with sections and subsections, technical specifications with numbered clauses, API documentation with grouped endpoints, legal documents with articles and sub-articles.

It is less useful for flat documents (blog posts, narrative text, meeting transcripts) that lack heading structure. The pre-screening heuristic in adaptive mode skips topology-aware chunking when a document has fewer than two heading levels.

Topology-aware chunking requires an LLM (for the two agent calls) but does not require an embedding provider. This makes it a good choice when you need structure-preserving chunking without the cost of embedding every block.

## Topology-aware chunking for technical documentation

API documentation is a natural fit. An API reference might have a top-level "Authentication" section with subsections for "API Keys," "OAuth," and "JWT." Each subsection contains a description, a code example, and a table of parameters. Semantic chunking might split the code example from its parameter table because the embedding similarity between code and a markdown table is low. Topology-aware chunking keeps them together because the Inspector classifies the subsection as atomic -- it is small enough to be one chunk, and its internal blocks are structurally coupled.

For specifications with numbered requirements (ISO standards, NIST guidelines, internal compliance policies), the SIR preserves the numbering hierarchy. A requirement like "4.3.2.1 The system SHALL encrypt data at rest" stays connected to its parent requirement "4.3.2 Data Protection" and its sibling requirements. Cross-reference edges link requirements that reference each other, ensuring that a retrieval system can surface the full context for any requirement, not just the sentence that mentions it.
