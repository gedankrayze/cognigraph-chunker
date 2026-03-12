# Cognition-Aware Chunking: Preserving Meaning, Not Just Topics

Semantic chunking finds where topics change. It measures embedding similarity between adjacent blocks, smooths the signal, and splits at valleys. This works well for documents with clear topic transitions — a research paper shifting from methods to results, or a manual moving from installation to configuration.

But many documents don't have clean topic boundaries. A clinical trial protocol discusses the same compound across background, dosing, endpoints, and adverse events — the "topic" is the same throughout, yet splitting it into one giant chunk defeats the purpose of chunking. A forensic incident report describes a chain of events where each paragraph builds on the previous one through entity references ("the suspect," "this artifact," "the compromised host") and causal links ("consequently," "as a result"). Semantic chunking may split these chains at arbitrary points because the embedding similarity is uniformly high, while the actual cognitive structure — who did what, and what depends on what — gets severed.

Cognition-aware chunking addresses this by scoring boundaries on multiple dimensions beyond embedding similarity. It treats text as a graph of propositions: entities are nodes, relations are edges, and chunks are subgraphs that should be internally coherent. The question is not "did the topic change?" but "would a reader lose track of an entity, a causal chain, or a reference if we split here?"

## What it adds to semantic chunking

The cognitive pipeline extends the existing semantic pipeline rather than replacing it. The block extraction, embedding, similarity computation, and signal smoothing stages are identical. The difference is what happens at the boundary scoring and assembly stages.

In semantic chunking, a boundary decision is based on a single signal: smoothed embedding similarity. Below a threshold, split. Above it, join.

In cognitive chunking, the boundary decision is a weighted combination of eight signals:

**Semantic similarity** (weight 0.30) — the same embedding-based signal as semantic chunking. This remains the strongest individual signal because it captures topic continuity at the broadest level.

**Entity continuity** (weight 0.20) — measures how many named entities are shared across a candidate boundary. If block A introduces "Compound XR-7742" and block B discusses its dosing, splitting between them orphans the entity reference in block B. The entity continuity score penalizes such splits.

**Discourse continuation** (weight 0.10) — detects when a block starts with a continuation marker like "Furthermore," "However," "As a result," or "This approach." These markers signal that the block depends on prior context and should not start a new chunk.

**Heading continuity** (weight 0.10) — keeps headings attached to their first content block. A heading separated from its content is worse than useless in a vector store — it matches queries it can't answer.

**Structural affinity** (weight 0.05) — keeps structurally related blocks together. A list item shouldn't be separated from the preceding list items. A code block shouldn't be separated from the paragraph that introduces it.

**Topic shift penalty** (weight 0.15) — the inverse of semantic similarity, applied as a negative pressure. Sharp similarity drops push toward splitting.

**Orphan risk** (weight 0.20) — penalizes splits that would leave a chunk starting with a pronoun ("It processes the data") or a demonstrative ("This approach reduces latency") with no antecedent. These are the splits that make a reader — or an LLM — ask "what does 'it' refer to?"

**Budget pressure** (weight 0.10) — a soft signal that increases as accumulated tokens approach the soft budget ceiling. This prevents the other signals from creating unboundedly large chunks.

The weighted combination produces a **join score** for each boundary. Higher means "keep together." The assembly stage finds valleys in the join score curve (the same local-minima detection used in semantic chunking) and places splits there. A hard token ceiling forces additional splits if the join scores are uniformly high.

## Enrichment: what the pipeline knows about each block

Before scoring, each block is enriched with cognitive signals extracted through lightweight heuristics — no LLM calls, no external services.

**Entity detection** identifies named entities (capitalized multi-word spans like "Hamilton Depression Rating Scale"), repeated noun phrases, pronouns, and demonstratives. Entities are normalized to lowercase keys for overlap detection across blocks.

**Discourse marker detection** recognizes continuation markers, contrast markers, causal connectives, exemplification phrases, elaboration markers, and conclusion signals. Over 70 English patterns are recognized, plus equivalent patterns in 13 additional languages including German, French, Spanish, Russian, Chinese, Japanese, Korean, and Arabic.

**Heading context propagation** assigns each block a heading ancestry path (e.g., `["Study Design", "Inclusion Criteria"]`) based on the heading hierarchy above it. This lets the boundary scorer detect when a block falls under a different heading than its neighbor.

**Continuation flags** mark blocks that start with pronouns, demonstratives, discourse markers, or that continue a numbered/bulleted list. These flags directly feed the orphan risk signal.

**Language detection** runs once per document using the `whatlang` crate, supporting approximately 70 languages. The detected language selects the appropriate pronoun, demonstrative, and discourse marker dictionaries. For CJK and Arabic text, script-based entity detection supplements the capitalization-based approach.

## Post-assembly enrichment

After chunks are assembled, two optional enrichment steps run via LLM calls (OpenAI-compatible API):

**Relation extraction** identifies subject-predicate-object triples within each chunk (e.g., "XR-7742 — inhibits — serotonin transporter"). These relations capture the propositional structure of the text and enable graph-based retrieval.

**Synopsis generation** produces a 1-2 sentence summary for each chunk, suitable for use as node labels in knowledge graphs or as search result snippets.

Both steps use structured JSON output (`response_format: json_schema`) for reliable parsing and are entirely optional — the core chunking pipeline runs without any LLM dependency.

## Cross-chunk entity tracking

The cognitive pipeline tracks which entities appear in which chunks. After assembly, it builds a `shared_entities` map: entity name to list of chunk indices. Only entities appearing in two or more chunks are included, making this a focused index of concept threads that span the document.

This enables graph-based retrieval patterns. Instead of searching for a single chunk that matches a query, a system can find the entry point chunk and follow entity links to retrieve the full context for a concept — all chunks that discuss "Compound XR-7742" or "the compromised host," regardless of their position in the document.

## Ambiguous boundary refinement

Not every boundary decision is clear-cut. When the join score falls in an uncertainty band (within half a standard deviation of the mean), the boundary is classified as ambiguous. These ambiguous boundaries — typically 10-20% of all boundaries — can optionally be sent to a cross-encoder reranker for refinement.

The reranker scores each ambiguous boundary's text pair through a sequence classification model (e.g., `ms-marco-MiniLM-L-6-v2` via ONNX Runtime). The reranker score is blended with the original semantic similarity at a configurable weight (default 70% reranker, 30% original), and the join score is updated. This selective approach avoids the O(n) cost of running a cross-encoder on every boundary while refining the decisions that matter most.

## Graph-shaped output

Cognitive chunks carry enough metadata to function as nodes in a knowledge graph. The graph export format structures the output as:

- **Nodes**: each chunk with its text, heading path, dominant entities, token estimate, and continuity confidence
- **Edges**: adjacency links (sequential flow) and entity links (shared concept threads between non-adjacent chunks)
- **Metadata**: chunk count, edge count, and shared entity statistics

This format is designed for direct import into graph databases like Neo4j or visualization tools, supporting retrieval patterns that follow both document flow and concept threads.

## Evaluation metrics

The pipeline computes four quality metrics automatically on every run:

**Entity orphan rate** — fraction of chunk boundaries that separate entities shared across the boundary. Lower is better. A split between "The system uses XR-7742" and "It inhibits serotonin reuptake" has high entity overlap; splitting here orphans the reference.

**Pronoun boundary rate** — fraction of chunks that start with an unresolved pronoun or demonstrative. Lower is better. "It also supports..." as the first sentence of a chunk signals missing context.

**Heading attachment rate** — fraction of headings that remain attached to their first content block. Higher is better. A heading alone in a chunk can't answer any query.

**Discourse break rate** — fraction of chunk boundaries that split a discourse continuation ("Furthermore," "However," "As a result"). Lower is better. These markers explicitly signal dependence on the previous block.

These metrics provide a quantitative assessment of chunk quality without requiring human evaluation or ground truth labels.

## When to use cognitive chunking

Cognitive chunking is more expensive than semantic chunking. The enrichment stage adds entity detection, discourse analysis, and heading propagation. The scoring stage evaluates eight signals instead of one. The optional reranker and LLM steps add further cost.

Use it when the documents have rich internal structure that semantic chunking doesn't capture:

- **Long-form reports** where entities and causal chains span multiple sections
- **Regulatory and compliance documents** where splitting a requirement from its rationale produces chunks that are individually misleading
- **Knowledge bases** where you need graph-based retrieval, not just vector similarity
- **Documents with dense cross-references** where pronouns, demonstratives, and discourse markers are frequent

For short, topically distinct documents, or for high-throughput pipelines where latency is the priority, semantic chunking remains the better choice.

## Cognition-aware chunking in digital forensics

Digital forensic reports follow a narrative structure that is uniquely hostile to naive chunking. An incident report describes a timeline of events: an initial compromise, lateral movement, data exfiltration, and remediation actions. Each paragraph builds on the previous one through entity chains — "the attacker," "the compromised host," "this IP address," "the exfiltrated dataset" — and causal connectives — "which led to," "as a result of," "subsequently."

Semantic chunking treats these paragraphs as topically similar (they're all about the same incident) and either lumps them into one oversized chunk or splits them at arbitrary points where embedding similarity happens to dip. The result is chunks where "the attacker" has no antecedent, where a remediation action is separated from the vulnerability it addresses, or where a timeline entry loses its temporal context.

Cognition-aware chunking preserves these chains. Entity continuity keeps "the compromised host" connected to the paragraph that identifies it. Discourse continuation keeps "Subsequently, the attacker moved laterally" attached to the paragraph describing the initial foothold. Orphan risk prevents chunks that start with "This artifact" when the artifact was defined two blocks earlier.

For forensic analysts building RAG systems over case files, this means retrieval returns self-contained narrative segments rather than fragments. When an analyst queries "what was the exfiltration method," the retrieved chunk includes the method, the target data, and the causal chain that led to it — not just the sentence that mentions "exfiltration" in isolation.

The graph export format is particularly valuable here. Forensic investigations involve entities (hosts, IP addresses, user accounts, malware samples) that appear across multiple phases of an incident. The shared entity map lets a retrieval system follow "192.168.1.42" from initial compromise through lateral movement to exfiltration, surfacing the full context for that host without relying on keyword search.

## Cognition-aware chunking in pharmaceutical and healthcare

Pharmaceutical and healthcare documents — clinical trial protocols, regulatory submissions, batch manufacturing records, adverse event reports — present a different challenge. These documents are long, structurally complex, and have strict requirements about what information must be kept together.

A clinical trial protocol describes a compound across multiple sections: mechanism of action, study design, inclusion criteria, exclusion criteria, dosing regimen, primary and secondary endpoints, statistical analysis plan. The compound name appears in every section. Patient eligibility criteria reference the disease definition from the background section. The statistical analysis plan references the endpoints defined three sections earlier.

Semantic chunking handles the section boundaries well — each section is a different topic. But within sections, the topic is uniform, and semantic chunking either produces one massive chunk per section or splits at random sentence boundaries. The result is that inclusion criteria get separated from the disease definition they reference, or a dosing rationale gets separated from the pharmacokinetic data that supports it.

Cognition-aware chunking preserves these cross-references. When the inclusion criteria say "MADRS total score >= 26 at both screening and baseline visits," the entity tracker connects this to the chunk where MADRS is defined. When the statistical analysis plan says "the primary endpoint is change from baseline in MADRS total score at Week 8," the relation extractor captures the subject-predicate-object triple linking the analysis to the endpoint definition.

For batch manufacturing records, the stakes are even higher. A deviation report describes what went wrong, what the root cause was, what corrective action was taken, and what the impact assessment concluded. These sections are causally linked — the corrective action only makes sense in the context of the root cause, and the impact assessment references both. Splitting these into separate chunks without preserving the causal chain means a retrieval system might return the corrective action without the deviation that prompted it, or the impact assessment without the corrective action it evaluated.

The evaluation metrics are especially useful in regulated environments. An entity orphan rate above zero means some chunks reference entities defined elsewhere — a quantitative signal that retrieved chunks may be incomplete. A pronoun boundary rate above zero means some chunks start with unresolved references. These metrics provide auditable quality indicators for validation of the chunking pipeline, supporting regulatory requirements around data integrity and traceability.
