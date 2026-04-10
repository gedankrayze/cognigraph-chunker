//! Topology-aware chunking pipeline.
//!
//! Builds a Structured Intermediate Representation (SIR) from the document,
//! then uses two LLM agents (Inspector + Refiner) to produce topology-preserving chunks.

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use super::blocks::{Block, BlockKind, split_blocks};
use super::enrichment::discourse::detect_discourse_markers;
use super::enrichment::entities::extract_entities;
use super::enrichment::heading_context::compute_heading_paths;
use super::sir::{Sir, SirEdge, SirEdgeType, SirNode, SirNodeType};
use super::topo_types::{
    SectionClass, SectionClassification, TopoChunk, TopoResult,
};
use crate::llm::topo_agents::{inspect_sir, refine_partition};
use crate::llm::CompletionClient;

/// Configuration for topology-aware chunking.
#[derive(Debug, Clone)]
pub struct TopoConfig {
    /// Soft token budget per chunk (default: 512).
    pub soft_budget: usize,
    /// Hard token ceiling per chunk (default: 768).
    pub hard_budget: usize,
    /// Whether to emit the SIR in the result (default: false).
    pub emit_sir: bool,
}

impl Default for TopoConfig {
    fn default() -> Self {
        Self {
            soft_budget: 512,
            hard_budget: 768,
            emit_sir: false,
        }
    }
}

/// Run the topology-aware chunking pipeline.
///
/// Requires markdown input (topo mode relies on heading structure).
pub async fn topo_chunk(
    text: &str,
    llm_client: &CompletionClient,
    _config: &TopoConfig,
) -> Result<TopoResult> {
    let blocks = split_blocks(text);
    if blocks.is_empty() {
        return Ok(TopoResult {
            chunks: vec![],
            sir: Sir {
                nodes: vec![],
                edges: vec![],
                root: 0,
            },
            classifications: vec![],
            block_count: 0,
        });
    }

    let block_count = blocks.len();

    // Step 1-2: Compute heading paths
    let (heading_paths, heading_terms) = compute_heading_paths(&blocks);

    // Step 3: Build SIR tree
    let sir = build_sir(&blocks, &heading_paths, &heading_terms);

    // Step 4: Call Inspector Agent
    let sir_json = serde_json::to_string_pretty(&sir)?;
    // Truncate if > 100k chars to fit LLM context
    let sir_for_llm = if sir_json.len() > 100_000 {
        sir_json[..100_000].to_string()
    } else {
        sir_json.clone()
    };

    let inspector_result = inspect_sir(llm_client, &sir_for_llm).await?;

    // Build classification map
    let mut classifications: Vec<SectionClassification> = Vec::new();
    let mut class_map: HashMap<usize, SectionClass> = HashMap::new();
    for ic in &inspector_result.classifications {
        let class = match ic.class.as_str() {
            "atomic" => SectionClass::Atomic,
            "splittable" => SectionClass::Splittable,
            "merge_candidate" => SectionClass::MergeCandidate,
            _ => SectionClass::Atomic, // fallback
        };
        class_map.insert(ic.section_id, class);
        classifications.push(SectionClassification {
            section_id: ic.section_id,
            class,
            reason: ic.reason.clone(),
        });
    }

    // Step 5: Collect text of splittable sections for the Refiner
    let splittable_texts = collect_splittable_texts(&sir, &class_map, &blocks);
    let inspector_json = serde_json::to_string_pretty(&inspector_result.classifications)?;

    let refiner_result = refine_partition(
        llm_client,
        &inspector_json,
        &sir_for_llm,
        &splittable_texts,
    )
    .await?;

    // Step 6: Assembly — map partition back to text spans
    let chunks = assemble_chunks(
        &refiner_result.partition,
        &blocks,
        &heading_paths,
        &class_map,
        &sir,
    );

    Ok(TopoResult {
        chunks,
        sir,
        classifications,
        block_count,
    })
}

/// Build a SIR tree from parsed blocks and heading paths.
pub fn build_sir(
    blocks: &[Block<'_>],
    _heading_paths: &[Vec<String>],
    heading_terms: &HashSet<String>,
) -> Sir {
    let mut nodes: Vec<SirNode> = Vec::new();
    let mut edges: Vec<SirEdge> = Vec::new();
    let mut next_id: usize = 0;

    // Create root node covering all blocks
    let root_id = next_id;
    next_id += 1;
    nodes.push(SirNode {
        id: root_id,
        node_type: SirNodeType::Section,
        heading: None,
        heading_level: Some(0),
        block_type: None,
        block_range: (0, blocks.len()),
        children: vec![],
        text_preview: preview_text(blocks, 0, blocks.len()),
        token_estimate: estimate_tokens_range(blocks, 0, blocks.len()),
    });

    // Walk blocks and create section nodes when headings are encountered
    let mut section_stack: Vec<(usize, u8)> = vec![(root_id, 0)]; // (node_id, level)
    let mut block_to_node: Vec<usize> = Vec::with_capacity(blocks.len());

    for (i, block) in blocks.iter().enumerate() {
        if block.kind == BlockKind::Heading {
            let (level, heading_text) = parse_heading(block.text);

            // Pop sections at same or deeper level
            while section_stack.len() > 1
                && section_stack.last().is_some_and(|(_, l)| *l >= level as u8)
            {
                section_stack.pop();
            }

            // Create section node
            let section_id = next_id;
            next_id += 1;

            // Determine end range: scan forward to next heading at same or higher level
            let section_end = find_section_end(blocks, i, level);

            nodes.push(SirNode {
                id: section_id,
                node_type: SirNodeType::Section,
                heading: Some(heading_text.to_string()),
                heading_level: Some(level as u8),
                block_type: None,
                block_range: (i, section_end),
                children: vec![],
                text_preview: preview_text(blocks, i, section_end),
                token_estimate: estimate_tokens_range(blocks, i, section_end),
            });

            // Add as child of current parent
            let parent_id = section_stack.last().map(|(id, _)| *id).unwrap_or(root_id);
            nodes[parent_id].children.push(section_id);

            section_stack.push((section_id, level as u8));

            // Map heading block to its section node
            block_to_node.push(section_id);
        } else {
            // Create content block node
            let block_id = next_id;
            next_id += 1;

            nodes.push(SirNode {
                id: block_id,
                node_type: SirNodeType::ContentBlock,
                heading: None,
                heading_level: None,
                block_type: Some(format!("{:?}", block.kind)),
                block_range: (i, i + 1),
                children: vec![],
                text_preview: truncate(block.text, 200),
                token_estimate: block.text.len().div_ceil(4),
            });

            // Add as child of current section
            let parent_id = section_stack.last().map(|(id, _)| *id).unwrap_or(root_id);
            nodes[parent_id].children.push(block_id);

            block_to_node.push(block_id);
        }
    }

    // Add entity co-reference edges between non-adjacent blocks
    let empty_stopwords: &[&str] = &["the", "a", "an", "of", "in", "to", "and", "or", "is", "are"];
    let block_entities: Vec<Vec<_>> = blocks
        .iter()
        .map(|b| extract_entities(b.text, heading_terms, empty_stopwords))
        .collect();

    for i in 0..blocks.len() {
        for j in (i + 2)..blocks.len() {
            // Only non-adjacent blocks (skip i+1 since adjacent continuity is normal)
            let overlap = crate::semantic::enrichment::entities::entity_overlap(
                &block_entities[i],
                &block_entities[j],
            );
            if overlap > 0.3 {
                edges.push(SirEdge {
                    from: block_to_node[i],
                    to: block_to_node[j],
                    edge_type: SirEdgeType::EntityCoref,
                });
            }
        }
    }

    // Add discourse continuation edges
    for (i, block) in blocks.iter().enumerate() {
        if i > 0 && !detect_discourse_markers(block.text).is_empty() {
            edges.push(SirEdge {
                from: block_to_node[i - 1],
                to: block_to_node[i],
                edge_type: SirEdgeType::DiscourseContinuation,
            });
        }
    }

    Sir {
        nodes,
        edges,
        root: root_id,
    }
}

/// Find where a section ends (next heading at same or higher level, or end of document).
fn find_section_end(blocks: &[Block<'_>], heading_idx: usize, heading_level: usize) -> usize {
    for i in (heading_idx + 1)..blocks.len() {
        if blocks[i].kind == BlockKind::Heading {
            let (level, _) = parse_heading(blocks[i].text);
            if level <= heading_level {
                return i;
            }
        }
    }
    blocks.len()
}

/// Parse heading level and text from markdown heading block.
fn parse_heading(text: &str) -> (usize, &str) {
    let trimmed = text.trim();
    let level = trimmed.chars().take_while(|&c| c == '#').count();
    let heading_text = trimmed[level..].trim();
    (level.max(1), heading_text)
}

/// Estimate tokens for a range of blocks.
pub fn estimate_tokens_range(blocks: &[Block<'_>], start: usize, end: usize) -> usize {
    blocks[start..end]
        .iter()
        .map(|b| b.text.len())
        .sum::<usize>()
        .div_ceil(4)
}

/// Build a text preview from a range of blocks.
fn preview_text(blocks: &[Block<'_>], start: usize, end: usize) -> String {
    let combined: String = blocks[start..end]
        .iter()
        .map(|b| b.text)
        .collect::<Vec<_>>()
        .join(" ");
    truncate(&combined, 200)
}

/// Truncate a string to max_len characters.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Collect full text of splittable sections for the Refiner.
fn collect_splittable_texts(
    sir: &Sir,
    class_map: &HashMap<usize, SectionClass>,
    blocks: &[Block<'_>],
) -> String {
    let mut parts: Vec<String> = Vec::new();

    for node in &sir.nodes {
        if node.node_type == SirNodeType::Section {
            if class_map.get(&node.id) == Some(&SectionClass::Splittable) {
                let (start, end) = node.block_range;
                let text: String = blocks[start..end]
                    .iter()
                    .map(|b| b.text)
                    .collect::<Vec<_>>()
                    .join("\n");
                parts.push(format!("--- Section {} (blocks {}..{}) ---\n{}", node.id, start, end, text));
            }
        }
    }

    parts.join("\n\n")
}

/// Assemble chunks from the Refiner's partition.
fn assemble_chunks(
    partition: &[crate::llm::topo_agents::RefinerChunk],
    blocks: &[Block<'_>],
    heading_paths: &[Vec<String>],
    class_map: &HashMap<usize, SectionClass>,
    sir: &Sir,
) -> Vec<TopoChunk> {
    let mut chunks = Vec::new();

    for part in partition {
        let start = part.block_range.first().copied().unwrap_or(0);
        let end = part.block_range.get(1).copied().unwrap_or(blocks.len());

        // Clamp to valid range
        let start = start.min(blocks.len());
        let end = end.min(blocks.len()).max(start);

        if start >= blocks.len() || start == end {
            continue;
        }

        let text: String = blocks[start..end]
            .iter()
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");

        let offset_start = blocks[start].offset;
        let offset_end = blocks[end - 1].offset + blocks[end - 1].text.len();
        let token_estimate = text.len().div_ceil(4);

        let heading_path = if start < heading_paths.len() {
            heading_paths[start].clone()
        } else {
            vec![]
        };

        // Determine classification label from section_ids
        let classification = part
            .section_ids
            .first()
            .and_then(|sid| class_map.get(sid))
            .map(|c| match c {
                SectionClass::Atomic => "atomic",
                SectionClass::Splittable => "splittable",
                SectionClass::MergeCandidate => "merged",
            })
            .unwrap_or("atomic")
            .to_string();

        // Find cross-references from SIR edges
        let cross_references = find_cross_references(sir, start, end, chunks.len());

        chunks.push(TopoChunk {
            text,
            offset_start,
            offset_end,
            token_estimate,
            heading_path,
            section_classification: classification,
            cross_references,
        });
    }

    chunks
}

/// Find other chunk indices that share entity co-reference edges with this block range.
fn find_cross_references(sir: &Sir, start: usize, end: usize, _current_chunk: usize) -> Vec<usize> {
    let mut refs = Vec::new();
    for edge in &sir.edges {
        if edge.edge_type == SirEdgeType::EntityCoref {
            // Check if one end is in our range and the other is outside
            let from_node = sir.nodes.iter().find(|n| n.id == edge.from);
            let to_node = sir.nodes.iter().find(|n| n.id == edge.to);

            if let (Some(from), Some(to)) = (from_node, to_node) {
                let from_in = from.block_range.0 >= start && from.block_range.0 < end;
                let to_in = to.block_range.0 >= start && to.block_range.0 < end;

                if from_in && !to_in {
                    refs.push(to.block_range.0);
                } else if to_in && !from_in {
                    refs.push(from.block_range.0);
                }
            }
        }
    }
    refs.sort();
    refs.dedup();
    refs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_sir_basic() {
        let md = "\
# Introduction

This is the introduction.

## Architecture

The architecture is modular.
It supports multiple providers.

### Scoring

The scoring formula is complex.

## Evaluation

Evaluation details here.
";
        let blocks = split_blocks(md);
        let (heading_paths, heading_terms) = compute_heading_paths(&blocks);
        let sir = build_sir(&blocks, &heading_paths, &heading_terms);

        // Root node should exist
        assert_eq!(sir.root, 0);
        assert_eq!(sir.nodes[0].node_type, SirNodeType::Section);
        assert_eq!(sir.nodes[0].block_range.0, 0);
        assert_eq!(sir.nodes[0].block_range.1, blocks.len());

        // Should have section nodes for each heading
        let sections: Vec<_> = sir
            .nodes
            .iter()
            .filter(|n| n.node_type == SirNodeType::Section && n.heading.is_some())
            .collect();
        assert!(
            sections.len() >= 4,
            "Should have sections for Introduction, Architecture, Scoring, Evaluation, got {}",
            sections.len()
        );

        // Content blocks should exist
        let content_blocks: Vec<_> = sir
            .nodes
            .iter()
            .filter(|n| n.node_type == SirNodeType::ContentBlock)
            .collect();
        assert!(!content_blocks.is_empty(), "Should have content blocks");
    }

    #[test]
    fn test_build_sir_flat_document() {
        let md = "\
Just a paragraph without any headings.
Another paragraph here.
";
        let blocks = split_blocks(md);
        let (heading_paths, heading_terms) = compute_heading_paths(&blocks);
        let sir = build_sir(&blocks, &heading_paths, &heading_terms);

        // Should still have a root node
        assert_eq!(sir.root, 0);
        assert_eq!(sir.nodes[0].node_type, SirNodeType::Section);

        // No section children (no headings), only content blocks
        let sections_with_heading: Vec<_> = sir
            .nodes
            .iter()
            .filter(|n| n.node_type == SirNodeType::Section && n.heading.is_some())
            .collect();
        assert_eq!(
            sections_with_heading.len(),
            0,
            "Flat document should have no heading sections"
        );

        // But should have content block nodes
        let content_blocks: Vec<_> = sir
            .nodes
            .iter()
            .filter(|n| n.node_type == SirNodeType::ContentBlock)
            .collect();
        assert!(
            content_blocks.len() >= 1,
            "Should have at least one content block"
        );
    }

    #[test]
    fn test_estimate_tokens() {
        let blocks = vec![
            Block {
                text: "Hello world, this is a test sentence with some words.",
                offset: 0,
                kind: BlockKind::Sentence,
            },
            Block {
                text: "Another sentence here.",
                offset: 53,
                kind: BlockKind::Sentence,
            },
        ];

        let total = estimate_tokens_range(&blocks, 0, 2);
        // 53 + 22 = 75 chars -> ceil(75/4) = 19 tokens
        assert_eq!(total, 19);

        let first_only = estimate_tokens_range(&blocks, 0, 1);
        // 53 chars -> ceil(53/4) = 14
        assert_eq!(first_only, 14);
    }
}
