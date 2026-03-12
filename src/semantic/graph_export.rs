//! Graph-shaped export for cognitive chunks.
//!
//! Converts a `CognitiveResult` into a JSON graph structure with nodes (chunks)
//! and edges (adjacency, shared entities). Suitable for graph databases,
//! visualization tools, and retrieval-augmented generation systems.

use serde::Serialize;

use super::cognitive_types::CognitiveResult;

/// A graph node representing a chunk.
#[derive(Debug, Serialize)]
pub struct ChunkNode {
    pub id: usize,
    pub text: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub heading_path: Vec<String>,
    pub dominant_entities: Vec<String>,
    pub token_estimate: usize,
    pub continuity_confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synopsis: Option<String>,
}

/// An edge between two chunk nodes.
#[derive(Debug, Serialize)]
pub struct ChunkEdge {
    pub source: usize,
    pub target: usize,
    pub edge_type: EdgeType,
    /// Shared entity name (only for EntityLink edges).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity: Option<String>,
}

/// Type of edge between chunks.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    /// Sequential adjacency (chunk N → chunk N+1).
    Adjacency,
    /// Two chunks share a named entity (concept thread).
    EntityLink,
}

/// The full graph export.
#[derive(Debug, Serialize)]
pub struct ChunkGraph {
    pub nodes: Vec<ChunkNode>,
    pub edges: Vec<ChunkEdge>,
    pub metadata: GraphMetadata,
}

/// Metadata about the graph.
#[derive(Debug, Serialize)]
pub struct GraphMetadata {
    pub chunk_count: usize,
    pub edge_count: usize,
    pub shared_entity_count: usize,
    pub block_count: usize,
}

/// Convert a `CognitiveResult` into a graph structure.
pub fn to_chunk_graph(result: &CognitiveResult) -> ChunkGraph {
    let nodes: Vec<ChunkNode> = result
        .chunks
        .iter()
        .map(|c| ChunkNode {
            id: c.chunk_index,
            text: c.text.clone(),
            offset_start: c.offset_start,
            offset_end: c.offset_end,
            heading_path: c.heading_path.clone(),
            dominant_entities: c.dominant_entities.clone(),
            token_estimate: c.token_estimate,
            continuity_confidence: c.continuity_confidence,
            synopsis: c.synopsis.clone(),
        })
        .collect();

    let mut edges = Vec::new();

    // Adjacency edges
    for chunk in &result.chunks {
        if let Some(next) = chunk.next_chunk {
            edges.push(ChunkEdge {
                source: chunk.chunk_index,
                target: next,
                edge_type: EdgeType::Adjacency,
                entity: None,
            });
        }
    }

    // Entity link edges (concept threads)
    for (entity, chunk_indices) in &result.shared_entities {
        for pair in chunk_indices.windows(2) {
            edges.push(ChunkEdge {
                source: pair[0],
                target: pair[1],
                edge_type: EdgeType::EntityLink,
                entity: Some(entity.clone()),
            });
        }
    }

    let metadata = GraphMetadata {
        chunk_count: nodes.len(),
        edge_count: edges.len(),
        shared_entity_count: result.shared_entities.len(),
        block_count: result.block_count,
    };

    ChunkGraph {
        nodes,
        edges,
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::cognitive_types::CognitiveChunk;

    #[test]
    fn test_graph_export_basic() {
        let result = CognitiveResult {
            chunks: vec![
                CognitiveChunk {
                    text: "First chunk.".to_string(),
                    chunk_index: 0,
                    offset_start: 0,
                    offset_end: 12,
                    heading_path: vec!["Intro".to_string()],
                    dominant_entities: vec!["cognigraph".to_string()],
                    all_entities: vec!["cognigraph".to_string()],
                    dominant_relations: vec![],
                    token_estimate: 3,
                    continuity_confidence: 0.9,
                    boundary_reasons_start: vec![],
                    boundary_reasons_end: vec![],
                    synopsis: None,
                    prev_chunk: None,
                    next_chunk: Some(1),
                },
                CognitiveChunk {
                    text: "Second chunk.".to_string(),
                    chunk_index: 1,
                    offset_start: 12,
                    offset_end: 25,
                    heading_path: vec!["Intro".to_string()],
                    dominant_entities: vec!["cognigraph".to_string()],
                    all_entities: vec!["cognigraph".to_string()],
                    dominant_relations: vec![],
                    token_estimate: 3,
                    continuity_confidence: 0.8,
                    boundary_reasons_start: vec![],
                    boundary_reasons_end: vec![],
                    synopsis: None,
                    prev_chunk: Some(0),
                    next_chunk: None,
                },
            ],
            signals: vec![],
            block_count: 4,
            evaluation: crate::semantic::evaluation::EvaluationMetrics::default(),
            shared_entities: [("cognigraph".to_string(), vec![0, 1])]
                .into_iter()
                .collect(),
        };

        let graph = to_chunk_graph(&result);
        assert_eq!(graph.nodes.len(), 2);
        // 1 adjacency + 1 entity link
        assert_eq!(graph.edges.len(), 2);
        assert_eq!(graph.metadata.shared_entity_count, 1);
    }
}
