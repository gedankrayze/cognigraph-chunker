//! Structured Intermediate Representation (SIR) for topology-aware chunking.
//!
//! The SIR captures document structure as a tree of section and content-block nodes,
//! with cross-cutting edges for entity co-reference and discourse continuation.

use serde::{Deserialize, Serialize};

/// The type of a node in the SIR tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SirNodeType {
    /// A section (heading-delimited region).
    Section,
    /// A leaf content block (sentence, table, code, list, etc.).
    ContentBlock,
}

/// The type of a cross-cutting edge in the SIR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SirEdgeType {
    /// Two non-adjacent blocks share entity mentions.
    EntityCoref,
    /// A block starts with a discourse continuation marker.
    DiscourseContinuation,
}

/// A node in the SIR tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SirNode {
    /// Unique node identifier.
    pub id: usize,
    /// Whether this is a section or content block.
    pub node_type: SirNodeType,
    /// Heading text (sections only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heading: Option<String>,
    /// Heading level 1..=6 (sections only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heading_level: Option<u8>,
    /// Serialized block kind (content blocks only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_type: Option<String>,
    /// Range of block indices covered by this node (inclusive start, exclusive end).
    pub block_range: (usize, usize),
    /// Child node IDs.
    pub children: Vec<usize>,
    /// Preview of the text content (first 200 chars).
    pub text_preview: String,
    /// Estimated token count (~4 chars/token).
    pub token_estimate: usize,
}

/// A cross-cutting edge between two SIR nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SirEdge {
    /// Source node ID.
    pub from: usize,
    /// Target node ID.
    pub to: usize,
    /// Edge type.
    pub edge_type: SirEdgeType,
}

/// The complete Structured Intermediate Representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sir {
    /// All nodes in the SIR.
    pub nodes: Vec<SirNode>,
    /// Cross-cutting edges.
    pub edges: Vec<SirEdge>,
    /// Root node ID.
    pub root: usize,
}
