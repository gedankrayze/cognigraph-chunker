//! Data types for topology-aware chunking output.

use serde::{Deserialize, Serialize};

use super::sir::Sir;

/// Classification of how a section should be treated during chunking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SectionClass {
    /// Section is small enough to be a single chunk.
    Atomic,
    /// Section is too large and must be split.
    Splittable,
    /// Section is too small and should be merged with neighbours.
    MergeCandidate,
}

/// Inspector-produced classification for a single section.
#[derive(Debug, Clone, Serialize)]
pub struct SectionClassification {
    /// SIR node ID of the section.
    pub section_id: usize,
    /// How the section should be treated.
    pub class: SectionClass,
    /// Reasoning from the Inspector agent.
    pub reason: String,
}

/// A chunk produced by topology-aware chunking.
#[derive(Debug, Clone, Serialize)]
pub struct TopoChunk {
    /// The chunk text.
    pub text: String,
    /// Byte offset of chunk start in the source document.
    pub offset_start: usize,
    /// Byte offset of chunk end in the source document.
    pub offset_end: usize,
    /// Estimated token count.
    pub token_estimate: usize,
    /// Heading ancestry path for this chunk.
    pub heading_path: Vec<String>,
    /// How this section was classified: "atomic", "splittable", or "merged".
    pub section_classification: String,
    /// Indices of other chunks that share entity cross-references.
    pub cross_references: Vec<usize>,
}

/// Result of the topology-aware chunking pipeline.
#[derive(Debug)]
pub struct TopoResult {
    /// The produced chunks.
    pub chunks: Vec<TopoChunk>,
    /// The SIR built from the document.
    pub sir: Sir,
    /// Section classifications from the Inspector.
    pub classifications: Vec<SectionClassification>,
    /// Total number of blocks parsed.
    pub block_count: usize,
}
