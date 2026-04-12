mod adaptive;
mod chunker;
mod cognitive;
mod enriched;
mod error;
mod intent;
mod merge;
mod quality_metrics;
mod semantic;
mod signal;
mod splitter;
mod topo;

use pyo3::prelude::*;

#[pymodule]
fn cognigraph_chunker(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Error
    m.add(
        "CognigraphError",
        m.py().get_type::<error::CognigraphError>(),
    )?;

    // Chunker
    m.add_class::<chunker::PyChunker>()?;

    // Splitter
    m.add_function(wrap_pyfunction!(splitter::py_split_at_delimiters, m)?)?;
    m.add_function(wrap_pyfunction!(splitter::py_split_at_patterns, m)?)?;
    m.add_class::<splitter::PyPatternSplitter>()?;

    // Merge
    m.add_function(wrap_pyfunction!(merge::merge_splits, m)?)?;
    m.add_function(wrap_pyfunction!(merge::find_merge_indices, m)?)?;
    m.add_class::<merge::PyMergeResult>()?;

    // Signal
    m.add_function(wrap_pyfunction!(signal::savgol_filter, m)?)?;
    m.add_function(wrap_pyfunction!(signal::windowed_cross_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(signal::find_local_minima, m)?)?;
    m.add_function(wrap_pyfunction!(signal::filter_split_indices, m)?)?;
    m.add_class::<signal::PyMinimaResult>()?;
    m.add_class::<signal::PyFilteredIndices>()?;

    // Semantic providers
    m.add_class::<semantic::providers::PyOllamaProvider>()?;
    m.add_class::<semantic::providers::PyOpenAiProvider>()?;
    m.add_class::<semantic::providers::PyOnnxProvider>()?;

    // Semantic chunking
    m.add_class::<semantic::PySemanticConfig>()?;
    m.add_class::<semantic::PySemanticResult>()?;
    m.add_function(wrap_pyfunction!(semantic::py_semantic_chunk, m)?)?;

    // Cognitive chunking
    m.add_class::<cognitive::PyCognitiveConfig>()?;
    m.add_class::<cognitive::PyCognitiveResult>()?;
    m.add_class::<cognitive::PyCognitiveChunk>()?;
    m.add_class::<cognitive::PyRelationTriple>()?;
    m.add_function(wrap_pyfunction!(cognitive::py_cognitive_chunk, m)?)?;

    // Intent-driven chunking
    m.add_class::<intent::PyIntentConfig>()?;
    m.add_class::<intent::PyIntentResult>()?;
    m.add_class::<intent::PyIntentChunk>()?;
    m.add_class::<intent::PyPredictedIntent>()?;
    m.add_function(wrap_pyfunction!(intent::py_intent_chunk, m)?)?;

    // Enriched chunking
    m.add_class::<enriched::PyEnrichedConfig>()?;
    m.add_class::<enriched::PyEnrichedResult>()?;
    m.add_class::<enriched::PyEnrichedChunk>()?;
    m.add_class::<enriched::PyTypedEntity>()?;
    m.add_class::<enriched::PyMergeRecord>()?;
    m.add_function(wrap_pyfunction!(enriched::py_enriched_chunk, m)?)?;

    // Topology-aware chunking
    m.add_class::<topo::PyTopoConfig>()?;
    m.add_class::<topo::PyTopoResult>()?;
    m.add_class::<topo::PyTopoChunk>()?;
    m.add_class::<topo::PySectionClassification>()?;
    m.add_function(wrap_pyfunction!(topo::py_topo_chunk, m)?)?;

    // Quality metrics
    m.add_class::<quality_metrics::PyMetricWeights>()?;
    m.add_class::<quality_metrics::PyQualityMetrics>()?;
    m.add_class::<quality_metrics::PyChunkForEval>()?;
    m.add_class::<quality_metrics::PyMetricConfig>()?;
    m.add_function(wrap_pyfunction!(quality_metrics::py_evaluate_chunks, m)?)?;

    // Adaptive chunking
    m.add_class::<adaptive::PyAdaptiveConfig>()?;
    m.add_class::<adaptive::PyAdaptiveResult>()?;
    m.add_class::<adaptive::PyAdaptiveReport>()?;
    m.add_class::<adaptive::PyCandidateScore>()?;
    m.add_class::<adaptive::PyScreeningDecision>()?;
    m.add_function(wrap_pyfunction!(adaptive::py_adaptive_chunk, m)?)?;

    Ok(())
}
