mod chunker;
mod cognitive;
mod error;
mod merge;
mod semantic;
mod signal;
mod splitter;

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
    m.add_function(wrap_pyfunction!(cognitive::py_cognitive_chunk, m)?)?;

    Ok(())
}
