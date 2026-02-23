use pyo3::prelude::*;

use cognigraph_chunker::core::{IncludeDelim, PatternSplitter, split_at_delimiters, split_at_patterns};

fn parse_include_delim(s: &str) -> PyResult<IncludeDelim> {
    match s {
        "prev" => Ok(IncludeDelim::Prev),
        "next" => Ok(IncludeDelim::Next),
        "none" => Ok(IncludeDelim::None),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "include_delim must be 'prev', 'next', or 'none'",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (text, delimiters, *, include_delim="prev", min_chars=0))]
pub fn py_split_at_delimiters(
    text: &str,
    delimiters: &[u8],
    include_delim: &str,
    min_chars: usize,
) -> PyResult<Vec<(usize, usize)>> {
    let mode = parse_include_delim(include_delim)?;
    Ok(split_at_delimiters(text.as_bytes(), delimiters, mode, min_chars))
}

#[pyfunction]
#[pyo3(signature = (text, patterns, *, include_delim="prev", min_chars=0))]
pub fn py_split_at_patterns(
    text: &str,
    patterns: Vec<Vec<u8>>,
    include_delim: &str,
    min_chars: usize,
) -> PyResult<Vec<(usize, usize)>> {
    let mode = parse_include_delim(include_delim)?;
    let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
    Ok(split_at_patterns(text.as_bytes(), &pattern_refs, mode, min_chars))
}

#[pyclass(name = "PatternSplitter")]
pub struct PyPatternSplitter {
    inner: PatternSplitter,
}

#[pymethods]
impl PyPatternSplitter {
    #[new]
    fn new(patterns: Vec<Vec<u8>>) -> Self {
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
        Self {
            inner: PatternSplitter::new(&pattern_refs),
        }
    }

    #[pyo3(signature = (text, *, include_delim="prev", min_chars=0))]
    fn split(
        &self,
        text: &str,
        include_delim: &str,
        min_chars: usize,
    ) -> PyResult<Vec<(usize, usize)>> {
        let mode = parse_include_delim(include_delim)?;
        Ok(self.inner.split(text.as_bytes(), mode, min_chars))
    }
}
