use pyo3::prelude::*;

use cognigraph_chunker::core::OwnedChunker;

#[pyclass(name = "Chunker")]
pub struct PyChunker {
    inner: OwnedChunker,
}

#[pymethods]
impl PyChunker {
    #[new]
    #[pyo3(signature = (text, *, size=4096, delimiters=None, pattern=None, prefix=false, consecutive=false, forward_fallback=false))]
    fn new(
        text: &str,
        size: usize,
        delimiters: Option<&[u8]>,
        pattern: Option<&[u8]>,
        prefix: bool,
        consecutive: bool,
        forward_fallback: bool,
    ) -> Self {
        let mut chunker = OwnedChunker::new(text.as_bytes().to_vec()).size(size);

        if let Some(d) = delimiters {
            chunker = chunker.delimiters(d.to_vec());
        }
        if let Some(p) = pattern {
            chunker = chunker.pattern(p.to_vec());
        }
        if prefix {
            chunker = chunker.prefix();
        }
        if consecutive {
            chunker = chunker.consecutive();
        }
        if forward_fallback {
            chunker = chunker.forward_fallback();
        }

        Self { inner: chunker }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<String> {
        self.inner
            .next_chunk()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
    }

    fn collect_chunks(&mut self) -> Vec<String> {
        let mut chunks = Vec::new();
        while let Some(c) = self.inner.next_chunk() {
            chunks.push(String::from_utf8_lossy(&c).into_owned());
        }
        chunks
    }

    fn collect_offsets(&mut self) -> Vec<(usize, usize)> {
        self.inner.collect_offsets()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}
