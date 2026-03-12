//! Embedding providers for semantic chunking.

use std::{env, path::PathBuf};

pub mod cloudflare;
pub mod oauth;
pub mod ollama;
pub mod onnx;
pub mod openai;
pub mod reranker;

use anyhow::Result;

/// Trait for embedding providers.
///
/// Each provider takes a batch of text strings and returns their embedding vectors.
#[allow(async_fn_in_trait)]
pub trait EmbeddingProvider {
    /// Embed a batch of text strings, returning one vector per input text.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>>;

    /// Return the embedding dimension (if known ahead of time).
    fn dimension(&self) -> Option<usize> {
        None
    }
}

/// Supported embedding provider types.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ProviderType {
    /// OpenAI embeddings API
    Openai,
    /// Ollama local embeddings API
    Ollama,
    /// Local ONNX Runtime model
    Onnx,
    /// Cloudflare Workers AI embeddings
    Cloudflare,
    /// OAuth-authenticated OpenAI-compatible endpoint
    Oauth,
}

/// Ensure ONNX Runtime's shared library can be discovered before creating an ONNX session.
///
/// The crate is configured with `load-dynamic`, so missing runtimes do not fail at
/// compile time. This preflight check fails fast with a clear message instead of
/// hanging when `Session::builder().commit_from_file()` attempts to load the runtime.
pub fn ensure_onnx_runtime_available() -> Result<PathBuf> {
    let resolved = if let Ok(path) = env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(path.trim());
        resolve_explicit_ort_path(path).ok_or_else(|| {
            anyhow::anyhow!(
                "ORT_DYLIB_PATH is set, but the ONNX Runtime shared library was not found there."
            )
        })?
    } else {
        search_default_ort_locations().ok_or_else(|| {
            anyhow::anyhow!(
                "ONNX Runtime shared library not found. Install it (for example: `brew install onnxruntime`), \
                or set ORT_DYLIB_PATH to the library path (e.g. `.../libonnxruntime.dylib` or `.../onnxruntime.dll`)."
            )
        })?
    };

    // Make discovery deterministic for onnxruntime crate internals by setting the explicit path
    // before any session builder initializes the library.
    if env::var_os("ORT_DYLIB_PATH").is_none() {
        unsafe {
            env::set_var("ORT_DYLIB_PATH", &resolved);
        }
    }

    Ok(resolved)
}

fn resolve_explicit_ort_path(path: PathBuf) -> Option<PathBuf> {
    let library_names = onnx_runtime_library_names();
    if path.is_file() {
        return Some(path);
    }

    if path.is_dir() {
        for name in library_names {
            let candidate = path.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

fn search_default_ort_locations() -> Option<PathBuf> {
    let library_names = onnx_runtime_library_names();
    for dir in candidate_library_dirs() {
        for name in library_names {
            let path = dir.join(name);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn candidate_library_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    dirs.extend(env_search_dirs("LD_LIBRARY_PATH"));

    #[cfg(target_os = "macos")]
    dirs.extend(env_search_dirs("DYLD_LIBRARY_PATH"));

    dirs.extend(env_search_dirs("PATH"));

    #[cfg(target_os = "macos")]
    dirs.extend(["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib"].map(PathBuf::from));
    #[cfg(target_os = "linux")]
    dirs.extend(
        [
            "/usr/lib",
            "/usr/local/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/opt/conda/lib",
            "/usr/lib64",
        ]
        .map(PathBuf::from),
    );

    #[cfg(target_os = "windows")]
    dirs.push(std::env::current_dir().unwrap_or_default());

    dirs
}

fn env_search_dirs(name: &str) -> Vec<PathBuf> {
    env::var(name)
        .ok()
        .unwrap_or_default()
        .split(path_list_separator())
        .filter_map(|entry| {
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(PathBuf::from(trimmed))
            }
        })
        .collect()
}

const fn path_list_separator() -> &'static str {
    if cfg!(target_os = "windows") {
        ";"
    } else {
        ":"
    }
}

#[allow(clippy::needless_return)]
const fn onnx_runtime_library_names() -> &'static [&'static str] {
    #[cfg(target_os = "macos")]
    {
        return &["libonnxruntime.dylib"];
    }
    #[cfg(target_os = "windows")]
    {
        return &["onnxruntime.dll"];
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        return &[
            "libonnxruntime.so",
            "libonnxruntime.so.1",
            "libonnxruntime.so.1.16",
        ];
    }
}
