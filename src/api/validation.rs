//! Request validation helpers shared by all API handlers.

use std::path::PathBuf;

use super::semantic::validate_base_url;

/// Validate every user-supplied outbound URL field in a request.
///
/// All fields the server will connect to (`base_url`, `llm_base_url`,
/// `oauth_token_url`, `oauth_base_url`) must go through the same SSRF
/// validation — an attacker who can steer any one of them can reach
/// internal services or cloud metadata endpoints.
pub fn validate_outbound_urls(
    allow_private: bool,
    urls: &[(&str, Option<&str>)],
) -> anyhow::Result<()> {
    for (field, value) in urls {
        if let Some(value) = value {
            validate_base_url(value, allow_private).map_err(|e| anyhow::anyhow!("{field}: {e}"))?;
        }
    }
    Ok(())
}

/// Validate a client-supplied ONNX model path against the server allowlist.
///
/// Remote clients must not be able to point the server at arbitrary
/// filesystem paths (file-existence probing, parsing DoS). Model loading is
/// disabled unless the server was started with `--onnx-model-dir`; requested
/// paths must resolve inside that directory. Relative paths are resolved
/// against it.
pub fn validate_model_path(
    requested: &str,
    allowed_dir: &Option<PathBuf>,
) -> anyhow::Result<PathBuf> {
    let Some(allowed) = allowed_dir else {
        anyhow::bail!(
            "Invalid model_path: loading ONNX models via the API is disabled. Start the \
             server with --onnx-model-dir <DIR> to allow model paths under that directory."
        );
    };

    let allowed = allowed
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("Configured --onnx-model-dir is not accessible: {e}"))?;

    let requested_path = std::path::Path::new(requested);
    let candidate = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        allowed.join(requested_path)
    };

    // Canonicalize to resolve symlinks and `..` before the prefix check.
    // A single uniform error avoids file-existence probing outside the dir.
    let resolved = candidate
        .canonicalize()
        .map_err(|_| anyhow::anyhow!("Invalid model_path: not found or not permitted"))?;

    anyhow::ensure!(
        resolved.starts_with(&allowed),
        "Invalid model_path: not found or not permitted"
    );

    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dirs() -> (PathBuf, PathBuf) {
        let base =
            std::env::temp_dir().join(format!("cognigraph-model-dir-test-{}", std::process::id()));
        let inside = base.join("model-a");
        std::fs::create_dir_all(&inside).unwrap();
        (base, inside)
    }

    #[test]
    fn test_model_path_disabled_without_allowlist() {
        let err = validate_model_path("/anywhere/model", &None).unwrap_err();
        assert!(err.to_string().contains("disabled"), "{err}");
    }

    #[test]
    fn test_model_path_inside_allowlist_ok() {
        let (base, inside) = test_dirs();
        let allowed = Some(base.clone());

        // Absolute path inside the dir
        assert!(validate_model_path(inside.to_str().unwrap(), &allowed).is_ok());
        // Relative path resolved against the dir
        assert!(validate_model_path("model-a", &allowed).is_ok());
    }

    #[test]
    fn test_model_path_outside_allowlist_rejected() {
        let (base, _inside) = test_dirs();
        let allowed = Some(base.clone());

        // An existing directory outside the allowlist
        let err = validate_model_path("/etc", &allowed).unwrap_err();
        assert!(err.to_string().contains("not permitted"), "{err}");

        // Path traversal out of the allowlist
        let err = validate_model_path("../", &allowed).unwrap_err();
        assert!(err.to_string().contains("not permitted"), "{err}");
    }

    #[test]
    fn test_outbound_urls_names_the_field() {
        let err = validate_outbound_urls(
            false,
            &[
                ("base_url", None),
                ("llm_base_url", Some("http://127.0.0.1:9000")),
            ],
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("llm_base_url"), "{msg}");
        assert!(msg.contains("private"), "{msg}");
    }

    #[test]
    fn test_outbound_urls_all_none_ok() {
        assert!(validate_outbound_urls(false, &[("base_url", None)]).is_ok());
    }
}
