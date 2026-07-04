//! Credential and configuration resolution.
//!
//! Single source of truth for `.env.*` file parsing and the
//! flag > environment variable > file precedence used by every provider.
//! Previously each provider hand-rolled its own parser with subtly
//! different behavior (none stripped quotes, only some skipped comments).

use anyhow::Result;

/// Look up `key` in a dotenv-style file.
///
/// Supports `KEY=VALUE` lines, `# comments`, an optional `export ` prefix,
/// and single/double quotes around the value. Returns `None` for missing
/// files, missing keys, or empty values.
pub fn env_file_lookup(path: &str, key: &str) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        if k.trim() != key {
            continue;
        }
        let v = strip_quotes(v.trim());
        if !v.is_empty() {
            return Some(v.to_string());
        }
    }
    None
}

/// Strip one pair of matching surrounding quotes, if present.
fn strip_quotes(v: &str) -> &str {
    let bytes = v.as_bytes();
    if bytes.len() >= 2 {
        let (first, last) = (bytes[0], bytes[bytes.len() - 1]);
        if first == last && (first == b'"' || first == b'\'') {
            return &v[1..v.len() - 1];
        }
    }
    v
}

/// Resolve a setting with the standard precedence:
/// explicit value > environment variable > dotenv file (same key name).
pub fn resolve_setting(
    explicit: &Option<String>,
    env_var: &str,
    dotenv_file: &str,
) -> Option<String> {
    if let Some(val) = explicit
        && !val.is_empty()
    {
        return Some(val.clone());
    }
    if let Ok(val) = std::env::var(env_var)
        && !val.is_empty()
    {
        return Some(val);
    }
    env_file_lookup(dotenv_file, env_var)
}

/// Resolve the OpenAI API key from flag, `OPENAI_API_KEY` env var,
/// or `.env.openai` file.
pub fn resolve_openai_key(flag: &Option<String>) -> Result<String> {
    resolve_setting(flag, "OPENAI_API_KEY", ".env.openai").ok_or_else(|| {
        anyhow::anyhow!(
            "OpenAI API key not found.\n\
             Provide it via one of:\n  \
             --api-key <KEY> (or the api_key request field)\n  \
             OPENAI_API_KEY environment variable\n  \
             .env.openai file (OPENAI_API_KEY=sk-...)"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_env_file(name: &str, content: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "cognigraph-config-test-{}-{name}",
            std::process::id()
        ));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_env_file_lookup_basic_and_quotes() {
        let path = write_env_file(
            "basic",
            "# comment\n\
             PLAIN=value1\n\
             DOUBLE=\"quoted value\"\n\
             SINGLE='single quoted'\n\
             export EXPORTED=exp-value\n\
             EMPTY=\n\
             SPACED =  padded  \n",
        );
        let p = path.to_str().unwrap();

        assert_eq!(env_file_lookup(p, "PLAIN").as_deref(), Some("value1"));
        // Quotes must be stripped — a quoted API key otherwise produces a confusing 401
        assert_eq!(
            env_file_lookup(p, "DOUBLE").as_deref(),
            Some("quoted value")
        );
        assert_eq!(
            env_file_lookup(p, "SINGLE").as_deref(),
            Some("single quoted")
        );
        assert_eq!(env_file_lookup(p, "EXPORTED").as_deref(), Some("exp-value"));
        assert_eq!(env_file_lookup(p, "EMPTY"), None);
        assert_eq!(env_file_lookup(p, "SPACED").as_deref(), Some("padded"));
        assert_eq!(env_file_lookup(p, "MISSING"), None);
    }

    #[test]
    fn test_env_file_lookup_exact_key_match() {
        let path = write_env_file("exact", "MY_OPENAI_API_KEY=wrong\nOPENAI_API_KEY=right\n");
        let p = path.to_str().unwrap();
        assert_eq!(
            env_file_lookup(p, "OPENAI_API_KEY").as_deref(),
            Some("right")
        );
    }

    #[test]
    fn test_env_file_lookup_missing_file() {
        assert_eq!(env_file_lookup("/nonexistent/.env.test", "KEY"), None);
    }

    #[test]
    fn test_resolve_setting_explicit_wins() {
        let path = write_env_file("prec", "COGNIGRAPH_TEST_UNSET_VAR=from-file\n");
        let explicit = Some("from-flag".to_string());
        assert_eq!(
            resolve_setting(
                &explicit,
                "COGNIGRAPH_TEST_UNSET_VAR",
                path.to_str().unwrap()
            )
            .as_deref(),
            Some("from-flag")
        );
        // Empty explicit falls through to the file
        assert_eq!(
            resolve_setting(
                &Some(String::new()),
                "COGNIGRAPH_TEST_UNSET_VAR",
                path.to_str().unwrap()
            )
            .as_deref(),
            Some("from-file")
        );
    }
}
