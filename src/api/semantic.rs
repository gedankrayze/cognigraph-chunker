//! POST /api/v1/semantic handler.

use std::net::IpAddr;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use crate::embeddings::EmbeddingProvider;
use crate::semantic::{SemanticConfig, semantic_chunk, semantic_chunk_plain};

use super::AppState;
use super::errors::ApiError;
use super::types::{ChunksResponse, MergeParams, chunks_response, maybe_merge_api};

fn default_sim_window() -> usize {
    3
}
fn default_sg_window() -> usize {
    11
}
fn default_poly_order() -> usize {
    3
}
fn default_threshold() -> f64 {
    0.5
}
fn default_min_distance() -> usize {
    2
}

#[derive(Debug, Deserialize)]
pub struct SemanticRequest {
    pub text: String,
    #[serde(flatten)]
    pub provider_opts: crate::embeddings::EmbedProviderOpts,
    #[serde(default = "default_sim_window")]
    pub sim_window: usize,
    #[serde(default = "default_sg_window")]
    pub sg_window: usize,
    #[serde(default = "default_poly_order")]
    pub poly_order: usize,
    #[serde(default = "default_threshold")]
    pub threshold: f64,
    #[serde(default = "default_min_distance")]
    pub min_distance: usize,
    #[serde(default)]
    pub no_markdown: bool,
    #[serde(default, flatten)]
    pub merge_params: MergeParams,
}

/// Check if an IPv4 address is private/loopback/non-routable.
fn is_private_ipv4(v4: std::net::Ipv4Addr) -> bool {
    let octets = v4.octets();
    v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        // 0.0.0.0 connects to localhost on most platforms
        || v4.is_unspecified()
        || v4.is_broadcast()
        || v4.is_multicast()
        // TEST-NET ranges (192.0.2/24, 198.51.100/24, 203.0.113/24)
        || v4.is_documentation()
        // CGNAT 100.64.0.0/10
        || (octets[0] == 100 && (octets[1] & 0b1100_0000) == 64)
        // IETF protocol assignments 192.0.0.0/24
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
        // Benchmarking 198.18.0.0/15
        || (octets[0] == 198 && (octets[1] & 0xfe) == 18)
        // Reserved 240.0.0.0/4
        || octets[0] >= 240
}

/// Check if an IP address is private/loopback/link-local/non-routable.
///
/// Handles IPv4-mapped IPv6 addresses (e.g. `::ffff:127.0.0.1`) by normalizing
/// to IPv4 before checking.
fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_private_ipv4(v4),
        IpAddr::V6(v6) => {
            // Check loopback (::1) and unspecified (::) BEFORE the IPv4
            // conversions: the legacy IPv4-compatible form maps ::1 to
            // 0.0.0.1 and :: to 0.0.0.0, which would misclassify ::1.
            if v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() {
                return true;
            }

            // Check IPv4-mapped (::ffff:x.x.x.x) and IPv4-compatible (::x.x.x.x) forms
            if let Some(v4) = v6.to_ipv4_mapped().or_else(|| v6.to_ipv4()) {
                return is_private_ipv4(v4);
            }

            let segments = v6.segments();
            // Unique local (fc00::/7): first byte is fc or fd
            let is_unique_local = (segments[0] & 0xfe00) == 0xfc00;
            // Link-local (fe80::/10)
            let is_link_local = (segments[0] & 0xffc0) == 0xfe80;
            is_unique_local || is_link_local
        }
    }
}

/// Validate that a base_url does not point to private/loopback addresses.
///
/// Checks both the literal host and DNS-resolved addresses to prevent rebinding attacks.
pub fn validate_base_url(raw: &str, allow_private: bool) -> anyhow::Result<()> {
    if allow_private {
        return Ok(());
    }

    let parsed = url::Url::parse(raw).map_err(|e| anyhow::anyhow!("Invalid base_url: {e}"))?;

    let scheme = parsed.scheme();
    anyhow::ensure!(
        scheme == "http" || scheme == "https",
        "Invalid base_url scheme '{scheme}': must be http or https"
    );

    // Classify the host via the parsed URL so IPv6 literals ("[::1]") are
    // treated as IPs and never fall through to the DNS path.
    let host = match parsed.host() {
        Some(url::Host::Ipv4(v4)) => {
            if is_private_ip(IpAddr::V4(v4)) {
                anyhow::bail!(
                    "Invalid base_url: private/loopback addresses are not allowed (use --allow-private-urls to override)"
                );
            }
            return Ok(());
        }
        Some(url::Host::Ipv6(v6)) => {
            if is_private_ip(IpAddr::V6(v6)) {
                anyhow::bail!(
                    "Invalid base_url: private/loopback addresses are not allowed (use --allow-private-urls to override)"
                );
            }
            return Ok(());
        }
        Some(url::Host::Domain(domain)) => domain.to_string(),
        None => anyhow::bail!("Invalid base_url: missing host"),
    };
    let host = host.as_str();

    // Reject "localhost"
    if host.eq_ignore_ascii_case("localhost") {
        anyhow::bail!(
            "Invalid base_url: private/loopback addresses are not allowed (use --allow-private-urls to override)"
        );
    }

    // For hostnames, resolve DNS and check all resolved IPs.
    // NOTE: This is a TOCTOU check — reqwest resolves DNS independently at request time.
    // For strict enforcement, consider an outbound proxy or firewall policy.
    // This validation catches the common case and raises the bar significantly.
    let port = parsed.port_or_known_default().unwrap_or(443);
    let socket_addrs: Vec<std::net::SocketAddr> =
        std::net::ToSocketAddrs::to_socket_addrs(&(host, port))
            .map(|iter| iter.collect())
            .unwrap_or_default();

    if socket_addrs.is_empty() {
        anyhow::bail!("Invalid base_url: could not resolve hostname '{host}'");
    }

    for addr in &socket_addrs {
        if is_private_ip(addr.ip()) {
            anyhow::bail!(
                "Invalid base_url: hostname '{host}' resolves to private address {} (use --allow-private-urls to override)",
                addr.ip()
            );
        }
    }

    Ok(())
}

pub async fn semantic_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SemanticRequest>,
) -> Result<Json<ChunksResponse>, ApiError> {
    // SSRF validation for every user-supplied outbound URL
    super::validation::validate_outbound_urls(
        state.allow_private_urls,
        &[
            ("base_url", req.provider_opts.base_url.as_deref()),
            (
                "oauth_token_url",
                req.provider_opts.oauth_token_url.as_deref(),
            ),
            (
                "oauth_base_url",
                req.provider_opts.oauth_base_url.as_deref(),
            ),
        ],
    )?;

    let config = SemanticConfig {
        sim_window: req.sim_window,
        sg_window: req.sg_window,
        poly_order: req.poly_order,
        threshold: req.threshold,
        min_distance: req.min_distance,
        ..SemanticConfig::default()
    };

    let provider = super::provider::build_api_provider(&req.provider_opts, &state).await?;
    let result = run_semantic(&req.text, &provider, &config, req.no_markdown).await?;

    let chunks = maybe_merge_api(result, &req.merge_params);
    Ok(Json(chunks_response(chunks)))
}

async fn run_semantic<P: EmbeddingProvider>(
    text: &str,
    provider: &P,
    config: &SemanticConfig,
    no_markdown: bool,
) -> anyhow::Result<Vec<(String, usize)>> {
    let result = if no_markdown {
        semantic_chunk_plain(text, provider, config).await?
    } else {
        semantic_chunk(text, provider, config).await?
    };
    Ok(result.chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_private_ip_unspecified_and_reserved() {
        // 0.0.0.0 / :: connect to localhost on most platforms; the others are
        // non-routable ranges (CGNAT, IETF protocol assignments, benchmarking,
        // broadcast) that must not be reachable through the SSRF guard.
        for addr in [
            "0.0.0.0",
            "::",
            "100.64.0.1",
            "192.0.0.1",
            "198.18.0.1",
            "255.255.255.255",
        ] {
            assert!(
                is_private_ip(addr.parse().unwrap()),
                "{addr} must be treated as non-routable"
            );
        }
    }

    #[test]
    fn test_is_private_ip_known_private() {
        for addr in [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.10.10",
            "::1",
            "fe80::1",
            "fc00::1",
            "::ffff:127.0.0.1",
        ] {
            assert!(is_private_ip(addr.parse().unwrap()), "{addr} is private");
        }
    }

    #[test]
    fn test_is_private_ip_public() {
        for addr in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
            assert!(!is_private_ip(addr.parse().unwrap()), "{addr} is public");
        }
    }

    #[test]
    fn test_validate_base_url_rejects_unspecified() {
        assert!(
            validate_base_url("http://0.0.0.0:11434", false).is_err(),
            "0.0.0.0 must be rejected"
        );
        assert!(
            validate_base_url("http://[::]:11434", false).is_err(),
            ":: must be rejected"
        );
    }

    #[test]
    fn test_validate_base_url_allows_override() {
        assert!(validate_base_url("http://0.0.0.0:11434", true).is_ok());
        assert!(validate_base_url("http://127.0.0.1:11434", true).is_ok());
    }
}
