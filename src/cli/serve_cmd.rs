//! `serve` subcommand — starts the REST API server.

use clap::Args;
use tokio::net::TcpListener;

use cognigraph_chunker::api::{self, AppState};

#[derive(Args)]
pub struct ServeArgs {
    /// Host address to bind to
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 3000)]
    pub port: u16,

    /// API key for bearer token authentication (required unless --no-auth is set)
    #[arg(long)]
    pub api_key: Option<String>,

    /// Explicitly allow running without authentication (insecure)
    #[arg(long)]
    pub no_auth: bool,

    /// Allow embedding provider base_urls pointing to private/loopback IPs
    #[arg(long)]
    pub allow_private_urls: bool,

    /// Allowed CORS origins (repeatable; omit for same-origin only)
    #[arg(long = "cors-origin")]
    pub cors_origins: Vec<String>,
}

pub async fn run(args: &ServeArgs) -> anyhow::Result<()> {
    if args.api_key.is_none() && !args.no_auth {
        anyhow::bail!(
            "No --api-key provided. The API would be unauthenticated and open to all requests.\n\
             Either provide --api-key <KEY> or pass --no-auth to explicitly allow insecure mode."
        );
    }

    // Validate CORS origins at startup
    for origin in &args.cors_origins {
        use axum::http::header::HeaderValue;
        origin.parse::<HeaderValue>().map_err(|_| {
            anyhow::anyhow!(
                "Invalid --cors-origin value: '{origin}'. \
                 Each origin must be a valid Origin header (e.g. 'https://example.com')."
            )
        })?;
    }

    let state = AppState {
        api_key: args.api_key.clone(),
        allow_private_urls: args.allow_private_urls,
        cors_origins: args.cors_origins.clone(),
    };

    let app = api::router(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr).await?;

    eprintln!("CogniGraph Chunker API listening on http://{addr}");
    if args.api_key.is_some() {
        eprintln!("  Bearer token auth: enabled");
    } else {
        eprintln!("  WARNING: Running in --no-auth mode. API is unauthenticated.");
    }
    if args.cors_origins.is_empty() {
        eprintln!("  CORS: same-origin only (no cross-origin access)");
    } else {
        eprintln!("  CORS: {} allowed origin(s)", args.cors_origins.len());
    }

    axum::serve(listener, app).await?;

    Ok(())
}
