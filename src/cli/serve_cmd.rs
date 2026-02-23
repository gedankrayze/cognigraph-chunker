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

    /// API key for bearer token authentication (optional)
    #[arg(long)]
    pub api_key: Option<String>,
}

pub async fn run(args: &ServeArgs) -> anyhow::Result<()> {
    let state = AppState {
        api_key: args.api_key.clone(),
    };

    let app = api::router(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr).await?;

    eprintln!("CogniGraph Chunker API listening on http://{addr}");
    if args.api_key.is_some() {
        eprintln!("  Bearer token auth: enabled");
    }

    axum::serve(listener, app).await?;

    Ok(())
}
