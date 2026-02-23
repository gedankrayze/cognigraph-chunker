mod cli;

use std::io;

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{Shell, generate};

use cli::global_opts::GlobalOpts;

#[derive(Parser)]
#[command(name = "cognigraph-chunker")]
#[command(version, about = "CogniGraph Chunker — fast text chunking toolkit")]
#[command(after_help = "\
EXAMPLES:
  # Fixed-size chunking (4 KB chunks)
  cognigraph-chunker chunk -i document.md -s 4096

  # Split on sentence-ending punctuation
  cognigraph-chunker split -i document.md -d \".?!\"

  # Semantic chunking with Ollama (default)
  cognigraph-chunker semantic -i document.md

  # Semantic chunking with OpenAI, JSON output
  cognigraph-chunker semantic -i doc.md -p openai -f json

  # Merge small chunks into ~512 token groups
  cognigraph-chunker split -i doc.md --merge --chunk-size 512

  # Print chunk statistics
  cognigraph-chunker chunk -i doc.md --stats

  # Generate shell completions
  cognigraph-chunker completions bash > ~/.bash_completions/cognigraph-chunker
")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[command(flatten)]
    global: GlobalOpts,
}

#[derive(Subcommand)]
enum Commands {
    /// Fixed-size chunking with delimiter-aware boundaries
    #[command(after_help = "\
EXAMPLES:
  cognigraph-chunker chunk -i input.txt -s 2048
  cognigraph-chunker chunk -i input.txt -d \"\\n.?\" --prefix
  cat file.txt | cognigraph-chunker chunk -s 1024 -f json
")]
    Chunk(cli::chunk_cmd::ChunkArgs),

    /// Split text at every delimiter or pattern occurrence
    #[command(after_help = "\
EXAMPLES:
  cognigraph-chunker split -i doc.md -d \".?!\"
  cognigraph-chunker split -i doc.md -p \". ,? \" --include-delim next
  cognigraph-chunker split -i doc.md --min-chars 100 -f jsonl
")]
    Split(cli::split_cmd::SplitArgs),

    /// Semantic chunking using embedding similarity and Savitzky-Golay smoothing
    #[command(after_help = "\
EXAMPLES:
  cognigraph-chunker semantic -i doc.md
  cognigraph-chunker semantic -i doc.md -p openai -f json
  cognigraph-chunker semantic -i doc.md --sg-window 15 --threshold 0.3
  cognigraph-chunker semantic -i doc.md --emit-distances 2>distances.tsv
  cognigraph-chunker semantic -i doc.md --no-markdown
")]
    Semantic(cli::semantic_cmd::SemanticArgs),

    /// Generate shell completion scripts
    #[command(after_help = "\
EXAMPLES:
  cognigraph-chunker completions bash > ~/.bash_completions/cognigraph-chunker
  cognigraph-chunker completions zsh > ~/.zfunc/_cognigraph-chunker
  cognigraph-chunker completions fish > ~/.config/fish/completions/cognigraph-chunker.fish
")]
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Chunk(args) => cli::chunk_cmd::run(args, &cli.global),
        Commands::Split(args) => cli::split_cmd::run(args, &cli.global),
        Commands::Semantic(args) => cli::semantic_cmd::run(args, &cli.global).await,
        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            generate(*shell, &mut cmd, "cognigraph-chunker", &mut io::stdout());
            Ok(())
        }
    }
}
