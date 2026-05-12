use clap::{Parser, Subcommand};
use graphrag_core::Database;
use std::path::PathBuf;
use std::process::ExitCode;

const DEFAULT_STORE: &str = "conversations";
const DEFAULT_DIM: usize = 768; // nomic-embed-text-v1.5

#[derive(Parser)]
#[command(name = "graphrag", about = "Local hybrid graph + vector memory")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Save a note to the knowledge graph
    Note {
        /// The text to remember
        text: String,
        /// Optional source identifier
        #[arg(long)]
        source: Option<String>,
        /// Store name (default: conversations)
        #[arg(long)]
        store: Option<String>,
    },
    /// Search the knowledge graph
    Ask,
    /// Show recent notes
    Log {
        /// Maximum number of notes to show (default: 20)
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Store name (default: conversations)
        #[arg(long)]
        store: Option<String>,
    },
    /// Run entity extraction, embedding, and community detection
    Enrich,
}

fn data_dir() -> PathBuf {
    std::env::var("GRAPHRAG_DATA_DIR")
        .map(PathBuf::from)
        .ok()
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("graphrag")
        })
}

fn open_db() -> Result<Database, String> {
    let dir = data_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("create data dir: {e}"))?;
    Database::open(&dir.join("graphrag.db")).map_err(|e| format!("open db: {e}"))
}

fn cmd_note(text: &str, source: Option<&str>, store: &str) -> Result<(), String> {
    let db = open_db()?;
    if db.get_store(store).is_err() {
        db.create_store(store, DEFAULT_DIM)
            .map_err(|e| format!("create store {store}: {e}"))?;
    }
    let chunk_id = db
        .add_chunk(store, text, source, None)
        .map_err(|e| format!("add chunk: {e}"))?;
    println!("ok chunk_id={chunk_id}");
    Ok(())
}

fn cmd_log(limit: usize, store: &str) -> Result<(), String> {
    let db = open_db()?;
    if db.get_store(store).is_err() {
        // Empty store / never noted to: render nothing rather than erroring.
        return Ok(());
    }
    let chunks = db
        .list_recent_chunks(store, limit)
        .map_err(|e| format!("list chunks: {e}"))?;
    for c in chunks {
        let src = c.source.as_deref().unwrap_or("-");
        println!("{}\t[{}]\t{}", c.created_at, src, c.content);
    }
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Note {
            text,
            source,
            store,
        } => cmd_note(
            &text,
            source.as_deref(),
            store.as_deref().unwrap_or(DEFAULT_STORE),
        ),
        Commands::Log { limit, store } => cmd_log(limit, store.as_deref().unwrap_or(DEFAULT_STORE)),
        Commands::Ask | Commands::Enrich => Err("not yet implemented".to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
