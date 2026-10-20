use clap::{Parser, Subcommand};
use graphrag_core::{
    Database, Embedder, EmbedderConfig, EmbedderModel, RemoteEmbedderConfig,
    default_embedder_cache_dir,
};
mod llm_strategy;
mod memory;

use std::io::{BufRead, Write};

use std::path::PathBuf;
use std::process::ExitCode;

const DEFAULT_STORE: &str = memory::MEMORY_STORE;

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
        /// Group(s) the note belongs to (default: conversations)
        #[arg(long = "group")]
        groups: Vec<String>,
        /// Stable key (file path, page id); new text under a key replaces the old
        #[arg(long)]
        doc_key: Option<String>,
        /// Deprecated alias for --group
        #[arg(long, hide = true)]
        store: Option<String>,
    },
    /// Search the knowledge graph
    Ask {
        /// The query text
        query: String,
        /// Maximum number of hits to return (default: 5)
        #[arg(long, default_value_t = 5)]
        top: usize,
        /// Restrict to these groups (and their subgroups)
        #[arg(long = "in")]
        groups: Vec<String>,
        /// Emit JSON hits (chunk/document ids, per-lane scores) for tooling and evals
        #[arg(long)]
        json: bool,
        /// Lanes to run: any of vector,lexical,graph (default: all)
        #[arg(long)]
        lanes: Option<String>,
        /// Deprecated alias for --in
        #[arg(long, hide = true)]
        store: Option<String>,
    },
    /// Show recent notes
    Log {
        /// Maximum number of notes to show (default: 20)
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Deprecated; the log covers the whole memory
        #[arg(long, hide = true)]
        store: Option<String>,
    },
    /// Embed chunks whose embedding is missing or stale
    BackfillEmbeddings {
        /// Deprecated; ignored
        #[arg(long, hide = true)]
        store: Option<String>,
    },
    /// Run the ingestion worker: embed, extract, generalize, cluster, summarize
    Work {
        /// Restrict to a group (reserved; all groups today)
        #[arg(long = "in")]
        groups: Vec<String>,
        /// Process one batch, then exit
        #[arg(long)]
        once: bool,
        /// Stop after this many work items
        #[arg(long)]
        max_items: Option<usize>,
        /// Retry steps that failed before
        #[arg(long)]
        retry_failed: bool,
        /// Allow superseding more than 10% of extractions (model/prompt change)
        #[arg(long)]
        accept_reextract: bool,
        /// Pause background workers (creates the pause flag) and exit
        #[arg(long)]
        pause: bool,
        /// Resume background workers (removes the pause flag) and exit
        #[arg(long)]
        resume: bool,
    },
    /// One-time migrations
    Migrate {
        #[command(subcommand)]
        what: MigrateCmd,
    },
    /// Deprecated alias for `work`
    Enrich {
        #[arg(long, hide = true)]
        store: Option<String>,
        #[arg(long, hide = true)]
        clear: bool,
        #[arg(long, hide = true)]
        extract: bool,
        #[arg(long, hide = true)]
        max_iterations: Option<usize>,
        #[arg(long, hide = true)]
        tolerance: Option<f64>,
        #[arg(long, hide = true)]
        levels: Option<i32>,
        #[arg(long, hide = true)]
        min_size: Option<usize>,
    },
    /// Export a store as JSON Lines to stdout
    Export {
        /// Store name to export
        store: String,
    },
    /// Split a file into chunks: spans for tooling, or text for review
    Chunk {
        /// File to chunk
        path: PathBuf,
        /// Target chunk size in characters
        #[arg(long, default_value_t = 2000)]
        size: usize,
        /// Overlap in characters (0 disables)
        #[arg(long, default_value_t = 200)]
        overlap: usize,
        /// Force markdown-aware splitting on/off (default: infer from extension)
        #[arg(long)]
        markdown: Option<bool>,
        /// Emit only the n-th chunk (0-based)
        #[arg(long)]
        nth: Option<usize>,
        /// Output form: spans | text | review | json
        #[arg(long, default_value = "spans")]
        format: String,
        /// Chunk as source code on AST boundaries, inferring the parser from
        /// the file extension.
        #[arg(long)]
        code: bool,
        /// Force a specific parser, implying --code
        /// (rust|python|javascript|typescript|go|c|cpp).
        #[arg(long)]
        lang: Option<String>,
        /// Split on line boundaries, never mid-line. Implied for code whose
        /// extension has no parser.
        #[arg(long)]
        lines: bool,
        /// Widen (or narrow, if negative) each chunk by N lines of context.
        #[arg(long, allow_hyphen_values = true, default_value_t = 0)]
        expand: isize,
    },
    /// Compute embeddings for entities that lack them
    Embed {
        /// Store name
        #[arg(long)]
        store: Option<String>,
        /// Maximum entities to embed in this pass
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Report ingestion status per group and step
    Status {
        /// Deprecated; status covers the whole memory
        #[arg(long, hide = true)]
        store: Option<String>,
        /// Emit JSON instead of a human summary
        #[arg(long)]
        json: bool,
    },
    /// Emit pending map-reduce work units as JSON Lines on stdout
    Plan {
        /// Stage to plan: extract | summarize
        stage: String,
        /// Store name
        #[arg(long)]
        store: Option<String>,
        /// Model the units are intended for (selects prompting strategy)
        #[arg(long)]
        model: String,
        /// Maximum number of units to emit
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Ingest executor results as JSON Lines from stdin
    Apply {
        /// Stage to apply: extract | summarize
        stage: String,
        /// Store name
        #[arg(long)]
        store: Option<String>,
    },
    /// Resolve an extraction nonce to its exact entities and relations
    Provenance {
        /// Opaque nonce emitted by extraction planning
        nonce: String,
        /// Store name
        #[arg(long)]
        store: Option<String>,
    },
}

#[derive(Subcommand)]
enum MigrateCmd {
    /// Merge the old stores into one graph with groups and sections
    V2 {
        /// Show what would happen (read-only)
        #[arg(long)]
        plan: bool,
        /// Do it (in one transaction; refuses on unresolved sources)
        #[arg(long)]
        apply: bool,
        /// Root of the mica repo (mica-docs sources)
        #[arg(long)]
        repo_root: Option<PathBuf>,
        /// Directory holding project repos (operation sources)
        #[arg(long)]
        dev_root: Option<PathBuf>,
        /// The GraphRAG paper to import into `papers`
        #[arg(long)]
        paper: Option<PathBuf>,
        /// Print the plan or report as JSON
        #[arg(long)]
        json: bool,
    },
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

struct StoragePaths {
    data_dir: PathBuf,
}

impl StoragePaths {
    fn from_env() -> Self {
        Self {
            data_dir: data_dir(),
        }
    }

    fn db_path(&self) -> PathBuf {
        self.data_dir.join("graphrag.db")
    }
}

fn open_db() -> Result<Database, String> {
    open_db_at(&StoragePaths::from_env())
}

fn open_db_at(paths: &StoragePaths) -> Result<Database, String> {
    std::fs::create_dir_all(&paths.data_dir).map_err(|e| format!("create data dir: {e}"))?;
    Database::open(&paths.db_path()).map_err(|e| format!("open db: {e}"))
}

fn embedder_model() -> EmbedderModel {
    std::env::var("GRAPHRAG_EMBED_MODEL")
        .ok()
        .and_then(|model| match model.to_lowercase().as_str() {
            "minilm" | "mini" => Some(EmbedderModel::MiniLM),
            "nomic" | "nomic-embed-text" => Some(EmbedderModel::NomicEmbedText),
            "openai" | "openai-ada002" | "ada002" => Some(EmbedderModel::OpenAIAda002),
            "openai3" | "openai-3-small" | "openai3-small" => Some(EmbedderModel::OpenAI3Small),
            _ => None,
        })
        .unwrap_or(EmbedderModel::NomicEmbedText)
}

fn open_embedder() -> Result<Embedder, String> {
    let model = embedder_model();
    let remote_config = if model.is_remote() {
        let api_key = std::env::var("GRAPHRAG_OPENAI_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .map_err(
                |_| "GRAPHRAG_OPENAI_API_KEY or OPENAI_API_KEY required for remote embedder",
            )?;

        let mut config = RemoteEmbedderConfig::new(api_key);

        if let Ok(base_url) = std::env::var("GRAPHRAG_OPENAI_BASE_URL") {
            config = config.with_base_url(&base_url);
        }

        Some(config)
    } else {
        None
    };

    Embedder::new(EmbedderConfig {
        model,
        show_download_progress: true,
        cache_dir: Some(default_embedder_cache_dir()),
        remote: remote_config,
    })
    .map_err(|e| format!("embedder: {e}"))
}

fn cmd_log(limit: usize) -> Result<(), String> {
    let store = memory::MEMORY_STORE;
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

fn cmd_export(store: &str) -> Result<(), String> {
    let db = open_db()?;
    let mut stdout = std::io::stdout().lock();
    graphrag_core::export::export_store(&db, store, &mut stdout)
        .map_err(|e| format!("export {store}: {e}"))?;
    Ok(())
}

/// Emit pending work units as JSON Lines. `plan` is a pure read: it never
/// calls a model, so it is safe to run repeatedly and cheap to diff.
fn cmd_plan(stage: &str, store: &str, model: &str, limit: Option<usize>) -> Result<(), String> {
    let db = open_db()?;
    let units = match stage {
        "extract" => {
            let strategy = llm_strategy::LlmStrategy::for_model(model);
            graphrag_core::mr::plan_extract(&db, store, model, &strategy, limit)
        }
        // Summarization needs no strategy: the reply IS the summary.
        "summarize" => graphrag_core::mr::plan_summarize(&db, store, model, limit),
        other => {
            return Err(format!(
                "unknown plan stage {other:?} (supported: extract, summarize)"
            ));
        }
    }
    .map_err(|e| format!("plan {stage}: {e}"))?;
    let mut out = std::io::stdout().lock();
    for u in &units {
        let line = serde_json::to_string(u).map_err(|e| format!("serialize unit: {e}"))?;
        writeln!(out, "{line}").map_err(|e| format!("write unit: {e}"))?;
    }
    eprintln!("planned {} {stage} unit(s)", units.len());
    Ok(())
}

/// Ingest executor results from stdin. Results may arrive partially and out of
/// order; each is independent, so a crashed run simply leaves the rest pending.
fn cmd_apply(stage: &str, store: &str) -> Result<(), String> {
    let db = open_db()?;
    let mut raw: Vec<String> = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.map_err(|e| format!("read stdin: {e}"))?;
        if !line.trim().is_empty() {
            raw.push(line);
        }
    }
    match stage {
        "extract" => {
            let results: Vec<graphrag_core::mr::ExtractResult> = raw
                .iter()
                .enumerate()
                .map(|(n, l)| {
                    serde_json::from_str(l).map_err(|e| format!("parse result line {}: {e}", n + 1))
                })
                .collect::<Result<_, _>>()?;
            // Results share a model in practice; resolve the strategy per batch.
            let model = results.first().map(|r| r.model.clone()).unwrap_or_default();
            let strategy = llm_strategy::LlmStrategy::for_model(&model);
            let persisted = graphrag_core::mr::apply_extract(&db, store, &strategy, &results)
                .map_err(|e| format!("apply {stage}: {e}"))?;
            eprintln!(
                "applied {} result(s), persisted {persisted} triple(s)",
                results.len()
            );
        }
        "summarize" => {
            let results: Vec<graphrag_core::mr::SummaryResult> = raw
                .iter()
                .enumerate()
                .map(|(n, l)| {
                    serde_json::from_str(l).map_err(|e| format!("parse result line {}: {e}", n + 1))
                })
                .collect::<Result<_, _>>()?;
            let n = graphrag_core::mr::apply_summarize(&db, store, &results)
                .map_err(|e| format!("apply {stage}: {e}"))?;
            eprintln!("applied {n} summary result(s)");
        }
        other => {
            return Err(format!(
                "unknown apply stage {other:?} (supported: extract, summarize)"
            ));
        }
    }
    Ok(())
}

/// Print the pipeline phase table: which phases exist, what is ready, and
/// what changed underneath a previous run.
/// Embed entities that have no vector yet. Local compute, so the executor is
/// this process; the plan/apply split still bounds and checkpoints the work.
/// Chunk a file. Two audiences, selected by --format:
///   spans  - TSV of index/start/end/lines, for sed and other byte tooling
///   text   - the chunk bodies, NUL-free and newline separated by index
///   review - a human-readable table with previews
///   json   - machine-readable spans including the chunk text
/// Options for [`cmd_chunk`], grouped so the signature stays readable.
struct ChunkOpts<'a> {
    size: usize,
    overlap: usize,
    markdown: Option<bool>,
    nth: Option<usize>,
    format: &'a str,
    code: bool,
    lang: Option<&'a str>,
    lines: bool,
    expand: isize,
}

fn cmd_chunk(path: &PathBuf, o: ChunkOpts<'_>) -> Result<(), String> {
    let ChunkOpts {
        size,
        overlap,
        markdown,
        nth,
        format,
        code,
        lang,
        lines,
        expand,
    } = o;
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    // Infer the style from the extension unless overridden: markdown files get
    // header/code-block awareness, everything else paragraph/sentence splitting.
    let md = markdown.unwrap_or_else(|| {
        matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("md") | Some("markdown")
        )
    });
    // --lines forces line splitting; --lang implies --code; --code alone infers
    // the parser from the extension and degrades to lines when there is none.
    let all = if lines {
        graphrag_core::line_spans(&text, size)
    } else if code || lang.is_some() {
        match lang {
            None => {
                let got = graphrag_core::code_spans_auto(&text, &path.to_string_lossy(), size)?;
                if got.strategy == graphrag_core::ChunkStrategy::Lines {
                    eprintln!(
                        "note: no parser for {}; split on line boundaries",
                        path.display()
                    );
                }
                got.spans
            }
            Some(name) => {
                let language = graphrag_core::CodeLanguage::from_extension(name)
                    .or(match name {
                        "rust" => Some(graphrag_core::CodeLanguage::Rust),
                        "python" => Some(graphrag_core::CodeLanguage::Python),
                        "javascript" => Some(graphrag_core::CodeLanguage::JavaScript),
                        "typescript" => Some(graphrag_core::CodeLanguage::TypeScript),
                        "go" => Some(graphrag_core::CodeLanguage::Go),
                        "c" => Some(graphrag_core::CodeLanguage::C),
                        "cpp" => Some(graphrag_core::CodeLanguage::Cpp),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        format!(
                            "unknown language {name:?}; supported extensions: {}",
                            graphrag_core::supported_extensions().join(", ")
                        )
                    })?;
                graphrag_core::code_spans(&text, language, size)?
            }
        }
    } else {
        let mut cfg = graphrag_core::ChunkerConfig::new(size).with_markdown(md);
        cfg = if overlap == 0 {
            cfg.without_overlap()
        } else {
            cfg.with_overlap(overlap)
        };
        graphrag_core::chunk_spans(&text, &cfg)
    };

    let selected = match nth {
        Some(n) => match all.into_iter().nth(n) {
            Some(s) => vec![s],
            None => return Err(format!("chunk {n} does not exist in {}", path.display())),
        },
        None => all,
    };

    // Let the reader revise the ingest-time guess about how much context it needs.
    let spans: Vec<_> = if expand == 0 {
        selected
    } else {
        selected
            .iter()
            .map(|s| graphrag_core::resize_span(&text, s, expand))
            .collect()
    };

    let mut out = std::io::stdout().lock();
    match format {
        "spans" => {
            writeln!(
                out,
                "index\tstart\tend\tline_start\tline_end\tbytes\tfingerprint\tsed_range"
            )
            .map_err(|e| e.to_string())?;
            for s in &spans {
                writeln!(
                    out,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    s.index,
                    s.start,
                    s.end,
                    s.line_start,
                    s.line_end,
                    s.len(),
                    s.fingerprint,
                    s.sed_range()
                )
                .map_err(|e| e.to_string())?;
            }
        }
        "text" => {
            for s in &spans {
                writeln!(out, "{}", s.text).map_err(|e| e.to_string())?;
            }
        }
        "review" => {
            write!(out, "{}", graphrag_core::review_spans(&spans, 100))
                .map_err(|e| e.to_string())?;
            eprintln!(
                "{} chunk(s), {} style",
                spans.len(),
                if md { "markdown" } else { "plain" }
            );
        }
        "json" => {
            for s in &spans {
                let v = serde_json::json!({
                    "record": "chunk_span", "index": s.index,
                    "start": s.start, "end": s.end,
                    "line_start": s.line_start, "line_end": s.line_end,
                    "bytes": s.len(), "fingerprint": s.fingerprint, "text": s.text,
                });
                writeln!(out, "{v}").map_err(|e| e.to_string())?;
            }
        }
        other => {
            return Err(format!(
                "unknown --format {other:?} (spans|text|review|json)"
            ));
        }
    }
    Ok(())
}

fn cmd_embed(store: &str, limit: Option<usize>) -> Result<(), String> {
    let db = open_db()?;
    let units =
        graphrag_core::mr::plan_embed(&db, store, limit).map_err(|e| format!("plan embed: {e}"))?;
    if units.is_empty() {
        eprintln!("all entities embedded");
        return Ok(());
    }
    let embedder = open_embedder()?;
    let total = units.len();
    let mut results = Vec::with_capacity(total);
    for (i, u) in units.iter().enumerate() {
        let vector = embedder.embed(&u.text).map_err(|e| e.to_string())?;
        results.push(graphrag_core::mr::EmbedResult {
            entity_id: u.entity_id,
            vector,
        });
        if (i + 1) % 200 == 0 || i + 1 == total {
            eprintln!("embedded {}/{total}", i + 1);
        }
    }
    let n = graphrag_core::mr::apply_embed(&db, store, &results)
        .map_err(|e| format!("apply embed: {e}"))?;
    eprintln!("stored {n} entity embedding(s)");
    Ok(())
}

fn cmd_provenance(store: &str, nonce: &str) -> Result<(), String> {
    let db = open_db()?;
    let provenance = db
        .get_extraction_provenance(store, nonce)
        .map_err(|e| format!("provenance: {e}"))?
        .ok_or_else(|| format!("no extraction provenance {nonce:?} in store {store:?}"))?;
    let entities = provenance
        .entities
        .iter()
        .map(|entity| {
            serde_json::json!({
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "properties": entity.properties,
            })
        })
        .collect::<Vec<_>>();
    let relations = provenance
        .relations
        .iter()
        .map(|relation| {
            serde_json::json!({
                "id": relation.id,
                "head_id": relation.head_id,
                "tail_id": relation.tail_id,
                "relation": relation.relation,
                "canonical_relation": relation.canonical_relation,
                "properties": relation.properties,
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "provenance_nonce": provenance.nonce,
            "chunk_id": provenance.chunk_id,
            "model": provenance.model,
            "triple_count": provenance.triple_count,
            "response_sha256": provenance.response_sha256,
            "extracted_at": provenance.extracted_at,
            "entities": entities,
            "relations": relations,
        }))
        .map_err(|e| format!("serialize provenance: {e}"))?
    );
    Ok(())
}

fn cmd_migrate_v2(
    plan_only: bool,
    apply: bool,
    repo_root: Option<PathBuf>,
    dev_root: Option<PathBuf>,
    paper: Option<PathBuf>,
    json: bool,
) -> Result<(), String> {
    use graphrag_core::migrate_v2::{MigrationSources, apply_v2_migration, plan_v2};
    if plan_only == apply {
        return Err("pass exactly one of --plan or --apply".into());
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let dev_root = dev_root.unwrap_or_else(|| home.join("development"));
    let (embedder_id, dim) = embedder_identity();
    let sources = MigrationSources {
        mica_root: repo_root.unwrap_or_else(|| dev_root.join("mica")),
        paper_path: paper
            .unwrap_or_else(|| dev_root.join("graphrag-rs/docs/microsoft-graphrag-paper.md")),
        dev_root,
        embedder_id,
        dim,
    };
    let db = open_db()?;
    let plan = plan_v2(&db, &sources).map_err(|e| format!("plan: {e}"))?;
    if plan_only {
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&plan).map_err(|e| e.to_string())?
            );
        } else {
            println!(
                "{:<16} {:>10} {:>12} {:>8}",
                "GROUP", "FROM FILES", "FROM CHUNKS", "SINGLE"
            );
            for (g, p) in &plan.per_group {
                println!(
                    "{g:<16} {:>10} {:>12} {:>8}",
                    p.from_files, p.from_chunk_groups, p.single_chunk
                );
            }
            let d = &plan.deletions;
            println!(
                "\ndelete: {} entities, {} relations, {} communities, {} extraction checkpoints; replace {} chunks ({} paper copies dropped)",
                d.entities, d.relations, d.communities, d.extractions, d.chunks, d.paper_chunks
            );
            println!("unresolved: {}", plan.unresolved.len());
            for u in &plan.unresolved {
                println!("  {u}");
            }
        }
        return Ok(());
    }
    let report = apply_v2_migration(&db, &plan, &sources).map_err(|e| format!("apply: {e}"))?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    } else {
        println!(
            "migrated: {} documents, {} chunks, {} sections; {} embeddings re-attached; {} old documents removed",
            report.documents,
            report.chunks,
            report.sections,
            report.embeddings_reattached,
            report.deleted_documents
        );
        println!("next: graphrag work --accept-reextract");
    }
    Ok(())
}

/// The embedder's identity and dimension, from the same environment
/// `open_embedder` reads, without loading the model.
fn embedder_identity() -> (String, usize) {
    let model = embedder_model();
    (graphrag_core::embedder_model_id(model), model.dimension())
}

fn cmd_work(flags: memory::WorkFlags, pause: bool, resume: bool) -> ExitCode {
    let dir = data_dir();
    let flag = graphrag_worker::pause_flag(&dir);
    if pause || resume {
        let r = if pause {
            std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&flag, ""))
        } else {
            match std::fs::remove_file(&flag) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                other => other,
            }
        };
        return match r {
            Ok(()) => {
                println!("worker {}", if pause { "paused" } else { "resumed" });
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let emb = match open_embedder() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(70);
        }
    };
    ExitCode::from(memory::work(&dir, &flags, &emb).code() as u8)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Note {
            text,
            source,
            mut groups,
            doc_key,
            store,
        } => {
            memory::store_alias(store, &mut groups);
            open_embedder().and_then(|emb| {
                let r = memory::note(
                    &data_dir(),
                    &text,
                    &groups,
                    source.as_deref(),
                    doc_key.as_deref(),
                    &emb,
                )?;
                println!(
                    "ok document_id={} chunks={}{}",
                    r.document_id,
                    r.chunks,
                    if r.existing { " (existing text)" } else { "" }
                );
                Ok(())
            })
        }
        Commands::Log { limit, store } => {
            let mut ignored = Vec::new();
            memory::store_alias(store, &mut ignored);
            cmd_log(limit)
        }
        Commands::BackfillEmbeddings { store } => {
            let mut ignored = Vec::new();
            memory::store_alias(store, &mut ignored);
            open_embedder().and_then(|emb| {
                let n = memory::backfill(&data_dir(), &emb)?;
                println!("backfilled={n}");
                Ok(())
            })
        }
        Commands::Work {
            groups,
            once,
            max_items,
            retry_failed,
            accept_reextract,
            pause,
            resume,
        } => {
            return cmd_work(
                memory::WorkFlags {
                    groups,
                    once,
                    max_items,
                    retry_failed,
                    accept_reextract,
                },
                pause,
                resume,
            );
        }
        Commands::Migrate {
            what:
                MigrateCmd::V2 {
                    plan,
                    apply,
                    repo_root,
                    dev_root,
                    paper,
                    json,
                },
        } => cmd_migrate_v2(plan, apply, repo_root, dev_root, paper, json),
        Commands::Enrich { .. } => {
            eprintln!("warning: `enrich` is deprecated; use `graphrag work`");
            return cmd_work(
                memory::WorkFlags {
                    groups: vec![],
                    once: false,
                    max_items: None,
                    retry_failed: false,
                    accept_reextract: false,
                },
                false,
                false,
            );
        }
        Commands::Ask {
            query,
            top,
            mut groups,
            json,
            lanes,
            store,
        } => {
            memory::store_alias(store, &mut groups);
            let lanes = match lanes.as_deref().map(graphrag_core::recall::Lanes::parse) {
                None => Ok(graphrag_core::recall::Lanes::default()),
                Some(r) => r.map_err(|e| e.to_string()),
            };
            lanes.and_then(|lanes| {
                let emb = open_embedder()?;
                let hits = memory::ask(&data_dir(), &query, &groups, top, lanes, &emb)?;
                if json {
                    memory::print_hits_json(&hits)
                } else {
                    memory::print_hits(&hits);
                    Ok(())
                }
            })
        }
        Commands::Export { store } => cmd_export(&store),
        Commands::Chunk {
            path,
            size,
            overlap,
            markdown,
            nth,
            format,
            code,
            lang,
            lines,
            expand,
        } => cmd_chunk(
            &path,
            ChunkOpts {
                size,
                overlap,
                markdown,
                nth,
                format: &format,
                code,
                lang: lang.as_deref(),
                lines,
                expand,
            },
        ),
        Commands::Embed { store, limit } => {
            cmd_embed(store.as_deref().unwrap_or(DEFAULT_STORE), limit)
        }
        Commands::Status { store, json } => {
            let mut ignored = Vec::new();
            memory::store_alias(store, &mut ignored);
            let (id, dim) = embedder_identity();
            memory::status(
                &data_dir(),
                json,
                &memory::status_config(&data_dir(), id, dim),
            )
        }
        Commands::Plan {
            stage,
            store,
            model,
            limit,
        } => cmd_plan(
            &stage,
            store.as_deref().unwrap_or(DEFAULT_STORE),
            &model,
            limit,
        ),
        Commands::Apply { stage, store } => {
            cmd_apply(&stage, store.as_deref().unwrap_or(DEFAULT_STORE))
        }
        Commands::Provenance { nonce, store } => {
            cmd_provenance(store.as_deref().unwrap_or(DEFAULT_STORE), &nonce)
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphrag_core::text_embedder::TextEmbedder;
    use graphrag_core::{BruteForceVectorSource, VectorCandidateSource};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "graphrag-cli-unit-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    struct FakeEmbedder;

    impl TextEmbedder for FakeEmbedder {
        fn id(&self) -> String {
            "fake".into()
        }
        fn dim(&self) -> usize {
            3
        }
        fn embed(&self, text: &str) -> Result<Vec<f32>, graphrag_core::GraphRagError> {
            Ok(if text.contains("beta") {
                vec![0.0, 1.0, 0.0]
            } else {
                vec![1.0, 0.0, 0.0]
            })
        }
    }

    #[test]
    fn note_writes_chunk_and_vector_index() {
        // SAFETY: unit tests don't read GRAPHRAG_BIN concurrently with this write.
        unsafe { std::env::set_var("GRAPHRAG_BIN", "/usr/bin/true") };
        let dir = temp_dir("note-vector-index");
        let r = memory::note(&dir, "alpha note", &[], Some("unit"), None, &FakeEmbedder).unwrap();
        let db = Database::open(&dir.join("graphrag.db")).unwrap();
        assert_eq!(db.get_store(memory::MEMORY_STORE).unwrap().dim, 3);
        let chunk = db.document_chunks(r.document_id).unwrap()[0];
        let source = BruteForceVectorSource::for_store(&db, memory::MEMORY_STORE).unwrap();
        assert_eq!(
            source.top_candidates(&[1.0, 0.0, 0.0], 1).unwrap()[0].chunk_id,
            chunk
        );
    }

    #[test]
    fn backfill_embeds_only_missing_chunks() {
        let dir = temp_dir("backfill");
        std::fs::create_dir_all(&dir).unwrap();
        let db = Database::open(&dir.join("graphrag.db")).unwrap();
        db.create_store(memory::MEMORY_STORE, 3).unwrap();
        let alpha = db
            .add_chunk(memory::MEMORY_STORE, "alpha note", None, None)
            .unwrap();
        let beta = db
            .add_chunk(memory::MEMORY_STORE, "beta note", None, None)
            .unwrap();
        assert_eq!(memory::backfill(&dir, &FakeEmbedder).unwrap(), 2);
        assert_eq!(
            memory::backfill(&dir, &FakeEmbedder).unwrap(),
            0,
            "current chunks are skipped"
        );
        let source = BruteForceVectorSource::for_store(&db, memory::MEMORY_STORE).unwrap();
        assert_eq!(
            source.top_candidates(&[1.0, 0.0, 0.0], 1).unwrap()[0].chunk_id,
            alpha
        );
        assert_eq!(
            source.top_candidates(&[0.0, 1.0, 0.0], 1).unwrap()[0].chunk_id,
            beta
        );
    }
}
