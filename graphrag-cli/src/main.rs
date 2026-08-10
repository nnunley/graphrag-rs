use clap::{Parser, Subcommand};
use graphrag_core::{
    BruteForceVectorSource, CommunityGraph, Database, Embedder, EmbedderConfig, EmbedderModel,
    LexicalIndex, RemoteEmbedderConfig, VectorCandidateSource, default_embedder_cache_dir,
    load_standard_synonyms, load_standard_type_synonyms,
};
use graphrag_llm::{ChatClient, EntityTriple, OllamaChatClient, strategy_for_model};
use std::collections::HashMap;
mod llm_strategy;

use std::io::{BufRead, Write};

use std::path::PathBuf;
use std::process::ExitCode;

const DEFAULT_STORE: &str = "conversations";

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
    Ask {
        /// The query text
        query: String,
        /// Maximum number of hits to return (default: 5)
        #[arg(long, default_value_t = 5)]
        top: usize,
        /// Store name (default: conversations)
        #[arg(long)]
        store: Option<String>,
    },
    /// Show recent notes
    Log {
        /// Maximum number of notes to show (default: 20)
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Store name (default: conversations)
        #[arg(long)]
        store: Option<String>,
    },
    /// Rebuild vector indexes for existing chunks
    BackfillEmbeddings {
        /// Store name (default: conversations)
        #[arg(long)]
        store: Option<String>,
    },
    /// Detect and persist communities for an existing entity graph
    Enrich {
        /// Store name (default: conversations)
        #[arg(long)]
        store: Option<String>,
        /// Clear existing communities before detecting new ones
        #[arg(long)]
        clear: bool,
        /// Extract entities and relations from chunks via Ollama before building communities
        #[arg(long)]
        extract: bool,
        /// Maximum Leiden iterations
        #[arg(long, default_value_t = 100)]
        max_iterations: usize,
        /// Leiden convergence tolerance
        #[arg(long, default_value_t = 1e-6)]
        tolerance: f64,
        /// Maximum hierarchy levels (1 = flat)
        #[arg(long, default_value_t = 1)]
        levels: i32,
        /// Minimum community size to partition further
        #[arg(long, default_value_t = 5)]
        min_size: usize,
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
    /// Report pipeline phase status: what exists, what is ready, what changed
    Status {
        /// Store name
        #[arg(long)]
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

    /// Forward-compat sidecar for leit Phase 2 cursor-based persistence.
    /// Today these bytes can't be reloaded into a searchable index — we
    /// write them anyway so the on-disk format is in place when leit ships
    /// the load path.
    fn lexical_segment_path(&self, store: &str) -> PathBuf {
        self.data_dir
            .join("indexes")
            .join(format!("{store}.leitseg"))
    }
}

fn open_db() -> Result<Database, String> {
    open_db_at(&StoragePaths::from_env())
}

fn open_db_at(paths: &StoragePaths) -> Result<Database, String> {
    std::fs::create_dir_all(&paths.data_dir).map_err(|e| format!("create data dir: {e}"))?;
    Database::open(&paths.db_path()).map_err(|e| format!("open db: {e}"))
}

fn open_embedder() -> Result<Embedder, String> {
    let model = std::env::var("GRAPHRAG_EMBED_MODEL")
        .ok()
        .and_then(|model| match model.to_lowercase().as_str() {
            "minilm" | "mini" => Some(EmbedderModel::MiniLM),
            "nomic" | "nomic-embed-text" => Some(EmbedderModel::NomicEmbedText),
            "openai" | "openai-ada002" | "ada002" => Some(EmbedderModel::OpenAIAda002),
            "openai3" | "openai-3-small" | "openai3-small" => Some(EmbedderModel::OpenAI3Small),
            _ => None,
        })
        .unwrap_or(EmbedderModel::NomicEmbedText);

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

fn cmd_note(text: &str, source: Option<&str>, store: &str) -> Result<(), String> {
    let paths = StoragePaths::from_env();
    let embedder = open_embedder()?;
    let chunk_id = cmd_note_with_embedder(&paths, text, source, store, |text| {
        embedder.embed(text).map_err(|e| e.to_string())
    })?;
    println!("ok chunk_id={chunk_id}");
    Ok(())
}

fn cmd_note_with_embedder<F>(
    paths: &StoragePaths,
    text: &str,
    source: Option<&str>,
    store: &str,
    embed: F,
) -> Result<i64, String>
where
    F: Fn(&str) -> Result<Vec<f32>, String>,
{
    let embedding = embed(text)?;
    let db = open_db_at(paths)?;
    let store_record = match db.get_store(store) {
        Ok(store_record) => store_record,
        Err(_) => db
            .create_store(store, embedding.len())
            .map_err(|e| format!("create store {store}: {e}"))?,
    };
    if store_record.dim != embedding.len() {
        return Err(format!(
            "dimension mismatch: store has {}, embedder produced {}",
            store_record.dim,
            embedding.len()
        ));
    }

    let chunk_id = db
        .add_chunk(store, text, source, None)
        .map_err(|e| format!("add chunk: {e}"))?;
    db.set_chunk_embedding(chunk_id, &embedding)
        .map_err(|e| format!("store embedding: {e}"))?;
    rebuild_lexical_sidecar(paths, &db, store)?;
    Ok(chunk_id)
}

/// Rebuild the leit segment sidecar for a store. Called after every chunk
/// add; cheap at small corpora, scales with chunk count. leit Phase 1 has no
/// incremental-add API, so full rebuild is the only option.
fn rebuild_lexical_sidecar(paths: &StoragePaths, db: &Database, store: &str) -> Result<(), String> {
    let chunks = db
        .list_chunks(store)
        .map_err(|e| format!("list chunks for lexical rebuild: {e}"))?;
    let Some(index) = LexicalIndex::build_from_chunks(&chunks)
        .map_err(|e| format!("build lexical index: {e}"))?
    else {
        // Empty store — nothing to write. Don't leave a stale sidecar.
        let path = paths.lexical_segment_path(store);
        let _ = std::fs::remove_file(&path);
        return Ok(());
    };
    let bytes = index
        .to_segment_bytes()
        .map_err(|e| format!("serialize lexical segment: {e}"))?;
    let path = paths.lexical_segment_path(store);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create lexical dir: {e}"))?;
    }
    std::fs::write(&path, &bytes)
        .map_err(|e| format!("write lexical segment {}: {e}", path.display()))?;
    Ok(())
}

fn cmd_backfill_embeddings(store: &str) -> Result<(), String> {
    let paths = StoragePaths::from_env();
    eprintln!("initializing embedder...");
    let embedder = open_embedder()?;
    let count = cmd_backfill_embeddings_with_embedder(&paths, store, |text| {
        embedder.embed(text).map_err(|e| e.to_string())
    })?;
    println!("store={store} backfilled={count}");
    Ok(())
}

fn cmd_backfill_embeddings_with_embedder<F>(
    paths: &StoragePaths,
    store: &str,
    embed: F,
) -> Result<usize, String>
where
    F: Fn(&str) -> Result<Vec<f32>, String>,
{
    let db = open_db_at(paths)?;
    let store_record = db
        .get_store(store)
        .map_err(|e| format!("get store {store}: {e}"))?;
    let chunks = db
        .list_chunks(store)
        .map_err(|e| format!("list chunks: {e}"))?;
    for (idx, chunk) in chunks.iter().enumerate() {
        let embedding = embed(&chunk.content)?;
        if embedding.len() != store_record.dim {
            return Err(format!(
                "dimension mismatch for chunk {}: store has {}, embedder produced {}",
                chunk.id,
                store_record.dim,
                embedding.len()
            ));
        }
        db.set_chunk_embedding(chunk.id, &embedding)
            .map_err(|e| format!("store embedding for chunk {}: {e}", chunk.id))?;
        let completed = idx + 1;
        if completed == chunks.len() || completed % 25 == 0 {
            eprintln!("backfilled {completed}/{} chunks", chunks.len());
        }
    }

    Ok(chunks.len())
}

fn cmd_ask(query: &str, top: usize, store: &str) -> Result<(), String> {
    // Hybrid recall: brute-force cosine (semantic) + leit BM25 (lexical), fused via RRF.
    //
    // Mirrors the MCP server's tool_recall in shape (same embedder, same
    // chunk store), but stays CLI-shaped: rebuild the leit index per
    // invocation rather than caching, since the CLI is one-shot.
    let paths = StoragePaths::from_env();
    let db = open_db_at(&paths)?;
    let store_record = match db.get_store(store) {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };

    // --- Lane 1: exact vector recall ---
    let embedder = open_embedder()?;
    let embedding = embedder.embed(query).map_err(|e| e.to_string())?;
    if embedding.len() != store_record.dim {
        return Err(format!(
            "dimension mismatch: embedder produced {}, store has {}",
            embedding.len(),
            store_record.dim
        ));
    }
    // Pull a larger candidate pool than `top` so RRF has overlap to fuse.
    let candidate_k = top.max(5) * 4;
    let vector_source =
        BruteForceVectorSource::for_store(&db, store).map_err(|e| format!("load vectors: {e}"))?;
    let vector_hits = vector_source
        .top_candidates(&embedding, candidate_k)
        .map_err(|e| format!("vector search: {e}"))?;

    // --- Lane 2: leit BM25 recall (rebuild per query) ---
    let chunks_all = db
        .list_chunks(store)
        .map_err(|e| format!("list chunks: {e}"))?;
    let lexical_hits = match LexicalIndex::build_from_chunks(&chunks_all)
        .map_err(|e| format!("build lexical: {e}"))?
    {
        Some(idx) => idx
            .search(query, candidate_k)
            .map_err(|e| format!("lexical search: {e}"))?,
        None => Vec::new(),
    };

    if vector_hits.is_empty() && lexical_hits.is_empty() {
        return Ok(());
    }

    // --- Fuse via RRF (k=60, leit default) ---
    let vector_ranked: Vec<graphrag_core::leit_fusion::RankedResult> = vector_hits
        .iter()
        .enumerate()
        .map(|(i, h)| graphrag_core::leit_fusion::RankedResult::new(h.chunk_id.to_string(), i + 1))
        .collect();
    let lexical_ranked: Vec<graphrag_core::leit_fusion::RankedResult> = lexical_hits
        .iter()
        .enumerate()
        .map(|(i, (id, _))| graphrag_core::leit_fusion::RankedResult::new(id.to_string(), i + 1))
        .collect();
    let fused = graphrag_core::leit_fusion::fuse_default(&[vector_ranked, lexical_ranked]);

    // --- Render top-k ---
    let final_ids: Vec<i64> = fused
        .iter()
        .take(top)
        .filter_map(|f| f.id.parse().ok())
        .collect();
    let chunks = db
        .get_chunks_by_ids(&final_ids)
        .map_err(|e| format!("fetch chunks: {e}"))?;
    let chunks_by_id: HashMap<i64, &graphrag_core::Chunk> =
        chunks.iter().map(|c| (c.id, c)).collect();
    // Also surface per-lane source/score for transparency during the
    // hybrid-recall bringup. Drop these badges once the tuning is stable.
    // `vec=` is cosine similarity (higher = better), from BruteForceVectorSource.
    let vec_lookup: HashMap<i64, f32> = vector_hits.iter().map(|h| (h.chunk_id, h.score)).collect();
    let lex_lookup: HashMap<i64, f32> = lexical_hits.iter().copied().collect();
    for (rank, f) in fused.iter().take(top).enumerate() {
        let Ok(id) = f.id.parse::<i64>() else {
            continue;
        };
        let Some(chunk) = chunks_by_id.get(&id) else {
            continue;
        };
        let src = chunk.source.as_deref().unwrap_or("-");
        let vec_badge = vec_lookup
            .get(&id)
            .map(|s| format!("vec={s:.3}"))
            .unwrap_or_else(|| "vec=-".to_string());
        let lex_badge = lex_lookup
            .get(&id)
            .map(|s| format!("lex={s:.3}"))
            .unwrap_or_else(|| "lex=-".to_string());
        println!(
            "#{rank}\trrf={:.4}\t{vec_badge}\t{lex_badge}\t{}\t[{}]\t{}",
            f.score, chunk.created_at, src, chunk.content
        );
    }
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

fn cmd_enrich(
    store: &str,
    clear: bool,
    extract: bool,
    max_iterations: usize,
    tolerance: f64,
    levels: i32,
    min_size: usize,
) -> Result<(), String> {
    if levels < 1 {
        return Err("--levels must be at least 1".to_string());
    }
    if min_size < 1 {
        return Err("--min-size must be at least 1".to_string());
    }

    let db = open_db()?;
    let _ = db
        .get_store(store)
        .map_err(|e| format!("get store {store}: {e}"))?;

    load_standard_synonyms(&db).map_err(|e| format!("load relation synonyms: {e}"))?;
    load_standard_type_synonyms(&db).map_err(|e| format!("load entity type synonyms: {e}"))?;

    let extracted = if extract {
        let client = OllamaChatClient::from_env()?;
        Some(extract_entities_for_chunks(
            &db,
            store,
            &client,
            client.model(),
        )?)
    } else {
        None
    };

    if clear {
        db.clear_communities(store)
            .map_err(|e| format!("clear communities: {e}"))?;
    }

    let entities = db
        .list_entities(store)
        .map_err(|e| format!("list entities: {e}"))?;
    let relations = db
        .list_relations(store)
        .map_err(|e| format!("list relations: {e}"))?;

    let mut graph = CommunityGraph::new();
    for entity in &entities {
        graph.add_node(entity.id);
    }
    for relation in &relations {
        graph.add_edge(relation.head_id, relation.tail_id, 1.0);
    }

    let (community_count, depth, modularity) = if levels > 1 {
        persist_hierarchical_communities(
            &db,
            store,
            &graph,
            max_iterations,
            tolerance,
            min_size,
            levels,
        )?
    } else {
        persist_flat_communities(&db, store, &graph, max_iterations, tolerance)?
    };

    println!(
        "store={store} entities={} relations={} communities={} depth={} modularity={:.6}",
        entities.len(),
        relations.len(),
        community_count,
        depth,
        modularity
    );
    if let Some(extracted) = extracted {
        eprintln!("extracted={extracted} entity_relations");
    }
    Ok(())
}

fn extract_entities_for_chunks<C: ChatClient>(
    db: &Database,
    store: &str,
    client: &C,
    model: &str,
) -> Result<usize, String> {
    let chunks = db
        .list_chunks(store)
        .map_err(|e| format!("list chunks: {e}"))?;
    let total = chunks.len();
    let mut extracted = 0;

    // Resolve the per-model extraction strategy (2026-08-03 spike): triple-lines default,
    // JSON for nemotron-class. The strategy owns prompt-build, output-format, and parse.
    let strategy = strategy_for_model(model);
    let response_format = strategy.response_format();

    for (idx, chunk) in chunks.iter().enumerate() {
        let prompt = strategy.build_prompt(&chunk.content, idx, total);
        let response = client.complete_with_format(&prompt, response_format.as_ref())?;
        let parsed = strategy.parse(&response);
        persist_entity_triples(db, store, chunk.id, &parsed.triples)?;
        extracted += parsed.triples.len();

        let completed = idx + 1;
        if completed == total || completed % 25 == 0 {
            eprintln!("extracted metadata for {completed}/{total} chunks");
        }
    }

    Ok(extracted)
}

fn persist_entity_triples(
    db: &Database,
    store: &str,
    chunk_id: i64,
    triples: &[EntityTriple],
) -> Result<(), String> {
    for triple in triples {
        let head_id = db
            .get_or_create_entity(store, &triple.head, triple.head_type.as_deref(), None)
            .map_err(|e| format!("create head entity: {e}"))?;
        let tail_id = db
            .get_or_create_entity(store, &triple.tail, triple.tail_type.as_deref(), None)
            .map_err(|e| format!("create tail entity: {e}"))?;
        db.add_relation(store, head_id, tail_id, &triple.relation, None)
            .map_err(|e| format!("add relation: {e}"))?;
        db.link_chunk_entity(chunk_id, head_id)
            .map_err(|e| format!("link head entity: {e}"))?;
        db.link_chunk_entity(chunk_id, tail_id)
            .map_err(|e| format!("link tail entity: {e}"))?;
    }

    Ok(())
}

fn persist_flat_communities(
    db: &Database,
    store: &str,
    graph: &CommunityGraph,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(usize, i32, f64), String> {
    let result = graph.leiden(Some(max_iterations), tolerance);
    let mut community_count = 0;

    for community in &result.communities {
        let community_id = db
            .create_community(store, 0, result.modularity, None)
            .map_err(|e| format!("create community: {e}"))?;
        community_count += 1;

        for entity_id in community.collect_nodes() {
            db.link_entity_community(entity_id, community_id)
                .map_err(|e| format!("link entity community: {e}"))?;
        }
    }

    Ok((community_count, 1, result.modularity))
}

fn persist_hierarchical_communities(
    db: &Database,
    store: &str,
    graph: &CommunityGraph,
    max_iterations: usize,
    tolerance: f64,
    min_size: usize,
    levels: i32,
) -> Result<(usize, i32, f64), String> {
    let result = graph.leiden_hierarchical(Some(max_iterations), tolerance, min_size, Some(levels));
    let mut index_to_db_id: HashMap<usize, i64> = HashMap::new();
    let top_modularity = result
        .communities
        .first()
        .map(|community| community.modularity)
        .unwrap_or(0.0);

    for (idx, community) in result.communities.iter().enumerate() {
        let parent_id = community
            .parent_index
            .and_then(|parent_index| index_to_db_id.get(&parent_index).copied());
        let community_id = db
            .create_community(store, community.level, community.modularity, parent_id)
            .map_err(|e| format!("create community: {e}"))?;
        index_to_db_id.insert(idx, community_id);

        for &entity_id in &community.nodes {
            db.link_entity_community(entity_id, community_id)
                .map_err(|e| format!("link entity community: {e}"))?;
        }
    }

    Ok((result.communities.len(), result.depth, top_modularity))
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

fn cmd_status(store: &str, json: bool) -> Result<(), String> {
    let db = open_db()?;
    let st = graphrag_core::mr::pipeline_status(&db, store).map_err(|e| format!("status: {e}"))?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&st).map_err(|e| format!("serialize: {e}"))?
        );
        return Ok(());
    }
    println!("store: {}", st.store);
    println!(
        "{:<12} {:>7} {:>7} {:>8} {:>8}  GUIDANCE",
        "PHASE", "TOTAL", "DONE", "PENDING", "BLOCKED"
    );
    for p in &st.phases {
        println!(
            "{:<12} {:>7} {:>7} {:>8} {:>8}  {}",
            p.phase, p.total, p.done, p.pending, p.blocked, p.guidance
        );
        for (level, count, done) in &p.levels {
            println!("             level {level}: {done}/{count} summarized");
        }
    }
    println!("\nnext: {}", st.next);
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
        Commands::BackfillEmbeddings { store } => {
            cmd_backfill_embeddings(store.as_deref().unwrap_or(DEFAULT_STORE))
        }
        Commands::Enrich {
            store,
            clear,
            extract,
            max_iterations,
            tolerance,
            levels,
            min_size,
        } => cmd_enrich(
            store.as_deref().unwrap_or(DEFAULT_STORE),
            clear,
            extract,
            max_iterations,
            tolerance,
            levels,
            min_size,
        ),
        Commands::Ask { query, top, store } => {
            cmd_ask(&query, top, store.as_deref().unwrap_or(DEFAULT_STORE))
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
            cmd_status(store.as_deref().unwrap_or(DEFAULT_STORE), json)
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
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_paths(label: &str) -> StoragePaths {
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let data_dir =
            std::env::temp_dir().join(format!("graphrag-cli-unit-{label}-{pid}-{nanos}"));
        StoragePaths { data_dir }
    }

    fn fake_embed(text: &str) -> Result<Vec<f32>, String> {
        if text.contains("beta") {
            Ok(vec![0.0, 1.0, 0.0])
        } else {
            Ok(vec![1.0, 0.0, 0.0])
        }
    }

    struct FakeChatClient;

    impl ChatClient for FakeChatClient {
        fn complete(&self, _prompt: &graphrag_llm::ChatPrompt) -> Result<String, String> {
            Ok(r#"{"entities":[{"head":"Alice","head_type":"Person","relation":"uses","tail":"GraphRAG","tail_type":"Software"}]}"#.to_string())
        }
    }

    #[test]
    fn note_writes_chunk_and_vector_index() {
        let paths = temp_paths("note-vector-index");

        let chunk_id = cmd_note_with_embedder(
            &paths,
            "alpha note",
            Some("unit"),
            DEFAULT_STORE,
            fake_embed,
        )
        .expect("note should write db and embedding");

        let db = Database::open(&paths.db_path()).expect("db opens");
        let store = db.get_store(DEFAULT_STORE).expect("store exists");
        assert_eq!(store.dim, 3);

        let source =
            BruteForceVectorSource::for_store(&db, DEFAULT_STORE).expect("vector source loads");
        let results = source
            .top_candidates(&[1.0, 0.0, 0.0], 1)
            .expect("search works");
        assert_eq!(results[0].chunk_id, chunk_id);
    }

    #[test]
    fn backfill_embeddings_rebuilds_index_for_existing_chunks() {
        let paths = temp_paths("backfill-vector-index");
        let db = Database::open(&paths.db_path()).expect("db opens");
        db.create_store(DEFAULT_STORE, 3).expect("store created");
        let alpha_id = db
            .add_chunk(DEFAULT_STORE, "alpha note", None, None)
            .expect("alpha chunk");
        let beta_id = db
            .add_chunk(DEFAULT_STORE, "beta note", None, None)
            .expect("beta chunk");

        let count = cmd_backfill_embeddings_with_embedder(&paths, DEFAULT_STORE, fake_embed)
            .expect("backfill should store embeddings");

        assert_eq!(count, 2);
        let source =
            BruteForceVectorSource::for_store(&db, DEFAULT_STORE).expect("vector source loads");
        let alpha = source
            .top_candidates(&[1.0, 0.0, 0.0], 1)
            .expect("alpha search");
        let beta = source
            .top_candidates(&[0.0, 1.0, 0.0], 1)
            .expect("beta search");
        assert_eq!(alpha[0].chunk_id, alpha_id);
        assert_eq!(beta[0].chunk_id, beta_id);
    }

    #[test]
    fn extract_entities_for_chunks_persists_llm_metadata() {
        let paths = temp_paths("extract-entities");
        let db = Database::open(&paths.db_path()).expect("db opens");
        db.create_store(DEFAULT_STORE, 3).expect("store created");
        let chunk_id = db
            .add_chunk(DEFAULT_STORE, "Alice uses GraphRAG.", None, None)
            .expect("chunk");

        // "nemotron" resolves to JsonStrategy, matching FakeChatClient's JSON output.
        let extracted =
            extract_entities_for_chunks(&db, DEFAULT_STORE, &FakeChatClient, "nemotron3:33b")
                .expect("metadata extraction succeeds");

        assert_eq!(extracted, 1);
        let alice = db
            .get_entity_by_name(DEFAULT_STORE, "Alice")
            .expect("head entity");
        let graph = db
            .get_entity_by_name(DEFAULT_STORE, "GraphRAG")
            .expect("tail entity");
        let relations = db.get_relations_for_entity(alice.id).expect("relations");
        let linked = db.get_entities_for_chunk(chunk_id).expect("chunk entities");

        assert_eq!(relations[0].tail_id, graph.id);
        assert!(linked.iter().any(|entity| entity.name == "Alice"));
        assert!(linked.iter().any(|entity| entity.name == "GraphRAG"));
    }
}
