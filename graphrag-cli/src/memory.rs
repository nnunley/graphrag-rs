//! The one-graph surfaces: `note`, `ask`, `work`, `status`,
//! `backfill-embeddings`. Everything lives in one store; groups scope it.

use graphrag_core::documents::{AddDocument, AddOutcome, Origin};
use graphrag_core::fingerprints::{ChunkerProfile, embed_fp};
use graphrag_core::status::{StaticConfig, Step, StepState, SubjectKind};
use graphrag_core::text_embedder::TextEmbedder;
use graphrag_core::{Database, LexicalIndex};
use graphrag_worker::{ExitStatus, WorkerConfig};
use std::path::Path;

/// The single store everything is added to.
pub const MEMORY_STORE: &str = "memory";
/// Default group for notes.
pub const DEFAULT_GROUP: &str = "conversations";

/// Map a deprecated `--store` onto groups, with a warning.
pub fn store_alias(store: Option<String>, groups: &mut Vec<String>) {
    if let Some(s) = store {
        eprintln!("warning: --store is deprecated; use --in");
        groups.push(s);
    }
}

fn open(data_dir: &Path) -> Result<Database, String> {
    std::fs::create_dir_all(data_dir).map_err(|e| format!("create data dir: {e}"))?;
    Database::open(&data_dir.join("graphrag.db")).map_err(|e| format!("open db: {e}"))
}

fn ensure_store(db: &Database, dim: usize) -> Result<(), String> {
    match db.get_store(MEMORY_STORE) {
        Ok(s) if s.dim != dim => Err(format!(
            "dimension mismatch: store has {}, embedder produced {dim}",
            s.dim
        )),
        Ok(_) => Ok(()),
        Err(_) => db
            .create_store(MEMORY_STORE, dim)
            .map(|_| ())
            .map_err(|e| format!("create store: {e}")),
    }
}

/// Embed a chunk unless its embed step is already current for this embedder.
fn embed_chunk(db: &Database, emb: &dyn TextEmbedder, chunk: i64) -> Result<bool, String> {
    let text = db.chunk_text(chunk).map_err(|e| e.to_string())?;
    let fp = embed_fp(&text, &emb.id(), emb.dim());
    let current = db
        .step_fp(SubjectKind::Chunk, chunk, Step::Embed)
        .map_err(|e| e.to_string())?;
    if current
        .as_ref()
        .is_some_and(|(f, s)| *f == fp && s == "current")
    {
        return Ok(false);
    }
    let v = emb.embed(&text).map_err(|e| e.to_string())?;
    db.set_chunk_embedding(chunk, &v)
        .map_err(|e| e.to_string())?;
    db.record_step(
        SubjectKind::Chunk,
        chunk,
        Step::Embed,
        &fp,
        StepState::Current,
        None,
    )
    .map_err(|e| e.to_string())?;
    Ok(true)
}

/// Rebuild the leit segment sidecar (no incremental add in leit Phase 1).
pub fn rebuild_lexical_sidecar(data_dir: &Path, db: &Database, store: &str) -> Result<(), String> {
    let chunks = db
        .list_chunks(store)
        .map_err(|e| format!("list chunks for lexical rebuild: {e}"))?;
    let path = data_dir.join("indexes").join(format!("{store}.leitseg"));
    let Some(index) = LexicalIndex::build_from_chunks(&chunks)
        .map_err(|e| format!("build lexical index: {e}"))?
    else {
        let _ = std::fs::remove_file(&path);
        return Ok(());
    };
    let bytes = index
        .to_segment_bytes()
        .map_err(|e| format!("serialize lexical segment: {e}"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create lexical dir: {e}"))?;
    }
    std::fs::write(&path, &bytes)
        .map_err(|e| format!("write lexical segment {}: {e}", path.display()))
}

fn spawn_worker(data_dir: &Path) {
    if let Err(e) = graphrag_worker::spawn_detached(data_dir) {
        eprintln!("warning: could not start the worker: {e}");
    }
}

pub struct NoteResult {
    pub document_id: i64,
    pub chunks: usize,
    pub existing: bool,
}

/// Add a note (or any text) as a document, embed its chunks now so it is
/// searchable immediately, and start the worker for the remaining steps.
pub fn note(
    data_dir: &Path,
    text: &str,
    groups: &[String],
    source: Option<&str>,
    doc_key: Option<&str>,
    emb: &dyn TextEmbedder,
) -> Result<NoteResult, String> {
    let db = open(data_dir)?;
    ensure_store(&db, emb.dim())?;
    let groups: Vec<&str> = if groups.is_empty() {
        vec![DEFAULT_GROUP]
    } else {
        groups.iter().map(String::as_str).collect()
    };
    let profile = ChunkerProfile::default_markdown();
    let outcome = db
        .add_document(&AddDocument {
            store: MEMORY_STORE,
            content: text,
            origin: Origin::Source,
            source,
            doc_key,
            title: None,
            groups: &groups,
            surface: "cli",
            profile: &profile,
        })
        .map_err(|e| format!("add document: {e}"))?;
    let (document_id, existing) = match outcome {
        AddOutcome::Created { document_id } | AddOutcome::Replaced { document_id, .. } => {
            (document_id, false)
        }
        AddOutcome::Existing { document_id, .. } => (document_id, true),
    };
    let chunks = db.document_chunks(document_id).map_err(|e| e.to_string())?;
    for c in &chunks {
        embed_chunk(&db, emb, *c)?;
    }
    rebuild_lexical_sidecar(data_dir, &db, MEMORY_STORE)?;
    spawn_worker(data_dir);
    Ok(NoteResult {
        document_id,
        chunks: chunks.len(),
        existing,
    })
}

pub use graphrag_core::recall::RecallHit as Hit;

/// Hybrid recall (vector + BM25 + graph expansion, see core). Returned
/// chunks with incomplete steps are escalated and a worker is started.
pub fn ask(
    data_dir: &Path,
    query: &str,
    groups: &[String],
    top: usize,
    lanes: graphrag_core::recall::Lanes,
    emb: &dyn TextEmbedder,
) -> Result<Vec<Hit>, String> {
    let db = open(data_dir)?;
    let Ok(store) = db.get_store(MEMORY_STORE) else {
        return Ok(Vec::new());
    };
    let embedding = emb.embed(query).map_err(|e| e.to_string())?;
    if embedding.len() != store.dim {
        return Err(format!(
            "dimension mismatch: embedder produced {}, store has {}",
            embedding.len(),
            store.dim
        ));
    }
    let groups: Vec<&str> = groups.iter().map(String::as_str).collect();
    let now = db.now_utc().map_err(|e| e.to_string())?;
    let hits = db
        .hybrid_recall_lanes(
            MEMORY_STORE,
            query,
            &embedding,
            &groups,
            top,
            None,
            &now,
            lanes,
        )
        .map_err(|e| format!("recall: {e}"))?;
    let ids: Vec<i64> = hits.iter().map(|h| h.chunk_id).collect();
    // Retrieval sets priority: never waits on extraction.
    if db.escalate_chunks(&ids).map_err(|e| e.to_string())? > 0 {
        spawn_worker(data_dir);
    }
    Ok(hits)
}

/// One JSON array of hits, for tooling and the eval harness.
pub fn print_hits_json(hits: &[Hit]) -> Result<(), String> {
    let v: Vec<serde_json::Value> = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "chunk_id": h.chunk_id,
                "document_id": h.document_id,
                "source": h.source,
                "groups": h.groups,
                "heading_path": h.heading_path,
                "rrf": h.rrf,
                "lanes": {"vector": h.vec, "lexical": h.lex, "graph": h.graph},
                "created_at": h.created_at,
                "content": h.content,
            })
        })
        .collect();
    println!("{}", serde_json::to_string(&v).map_err(|e| e.to_string())?);
    Ok(())
}

pub fn print_hits(hits: &[Hit]) {
    let opt = |v: Option<String>| v.unwrap_or_else(|| "-".into());
    for (rank, h) in hits.iter().enumerate() {
        println!(
            "#{rank}\tchunk={}\trrf={:.4}\tvec={}\tlex={}\tgraph={}\t{}\t[{}]\t{}\t{}\t{}",
            h.chunk_id,
            h.rrf,
            opt(h.vec.map(|s| format!("{s:.3}"))),
            opt(h.lex.map(|s| format!("{s:.3}"))),
            opt(h.graph.map(|s| format!("{s:.3}"))),
            h.created_at,
            h.source.as_deref().unwrap_or("-"),
            if h.groups.is_empty() {
                "-".into()
            } else {
                h.groups.join(",")
            },
            if h.heading_path.is_empty() {
                "-".into()
            } else {
                h.heading_path.join(" › ")
            },
            h.content
        );
    }
}

pub struct WorkFlags {
    pub groups: Vec<String>,
    pub once: bool,
    pub max_items: Option<usize>,
    pub retry_failed: bool,
    pub accept_reextract: bool,
}

/// The worker's configuration from flags and environment.
pub fn worker_config(data_dir: &Path, flags: &WorkFlags) -> WorkerConfig {
    let mut cfg = WorkerConfig::from_env(data_dir.to_path_buf());
    cfg.store = MEMORY_STORE.into();
    cfg.max_items = if flags.once {
        Some(cfg.concurrency.max(1))
    } else {
        flags.max_items
    };
    cfg.group = flags.groups.first().cloned();
    cfg.retry_failed = flags.retry_failed;
    cfg.accept_reextract = flags.accept_reextract;
    cfg
}

pub fn work(data_dir: &Path, flags: &WorkFlags, emb: &dyn TextEmbedder) -> ExitStatus {
    let cfg = worker_config(data_dir, flags);
    let llm = graphrag_llm::openai::OpenAiChatClient::from_env()
        .with_thinking(graphrag_llm::paper_json::ENABLE_THINKING);
    let status = graphrag_worker::run(&cfg, &llm, emb);
    match &status {
        ExitStatus::Done => eprintln!("work: done"),
        ExitStatus::AlreadyRunning => {}
        ExitStatus::Usage => eprintln!("work: refused (see above)"),
        ExitStatus::Unavailable => eprintln!("work: LLM endpoint or model unavailable"),
        ExitStatus::Outage => eprintln!("work: LLM outage; will retry on the next trigger"),
        ExitStatus::Error(e) => eprintln!("work: error: {e}"),
    }
    status
}

/// Expectations for `status`, without loading the embedding model.
pub fn status_config(data_dir: &Path, embedder_id: String, dim: usize) -> StaticConfig {
    let mut cfg = WorkerConfig::from_env(data_dir.to_path_buf());
    cfg.store = MEMORY_STORE.into();
    cfg.static_config_with(embedder_id, dim)
}

pub fn status(data_dir: &Path, json: bool, sc: &StaticConfig) -> Result<(), String> {
    let db = open(data_dir)?;
    let report = db.status_report(sc).map_err(|e| format!("status: {e}"))?;
    let worker_running = matches!(graphrag_worker::lock::acquire(data_dir), Ok(None));
    let state = db.worker_state().map_err(|e| e.to_string())?;
    if json {
        let v = serde_json::json!({
            "report": report,
            "complete": report.is_complete(),
            "worker_running": worker_running,
            "worker_state": state,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let steps = [
        Step::Chunk,
        Step::Embed,
        Step::Extract,
        Step::Generalize,
        Step::Communities,
        Step::Summary,
    ];
    let cell = |c: &graphrag_core::status::StepCounts, s: Step| {
        let g = |m: &std::collections::BTreeMap<Step, usize>| *m.get(&s).unwrap_or(&0);
        format!(
            "{}/{}/{}/{}",
            g(&c.current),
            g(&c.stale),
            g(&c.missing),
            g(&c.failed)
        )
    };
    println!(
        "complete: {}   worker running: {worker_running}",
        report.is_complete()
    );
    println!(
        "{:<16} {}",
        "GROUP",
        steps
            .iter()
            .map(|s| format!("{:>15}", s.as_str()))
            .collect::<String>()
    );
    println!(
        "{:<16} {}",
        "",
        steps
            .iter()
            .map(|_| format!("{:>15}", "cur/stl/mis/fl"))
            .collect::<String>()
    );
    for (g, c) in &report.per_group {
        println!(
            "{g:<16} {}",
            steps
                .iter()
                .map(|s| format!("{:>15}", cell(c, *s)))
                .collect::<String>()
        );
    }
    println!(
        "{:<16} {}",
        "TOTAL",
        steps
            .iter()
            .map(|s| format!("{:>15}", cell(&report.total, *s)))
            .collect::<String>()
    );
    let [b0, b1, b2] = report.backlog_by_priority;
    println!("\nbacklog: recall-touched {b2}, new {b1}, backlog {b0}");
    println!(
        "pending summaries: {}   unmapped relations: {}   rejected entities: {}   type conflicts: {}",
        report.pending_summaries,
        report.unmapped_relations,
        report.rejected_entities,
        report.type_conflicts
    );
    println!(
        "classes: {} (+{} in 24h)   lifted edges admitted: {}   dormant: {} entities, {} relations",
        report.classes,
        report.new_classes_24h,
        report.lifted_admitted,
        report.dormant_entities,
        report.dormant_relations
    );
    for (k, v) in &state {
        println!("{k}: {v}");
    }
    for (id, step, err) in &report.failed_samples {
        println!("failed: chunk {id} {}: {err}", step.as_str());
    }
    Ok(())
}

/// Embed chunks whose embed step is missing or stale for this embedder.
pub fn backfill(data_dir: &Path, emb: &dyn TextEmbedder) -> Result<usize, String> {
    let db = open(data_dir)?;
    ensure_store(&db, emb.dim())?;
    let chunks = db
        .list_chunks(MEMORY_STORE)
        .map_err(|e| format!("list chunks: {e}"))?;
    let mut n = 0;
    for (i, c) in chunks.iter().enumerate() {
        if embed_chunk(&db, emb, c.id)? {
            n += 1;
        }
        if (i + 1) % 25 == 0 || i + 1 == chunks.len() {
            eprintln!("checked {}/{} chunks, embedded {n}", i + 1, chunks.len());
        }
    }
    rebuild_lexical_sidecar(data_dir, &db, MEMORY_STORE)?;
    Ok(n)
}
