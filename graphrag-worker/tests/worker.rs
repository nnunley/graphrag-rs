//! Task 15: the self-driving ingestion worker, against fakes.

mod support;

use graphrag_core::Database;
use graphrag_core::documents::{AddDocument, Origin};
use graphrag_core::fingerprints::ChunkerProfile;
use graphrag_core::status::{Step, SubjectKind};
use graphrag_worker::{ExitStatus, WorkerConfig, lock, pause_flag, run};
use rusqlite::Connection;
use support::{FakeEmbedder, FakeLlm, Mode};
use tempfile::TempDir;

const STORE: &str = "memory";

fn setup() -> (TempDir, WorkerConfig) {
    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("graphrag.db")).unwrap();
    db.create_store(STORE, 8).unwrap();
    let mut cfg = WorkerConfig::new(dir.path().to_path_buf());
    cfg.store = STORE.into();
    cfg.concurrency = 2;
    (dir, cfg)
}

fn db(cfg: &WorkerConfig) -> Database {
    Database::open(&cfg.data_dir.join("graphrag.db")).unwrap()
}

fn add(cfg: &WorkerConfig, text: &str, groups: &[&str]) {
    db(cfg)
        .add_document(&AddDocument {
            store: STORE,
            content: text,
            origin: Origin::Source,
            source: None,
            doc_key: None,
            title: None,
            groups,
            surface: "cli",
            profile: &ChunkerProfile::default_markdown(),
        })
        .unwrap();
}

fn seed(cfg: &WorkerConfig) {
    add(
        cfg,
        "# GraphRAG\nGraphRAG uses Leiden and Rust.\n## Detail\nLeiden finds Communities in Graphs.\n",
        &["papers"],
    );
    add(
        cfg,
        "# Note\nI used Leiden with Rust and Tokio today.\n# Later\nTokio and Rust again with Serde.\n",
        &["conversations"],
    );
}

fn count(cfg: &WorkerConfig, sql: &str) -> i64 {
    Connection::open(cfg.data_dir.join("graphrag.db"))
        .unwrap()
        .query_row(sql, [], |r| r.get(0))
        .unwrap()
}

fn snapshot(cfg: &WorkerConfig) -> Vec<String> {
    let conn = Connection::open(cfg.data_dir.join("graphrag.db")).unwrap();
    let mut s = conn
        .prepare(
            "SELECT h.name || ' -' || COALESCE(r.canonical_relation, r.relation) || '-> ' || t.name || ' x' ||
                    (SELECT COUNT(DISTINCT evidence_kind || evidence_id) FROM attempt_relations ar WHERE ar.relation_id = r.id)
             FROM relations r JOIN entities h ON h.id = r.head_id JOIN entities t ON t.id = r.tail_id",
        )
        .unwrap();
    let mut out: Vec<String> = s
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    let mut e = conn
        .prepare("SELECT name FROM entities WHERE kind = 'entity'")
        .unwrap();
    out.extend(
        e.query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(Result::unwrap),
    );
    out.sort();
    out
}

#[test]
fn drains_seeded_graph_to_complete() {
    let (_d, cfg) = setup();
    seed(&cfg);
    let (llm, emb) = (FakeLlm::new(), FakeEmbedder::new());
    assert_eq!(run(&cfg, &llm, &emb), ExitStatus::Done);
    let report = db(&cfg).status_report(&cfg.static_config(&emb)).unwrap();
    assert!(report.is_complete(), "{report:?}");
    assert_eq!(report.total.current.get(&Step::Communities), Some(&1));
    assert!(
        count(
            &cfg,
            "SELECT COUNT(*) FROM community_runs WHERE current = 1"
        ) == 1
    );
    assert_eq!(
        count(&cfg, "SELECT COUNT(*) FROM communities c WHERE NOT EXISTS (SELECT 1 FROM step_status s
             WHERE s.subject_kind='community' AND s.subject_id=c.id AND s.step='summary' AND s.state='current')"),
        0,
        "every community summarized or skipped"
    );
    assert!(count(&cfg, "SELECT COUNT(*) FROM chunk_embeddings") > 0);
    assert!(count(&cfg, "SELECT COUNT(*) FROM attempts") > 0);
}

#[test]
fn second_run_does_zero_work() {
    let (_d, cfg) = setup();
    seed(&cfg);
    let (llm, emb) = (FakeLlm::new(), FakeEmbedder::new());
    assert_eq!(run(&cfg, &llm, &emb), ExitStatus::Done);
    let (llm2, emb2) = (FakeLlm::new(), FakeEmbedder::new());
    assert_eq!(run(&cfg, &llm2, &emb2), ExitStatus::Done);
    assert_eq!(llm2.calls(), 0);
    assert_eq!(emb2.calls(), 0);
}

#[test]
fn kill_and_resume_matches_uninterrupted() {
    let (_a, full) = setup();
    seed(&full);
    assert_eq!(
        run(&full, &FakeLlm::new(), &FakeEmbedder::new()),
        ExitStatus::Done
    );

    let (_b, mut part) = setup();
    seed(&part);
    part.max_items = Some(3);
    assert_eq!(
        run(&part, &FakeLlm::new(), &FakeEmbedder::new()),
        ExitStatus::Done
    );
    part.max_items = None;
    assert_eq!(
        run(&part, &FakeLlm::new(), &FakeEmbedder::new()),
        ExitStatus::Done
    );

    assert_eq!(snapshot(&full), snapshot(&part));
    assert_eq!(
        count(&full, "SELECT COUNT(*) FROM attempts"),
        count(&part, "SELECT COUNT(*) FROM attempts")
    );
}

#[test]
fn transport_failures_back_off_then_outage() {
    let (_d, mut cfg) = setup();
    for i in 0..6 {
        add(&cfg, &format!("Note {i} about Alpha{i} and Beta{i}."), &[]);
    }
    cfg.outage_threshold = 5;
    let llm = FakeLlm::new();
    llm.set_mode(Mode::Transport);
    assert_eq!(run(&cfg, &llm, &FakeEmbedder::new()), ExitStatus::Outage);
    assert_eq!(ExitStatus::Outage.code(), 75);
    assert!(
        count(
            &cfg,
            "SELECT COUNT(*) FROM step_status WHERE step='extract' AND state='retry'
                     AND next_try_at > strftime('%Y-%m-%dT%H:%M:%fZ','now')"
        ) >= 5
    );
    assert_eq!(
        count(&cfg, "SELECT COUNT(*) FROM attempts"),
        0,
        "transport failures write no attempt"
    );
    assert!(
        count(
            &cfg,
            "SELECT COUNT(*) FROM worker_state WHERE key = 'last_llm_error'"
        ) == 1
    );
}

#[test]
fn malformed_three_times_is_failed() {
    let (_d, cfg) = setup();
    add(&cfg, "Alpha meets Beta.", &[]);
    let llm = FakeLlm::new();
    llm.set_mode(Mode::Malformed);
    assert_eq!(run(&cfg, &llm, &FakeEmbedder::new()), ExitStatus::Done);
    assert_eq!(
        count(
            &cfg,
            "SELECT COUNT(*) FROM step_status WHERE step='extract' AND state='failed'"
        ),
        1
    );
    assert_eq!(llm.calls(), 3, "three malformed tries, then failed");
    let report = db(&cfg)
        .status_report(&cfg.static_config(&FakeEmbedder::new()))
        .unwrap();
    assert_eq!(report.total.failed.get(&Step::Extract), Some(&1));
}

#[test]
fn model_not_served_exits_69_writes_nothing() {
    let (_d, cfg) = setup();
    seed(&cfg);
    let mut llm = FakeLlm::new();
    llm.models = vec!["some-other-model".into()];
    let emb = FakeEmbedder::new();
    assert_eq!(run(&cfg, &llm, &emb), ExitStatus::Unavailable);
    assert_eq!(ExitStatus::Unavailable.code(), 69);
    assert_eq!(emb.calls(), 0);
    assert_eq!(llm.calls(), 0);
    assert_eq!(count(&cfg, "SELECT COUNT(*) FROM chunk_embeddings"), 0);
    assert_eq!(count(&cfg, "SELECT COUNT(*) FROM attempts"), 0);
}

#[test]
fn reextract_guard_requires_flag() {
    let (_d, mut cfg) = setup();
    seed(&cfg);
    assert_eq!(
        run(&cfg, &FakeLlm::new(), &FakeEmbedder::new()),
        ExitStatus::Done
    );
    cfg.prompt_version = "paper-json-v9".into();
    let llm = FakeLlm::new();
    assert_eq!(run(&cfg, &llm, &FakeEmbedder::new()), ExitStatus::Usage);
    assert_eq!(ExitStatus::Usage.code(), 64);
    assert_eq!(llm.calls(), 0);
    cfg.accept_reextract = true;
    let llm = FakeLlm::new();
    assert_eq!(run(&cfg, &llm, &FakeEmbedder::new()), ExitStatus::Done);
    assert!(llm.calls() > 0);
}

#[test]
fn second_worker_exits_already_running() {
    let (_d, cfg) = setup();
    seed(&cfg);
    let _held = lock::acquire(&cfg.data_dir).unwrap().expect("first lock");
    let llm = FakeLlm::new();
    assert_eq!(
        run(&cfg, &llm, &FakeEmbedder::new()),
        ExitStatus::AlreadyRunning
    );
    assert_eq!(ExitStatus::AlreadyRunning.code(), 0);
    assert_eq!(llm.calls(), 0);
}

#[test]
fn escalated_chunk_processed_first() {
    let (_d, mut cfg) = setup();
    for i in 0..3 {
        add(&cfg, &format!("Plain note number {i} about Topic{i}."), &[]);
    }
    let last: i64 = count(&cfg, "SELECT MAX(id) FROM chunks");
    db(&cfg).escalate_chunks(&[last]).unwrap();
    cfg.concurrency = 1;
    cfg.max_items = Some(2);
    let llm = FakeLlm::new();
    run(&cfg, &llm, &FakeEmbedder::new());
    let first = llm.extracted.lock().unwrap().first().cloned().unwrap();
    assert!(
        first.contains("Topic2"),
        "escalated chunk extracted first, got {first:?}"
    );
}

#[test]
fn pause_flag_stops_worker() {
    let (_d, cfg) = setup();
    seed(&cfg);
    std::fs::write(pause_flag(&cfg.data_dir), "").unwrap();
    let (llm, emb) = (FakeLlm::new(), FakeEmbedder::new());
    assert_eq!(run(&cfg, &llm, &emb), ExitStatus::Done);
    assert_eq!(llm.calls() + emb.calls(), 0);
}

#[test]
fn no_automated_path_removes_documents() {
    let (_d, cfg) = setup();
    // Old, node-less notes (the v1-converted shape) plus a new one.
    {
        let d = db(&cfg);
        for i in 0..4 {
            d.add_chunk(
                STORE,
                &format!("Old note {i}: Alpha{i} met Beta{i} long ago.\n"),
                Some("brain"),
                None,
            )
            .unwrap();
        }
    }
    add(&cfg, "# New\nGamma and Delta.", &["conversations"]);
    let contents = |cfg: &WorkerConfig| {
        let conn = Connection::open(cfg.data_dir.join("graphrag.db")).unwrap();
        let mut s = conn
            .prepare("SELECT id, content FROM documents ORDER BY id")
            .unwrap();
        s.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
    };
    let before = contents(&cfg);
    // Age every use so salience is as low as it gets.
    Connection::open(cfg.data_dir.join("graphrag.db"))
        .unwrap()
        .execute("UPDATE uses SET at = '2020-01-01T00:00:00.000Z'", [])
        .unwrap();
    assert_eq!(
        run(&cfg, &FakeLlm::new(), &FakeEmbedder::new()),
        ExitStatus::Done
    );
    assert_eq!(contents(&cfg), before, "document text and count unchanged");
    assert_eq!(
        count(&cfg, "SELECT COUNT(*) FROM documents d WHERE NOT EXISTS (
             SELECT 1 FROM chunks c JOIN chunk_embeddings e ON e.chunk_id = c.id
             JOIN step_status s ON s.subject_kind='chunk' AND s.subject_id=c.id AND s.step='embed' AND s.state='current'
             WHERE c.document_id = d.id)"),
        0,
        "every document keeps at least one current, embedded chunk"
    );
    let _ = SubjectKind::Chunk;
}
