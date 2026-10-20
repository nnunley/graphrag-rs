//! Task 1: connection pragmas, the v2 schema gate, and chunks as spans into
//! immutable documents.

use graphrag_core::Database;
use rusqlite::Connection;
use std::sync::{Arc, Barrier};
use std::time::Duration;
use tempfile::TempDir;

const V2_TABLES: &[&str] = &[
    "documents",
    "document_adds",
    "sections",
    "assertions",
    "step_status",
    "attempts",
    "attempt_entities",
    "attempt_relations",
    "community_runs",
    "summaries",
    "uses",
    "relation_genls",
    "class_aliases",
    "lifted_edges",
    "lifted_from",
    "worker_state",
];

/// The v1 DDL as `init_schema` created it before v2 (the tables these tests touch).
const V1_DDL: &str = r#"
CREATE TABLE stores (name TEXT PRIMARY KEY, dim INTEGER NOT NULL, created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    store TEXT NOT NULL REFERENCES stores(name) ON DELETE CASCADE,
    content TEXT NOT NULL,
    source TEXT,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_chunks_store ON chunks(store);
CREATE TABLE chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL
);
"#;

fn open(dir: &TempDir) -> Database {
    Database::open(&dir.path().join("t.db")).unwrap()
}

#[test]
fn open_sets_wal_and_busy_timeout() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    let (journal, busy, fk) = db.connection_pragmas().unwrap();
    assert_eq!(journal, "wal");
    assert_eq!(busy, 5000);
    assert_eq!(fk, 1);
}

#[test]
fn open_twice_is_idempotent_at_version_2() {
    let dir = TempDir::new().unwrap();
    {
        let db = open(&dir);
        assert_eq!(db.schema_version().unwrap(), 2);
    }
    let db = open(&dir);
    assert_eq!(db.schema_version().unwrap(), 2);
    let conn = Connection::open(dir.path().join("t.db")).unwrap();
    for t in V2_TABLES {
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                [t],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "table {t} missing");
    }
}

#[test]
fn v1_database_upgrades_in_place() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("t.db");
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(V1_DDL).unwrap();
        conn.execute_batch(
            "INSERT INTO stores (name, dim) VALUES ('s', 3);
             INSERT INTO chunks (id, store, content, source) VALUES (7, 's', 'same text', 'a');
             INSERT INTO chunks (id, store, content, source) VALUES (8, 's', 'same text', 'b');
             INSERT INTO chunks (id, store, content, source) VALUES (9, 's', 'é漢🙂 tail', 'c');
             INSERT INTO chunk_embeddings VALUES (9, x'0000803f');",
        )
        .unwrap();
    }
    let db = Database::open(&path).unwrap();
    assert_eq!(db.schema_version().unwrap(), 2);
    drop(db);

    let conn = Connection::open(&path).unwrap();
    let has_content: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('chunks') WHERE name='content'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(has_content, 0, "chunks must not store text");
    let docs: i64 = conn
        .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
        .unwrap();
    assert_eq!(docs, 2, "identical texts share one document");
    let emb: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id = 9",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(emb, 1, "embedding stays attached");
    drop(conn);

    let db = Database::open(&path).unwrap();
    assert_eq!(db.chunk_text(7).unwrap(), "same text");
    assert_eq!(db.chunk_text(8).unwrap(), "same text");
    assert_eq!(db.chunk_text(9).unwrap(), "é漢🙂 tail");
    let chunks = db.get_chunks_by_ids(&[7, 8, 9]).unwrap();
    assert_eq!(chunks.len(), 3);
    assert!(
        chunks
            .iter()
            .any(|c| c.id == 9 && c.content == "é漢🙂 tail")
    );
}

#[test]
fn chunk_text_slices_document_bytes() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.create_store("s", 3).unwrap();
    let text = "aé漢🙂b";
    let doc = db.add_chunk("s", text, None, None).unwrap();
    let conn = Connection::open(dir.path().join("t.db")).unwrap();
    let doc_id: i64 = conn
        .query_row("SELECT document_id FROM chunks WHERE id = ?1", [doc], |r| {
            r.get(0)
        })
        .unwrap();
    let bounds: Vec<usize> = (0..=text.len())
        .filter(|i| text.is_char_boundary(*i))
        .collect();
    for (i, &start) in bounds.iter().enumerate() {
        for &end in &bounds[i + 1..] {
            conn.execute(
                "INSERT INTO chunks (store, document_id, span_start, span_end, content_fp)
                 VALUES ('s', ?1, ?2, ?3, 'sha256:test')",
                rusqlite::params![doc_id, start as i64, end as i64],
            )
            .unwrap();
            let id = conn.last_insert_rowid();
            assert_eq!(
                db.chunk_text(id).unwrap(),
                &text[start..end],
                "{start}..{end}"
            );
            let c = &db.get_chunks_by_ids(&[id]).unwrap()[0];
            assert_eq!(c.content, &text[start..end]);
        }
    }
}

#[test]
fn document_text_is_immutable() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.create_store("s", 3).unwrap();
    db.add_chunk("s", "original", None, None).unwrap();
    drop(db);
    let conn = Connection::open(dir.path().join("t.db")).unwrap();
    let err = conn
        .execute("UPDATE documents SET content = 'x'", [])
        .unwrap_err();
    assert!(err.to_string().contains("documents are immutable"), "{err}");
}

#[test]
fn add_chunk_stores_text_once_in_a_document() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir);
    db.create_store("s", 3).unwrap();
    let a = db.add_chunk("s", "hello", Some("x"), None).unwrap();
    let b = db.add_chunk("s", "hello", Some("y"), None).unwrap();
    assert_ne!(a, b);
    assert_eq!(db.chunk_text(a).unwrap(), "hello");
    drop(db);
    let conn = Connection::open(dir.path().join("t.db")).unwrap();
    let docs: i64 = conn
        .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
        .unwrap();
    assert_eq!(docs, 1);
}

#[test]
fn concurrent_writer_waits_instead_of_failing() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("t.db");
    {
        let db = Database::open(&path).unwrap();
        db.create_store("s", 3).unwrap();
    }
    let barrier = Arc::new(Barrier::new(2));
    let (p1, b1) = (path.clone(), barrier.clone());
    let holder = std::thread::spawn(move || {
        let conn = Connection::open(&p1).unwrap();
        conn.execute_batch("PRAGMA busy_timeout=5000; BEGIN IMMEDIATE;")
            .unwrap();
        conn.execute("INSERT INTO stores (name, dim) VALUES ('held', 3)", [])
            .unwrap();
        b1.wait();
        std::thread::sleep(Duration::from_millis(300));
        conn.execute_batch("COMMIT;").unwrap();
    });
    barrier.wait();
    let db = Database::open(&path).unwrap();
    db.add_chunk(
        "s",
        "written while another writer held the lock",
        None,
        None,
    )
    .unwrap();
    holder.join().unwrap();
    assert_eq!(db.list_chunks("s").unwrap().len(), 1);
    assert!(db.get_store("held").is_ok());
}
