//! Schema v2: one content-addressed memory graph.
//!
//! Chunks store no text. A chunk is a byte span `[span_start, span_end)` into
//! an immutable `documents.content`, plus metadata; readers go through
//! `chunk_view`, which slices the document. Opening a v1 database converts
//! each old chunk into a document holding its text (identical texts share
//! one document) and a chunk spanning all of it that keeps its id.

use crate::error::GraphRagError;
use crate::fingerprints::document_fp;
use crate::token::fingerprint_text;
use rusqlite::{Connection, params};

pub const SCHEMA_VERSION: i64 = 2;

const DOCUMENTS_DDL: &str = r#"
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    content_fp TEXT NOT NULL UNIQUE,
    doc_key TEXT,
    origin TEXT NOT NULL DEFAULT 'source' CHECK(origin IN ('source','compiled')),
    source TEXT,
    title TEXT,
    node_id INTEGER REFERENCES entities(id) ON DELETE SET NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_documents_doc_key ON documents(doc_key);
CREATE TRIGGER IF NOT EXISTS documents_immutable
    BEFORE UPDATE OF content ON documents
    BEGIN SELECT RAISE(ABORT, 'documents are immutable'); END;

CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    heading TEXT NOT NULL,
    level INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    content_fp TEXT NOT NULL,
    node_id INTEGER REFERENCES entities(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_sections_document ON sections(document_id);
"#;

const CHUNKS_V2_DDL: &str = r#"
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    store TEXT NOT NULL REFERENCES stores(name) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    section_id INTEGER REFERENCES sections(id) ON DELETE SET NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL CHECK(span_end >= span_start),
    content_fp TEXT NOT NULL,
    source TEXT,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"#;

const CHUNK_INDEXES_AND_VIEW: &str = r#"
CREATE INDEX IF NOT EXISTS idx_chunks_store ON chunks(store);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_fp ON chunks(content_fp);
-- SQLite's substr counts characters on TEXT; slicing the BLOB form counts
-- bytes, which is what the span offsets are.
CREATE VIEW IF NOT EXISTS chunk_view AS
SELECT c.id, c.store,
       CAST(substr(CAST(d.content AS BLOB), c.span_start + 1, c.span_end - c.span_start) AS TEXT)
           AS content,
       c.source, c.metadata, c.created_at,
       c.document_id, c.section_id, c.span_start, c.span_end, c.content_fp
FROM chunks c JOIN documents d ON d.id = c.document_id;
"#;

const GRAPH_DDL: &str = r#"
CREATE TABLE IF NOT EXISTS document_adds (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source TEXT,
    surface TEXT NOT NULL,
    added_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_document_adds_document ON document_adds(document_id);

CREATE TABLE IF NOT EXISTS assertions (
    id INTEGER PRIMARY KEY,
    head_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    tail_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    asserter TEXT NOT NULL,
    reason TEXT,
    context TEXT,
    asserted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    retracted_at TEXT,
    retracted_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_assertions_head ON assertions(head_id);
CREATE INDEX IF NOT EXISTS idx_assertions_tail ON assertions(tail_id);

CREATE TABLE IF NOT EXISTS step_status (
    subject_kind TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    step TEXT NOT NULL,
    fp TEXT,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    next_try_at TEXT,
    priority INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (subject_kind, subject_id, step)
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_fp TEXT PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    profile TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK(outcome IN ('ok','empty')),
    response_sha256 TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    unmapped_relations INTEGER NOT NULL DEFAULT 0,
    rejected_entities INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_attempts_chunk ON attempts(chunk_id);

CREATE TABLE IF NOT EXISTS attempt_entities (
    attempt_fp TEXT NOT NULL REFERENCES attempts(attempt_fp) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_type TEXT,
    description TEXT,
    PRIMARY KEY (attempt_fp, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_attempt_entities_entity ON attempt_entities(entity_id);

CREATE TABLE IF NOT EXISTS attempt_relations (
    attempt_fp TEXT NOT NULL REFERENCES attempts(attempt_fp) ON DELETE CASCADE,
    relation_id INTEGER NOT NULL REFERENCES relations(id) ON DELETE CASCADE,
    raw_relation TEXT NOT NULL,
    description TEXT,
    strength INTEGER,
    evidence_kind TEXT NOT NULL CHECK(evidence_kind IN ('section','document')),
    evidence_id INTEGER NOT NULL,
    PRIMARY KEY (attempt_fp, relation_id, raw_relation)
);
CREATE INDEX IF NOT EXISTS idx_attempt_relations_relation ON attempt_relations(relation_id);

CREATE TABLE IF NOT EXISTS community_runs (
    run_id INTEGER PRIMARY KEY,
    store TEXT NOT NULL,
    graph_fp TEXT NOT NULL,
    config_json TEXT NOT NULL,
    seed INTEGER NOT NULL,
    modularity REAL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    current INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS summaries (
    community_fp TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS uses (
    id INTEGER PRIMARY KEY,
    item_kind TEXT NOT NULL,
    item_id INTEGER NOT NULL,
    at TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('mention','recall','assert')),
    source_id INTEGER,
    weight REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_uses_item ON uses(item_kind, item_id);

CREATE TABLE IF NOT EXISTS relation_genls (
    relation TEXT PRIMARY KEY,
    broader TEXT NOT NULL,
    origin TEXT NOT NULL CHECK(origin IN ('seed','llm')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS class_aliases (
    alias TEXT PRIMARY KEY,
    class_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS lifted_edges (
    id INTEGER PRIMARY KEY,
    relation TEXT NOT NULL,
    head_class INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    tail_class INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    pairs INTEGER NOT NULL,
    g2 REAL NOT NULL,
    admitted INTEGER NOT NULL DEFAULT 0,
    fp TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(relation, head_class, tail_class)
);

CREATE TABLE IF NOT EXISTS lifted_from (
    lifted_id INTEGER NOT NULL REFERENCES lifted_edges(id) ON DELETE CASCADE,
    relation_id INTEGER NOT NULL REFERENCES relations(id) ON DELETE CASCADE,
    PRIMARY KEY (lifted_id, relation_id)
);

CREATE TABLE IF NOT EXISTS worker_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"#;

/// Columns added to existing tables: (table, column, declaration).
const ADDED_COLUMNS: &[(&str, &str, &str)] = &[
    (
        "communities",
        "run_id",
        "INTEGER REFERENCES community_runs(run_id) ON DELETE CASCADE",
    ),
    ("communities", "fp", "TEXT"),
    ("entities", "kind", "TEXT NOT NULL DEFAULT 'entity'"),
    ("entities", "salience", "REAL"),
    (
        "entities",
        "edge_status",
        "TEXT NOT NULL DEFAULT 'active' CHECK(edge_status IN ('active','dormant','implied'))",
    ),
    ("relations", "salience", "REAL"),
    (
        "relations",
        "edge_status",
        "TEXT NOT NULL DEFAULT 'active' CHECK(edge_status IN ('active','dormant','implied'))",
    ),
];

fn has_column(conn: &Connection, table: &str, column: &str) -> Result<bool, GraphRagError> {
    Ok(conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info(?1) WHERE name = ?2",
        params![table, column],
        |row| row.get(0),
    )?)
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool, GraphRagError> {
    Ok(conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type = 'table' AND name = ?1",
        params![table],
        |row| row.get(0),
    )?)
}

/// Apply schema v2. Idempotent; sets `PRAGMA user_version = 2`.
pub(crate) fn apply_v2(conn: &Connection) -> Result<(), GraphRagError> {
    conn.execute_batch(DOCUMENTS_DDL)?;
    if table_exists(conn, "chunks")? && has_column(conn, "chunks", "content")? {
        convert_v1_chunks(conn)?;
    } else {
        conn.execute_batch(CHUNKS_V2_DDL)?;
    }
    conn.execute_batch(CHUNK_INDEXES_AND_VIEW)?;
    conn.execute_batch(GRAPH_DDL)?;
    for (table, column, decl) in ADDED_COLUMNS {
        if table_exists(conn, table)? && !has_column(conn, table, column)? {
            conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"))?;
        }
    }
    conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))?;
    Ok(())
}

/// `(id, store, content, source, metadata, created_at)` of a v1 chunk row.
type V1Chunk = (
    i64,
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
);

/// Rebuild `chunks` without `content`: each old chunk's text moves into a
/// content-addressed document and the chunk keeps its id as a full span.
fn convert_v1_chunks(conn: &Connection) -> Result<(), GraphRagError> {
    // Rebuilding a referenced table needs foreign keys off, and that pragma
    // is a no-op inside a transaction.
    conn.execute_batch("PRAGMA foreign_keys = OFF")?;
    let violations_before = foreign_key_violations(conn)?;
    let result = (|| -> Result<(), GraphRagError> {
        let tx = conn.unchecked_transaction()?;
        tx.execute_batch(&CHUNKS_V2_DDL.replace("TABLE IF NOT EXISTS chunks", "TABLE chunks_v2"))?;
        let rows: Vec<V1Chunk> = {
            let mut stmt = tx.prepare(
                "SELECT id, store, content, source, metadata, created_at FROM chunks ORDER BY id",
            )?;
            stmt.query_map([], |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                ))
            })?
            .collect::<Result<_, _>>()?
        };
        for (id, store, content, source, metadata, created_at) in rows {
            let fp = document_fp(&content);
            tx.execute(
                "INSERT OR IGNORE INTO documents (content, content_fp, source, created_at)
                 VALUES (?1, ?2, ?3, COALESCE(?4, datetime('now')))",
                params![content, fp, source, created_at],
            )?;
            let document_id: i64 = tx.query_row(
                "SELECT id FROM documents WHERE content_fp = ?1",
                params![fp],
                |r| r.get(0),
            )?;
            tx.execute(
                "INSERT INTO chunks_v2
                     (id, store, document_id, span_start, span_end, content_fp, source, metadata, created_at)
                 VALUES (?1, ?2, ?3, 0, ?4, ?5, ?6, ?7, COALESCE(?8, datetime('now')))",
                params![
                    id,
                    store,
                    document_id,
                    content.len() as i64,
                    fingerprint_text(&content),
                    source,
                    metadata,
                    created_at
                ],
            )?;
        }
        tx.execute_batch(
            "DROP INDEX IF EXISTS idx_chunks_store;
             DROP TABLE chunks;
             ALTER TABLE chunks_v2 RENAME TO chunks;",
        )?;
        // Old data may already hold dangling rows (an orphaned embedding);
        // the conversion must not add any.
        let violations_after = foreign_key_violations(&tx)?;
        if violations_after > violations_before {
            return Err(GraphRagError::Other(format!(
                "v2 chunk conversion would add {} foreign-key violations",
                violations_after - violations_before
            )));
        }
        tx.commit()?;
        Ok(())
    })();
    conn.execute_batch("PRAGMA foreign_keys = ON")?;
    result
}

fn foreign_key_violations(conn: &Connection) -> Result<i64, GraphRagError> {
    let mut stmt = conn.prepare("PRAGMA foreign_key_check")?;
    let mut rows = stmt.query([])?;
    let mut n = 0;
    while rows.next()?.is_some() {
        n += 1;
    }
    Ok(n)
}
