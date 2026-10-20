//! Shared fixtures for the one-graph tests: a fresh DB, documents, and
//! committed extraction attempts.
#![allow(dead_code)]

use graphrag_core::Database;
use graphrag_core::attempts::{AttemptOutcome, AttemptRecord, ExtractedEntity, ExtractedRelation};
use graphrag_core::documents::{AddDocument, AddOutcome, Origin};
use graphrag_core::fingerprints::{ChunkerProfile, extract_fp};
use graphrag_core::paper::ExtractionAttemptIdentityParts;
use rusqlite::Connection;
use tempfile::TempDir;

pub const STORE: &str = "memory";

pub struct Fx {
    pub dir: TempDir,
    pub db: Database,
}

pub fn fx() -> Fx {
    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("t.db")).unwrap();
    db.create_store(STORE, 3).unwrap();
    Fx { dir, db }
}

pub fn ent(name: &str, ty: &str) -> ExtractedEntity {
    ExtractedEntity {
        name: name.into(),
        entity_type: ty.into(),
        description: format!("{name} desc"),
        kind: None,
        is_class: false,
        broader: None,
    }
}

pub fn ent_kind(name: &str, ty: &str, kind: &str) -> ExtractedEntity {
    ExtractedEntity {
        kind: Some(kind.into()),
        ..ent(name, ty)
    }
}

pub fn rel(s: &str, t: &str, r: &str) -> ExtractedRelation {
    ExtractedRelation {
        source: s.into(),
        target: t.into(),
        relation: r.into(),
        description: format!("{s} {r} {t}"),
        strength: Some(5),
    }
}

/// `n` sections, each a one-line paragraph.
pub fn sections(n: usize, tag: &str) -> String {
    (0..n)
        .map(|i| format!("# {tag} {i}\nline {i} of {tag}\n"))
        .collect()
}

impl Fx {
    pub fn doc(&self, text: &str, groups: &[&str]) -> (i64, Vec<i64>) {
        let p = ChunkerProfile::default_markdown();
        let out = self
            .db
            .add_document(&AddDocument {
                store: STORE,
                content: text,
                origin: Origin::Source,
                source: None,
                doc_key: None,
                title: None,
                groups,
                surface: "cli",
                profile: &p,
            })
            .unwrap();
        let AddOutcome::Created { document_id } = out else {
            panic!("{out:?}")
        };
        (document_id, self.db.document_chunks(document_id).unwrap())
    }

    pub fn commit(
        &self,
        chunk: i64,
        raw: &str,
        ents: Vec<ExtractedEntity>,
        rels: Vec<ExtractedRelation>,
    ) {
        let text = self.db.chunk_text(chunk).unwrap();
        let fp = extract_fp(&ExtractionAttemptIdentityParts {
            store: STORE,
            chunk_id: chunk,
            chunk_content: &text,
            model: "m",
            extraction_profile: "paper-json",
            prompt_version: "v1",
            response_schema_version: "graph-v1",
            glean_max_rounds: 0,
        })
        .unwrap();
        let outcome = if ents.is_empty() && rels.is_empty() {
            AttemptOutcome::Empty
        } else {
            AttemptOutcome::Ok {
                entities: ents,
                relations: rels,
            }
        };
        self.db
            .commit_attempt(&AttemptRecord {
                chunk_id: chunk,
                attempt_fp: &fp,
                model: "m",
                profile: "paper-json",
                prompt_version: "v1",
                schema_version: "graph-v1",
                glean_max_rounds: 0,
                raw_response: raw,
                outcome,
                rejected_entities: 0,
            })
            .unwrap();
    }

    pub fn id(&self, name: &str) -> i64 {
        self.db.get_entity_by_name(STORE, name).unwrap().id
    }

    pub fn raw(&self) -> Connection {
        Connection::open(self.dir.path().join("t.db")).unwrap()
    }

    pub fn count(&self, sql: &str) -> i64 {
        self.raw().query_row(sql, [], |r| r.get(0)).unwrap()
    }
}
