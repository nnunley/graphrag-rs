//! Task 9: extraction attempts with supersession and evidence support.

use graphrag_core::Database;
use graphrag_core::assertions::Asserter;
use graphrag_core::attempts::{AttemptOutcome, AttemptRecord, ExtractedEntity, ExtractedRelation};
use graphrag_core::documents::{AddDocument, AddOutcome, Origin};
use graphrag_core::fingerprints::{ChunkerProfile, extract_fp};
use graphrag_core::paper::ExtractionAttemptIdentityParts;
use rusqlite::Connection;
use tempfile::TempDir;

struct Fx {
    dir: TempDir,
    db: Database,
}

fn fx() -> Fx {
    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("t.db")).unwrap();
    db.create_store("memory", 3).unwrap();
    Fx { dir, db }
}

fn ent(name: &str, ty: &str) -> ExtractedEntity {
    ExtractedEntity {
        name: name.into(),
        entity_type: ty.into(),
        description: format!("{name} desc"),
        kind: None,
        is_class: false,
        broader: None,
    }
}

fn rel(s: &str, t: &str, r: &str) -> ExtractedRelation {
    ExtractedRelation {
        source: s.into(),
        target: t.into(),
        relation: r.into(),
        description: format!("{s} {r} {t}"),
        strength: Some(5),
    }
}

impl Fx {
    fn doc(&self, text: &str) -> Vec<i64> {
        let p = ChunkerProfile::default_markdown();
        let out = self
            .db
            .add_document(&AddDocument {
                store: "memory",
                content: text,
                origin: Origin::Source,
                source: None,
                doc_key: None,
                title: None,
                groups: &[],
                surface: "cli",
                profile: &p,
            })
            .unwrap();
        let AddOutcome::Created { document_id } = out else {
            panic!("{out:?}")
        };
        self.db.document_chunks(document_id).unwrap()
    }
    fn fp(&self, chunk: i64, prompt: &str) -> String {
        let text = self.db.chunk_text(chunk).unwrap();
        extract_fp(&ExtractionAttemptIdentityParts {
            store: "memory",
            chunk_id: chunk,
            chunk_content: &text,
            model: "m",
            extraction_profile: "paper-json",
            prompt_version: prompt,
            response_schema_version: "graph-v1",
            glean_max_rounds: 0,
        })
        .unwrap()
    }
    fn commit(
        &self,
        chunk: i64,
        prompt: &str,
        raw: &str,
        outcome: AttemptOutcome,
    ) -> graphrag_core::attempts::CommitReport {
        let fp = self.fp(chunk, prompt);
        self.db
            .commit_attempt(&AttemptRecord {
                chunk_id: chunk,
                attempt_fp: &fp,
                model: "m",
                profile: "paper-json",
                prompt_version: prompt,
                schema_version: "graph-v1",
                glean_max_rounds: 0,
                raw_response: raw,
                outcome,
                rejected_entities: 0,
            })
            .unwrap()
    }
    fn ok(&self, chunk: i64, raw: &str, ents: Vec<ExtractedEntity>, rels: Vec<ExtractedRelation>) {
        self.commit(
            chunk,
            "v1",
            raw,
            AttemptOutcome::Ok {
                entities: ents,
                relations: rels,
            },
        );
    }
    fn relation_id(&self, head: &str, rel: &str, tail: &str) -> Option<i64> {
        let h = self.db.get_entity_by_name("memory", head).ok()?.id;
        let t = self.db.get_entity_by_name("memory", tail).ok()?.id;
        self.db
            .get_relations_for_entity(h)
            .unwrap()
            .into_iter()
            .find(|r| {
                r.head_id == h
                    && r.tail_id == t
                    && r.canonical_relation.as_deref().unwrap_or(&r.relation) == rel
            })
            .map(|r| r.id)
    }
    fn count(&self, sql: &str) -> i64 {
        Connection::open(self.dir.path().join("t.db"))
            .unwrap()
            .query_row(sql, [], |r| r.get(0))
            .unwrap()
    }
}

fn two_sections() -> String {
    "# One\nAlice uses GraphRAG.\n# Two\nAlice uses GraphRAG again.\n".into()
}

#[test]
fn two_sections_one_relation_support_two() {
    let f = fx();
    let c = f.doc(&two_sections());
    assert_eq!(c.len(), 2);
    for (i, chunk) in c.iter().enumerate() {
        f.ok(
            *chunk,
            &format!("r{i}"),
            vec![ent("Alice", "person"), ent("GraphRAG", "software")],
            vec![rel("Alice", "GraphRAG", "uses")],
        );
    }
    let r = f.relation_id("Alice", "uses", "GraphRAG").unwrap();
    assert_eq!(f.db.relation_support(r).unwrap(), 2);
}

#[test]
fn overlapping_chunks_in_one_section_support_one() {
    let f = fx();
    let long: String = (0..700).map(|i| format!("w{i} ")).collect();
    let c = f.doc(&format!("# Only\n{long}\n"));
    assert!(c.len() >= 2);
    for (i, chunk) in c.iter().take(2).enumerate() {
        f.ok(
            *chunk,
            &format!("r{i}"),
            vec![ent("Alice", "person"), ent("GraphRAG", "software")],
            vec![rel("Alice", "GraphRAG", "uses")],
        );
    }
    let r = f.relation_id("Alice", "uses", "GraphRAG").unwrap();
    assert_eq!(f.db.relation_support(r).unwrap(), 1);
}

#[test]
fn raw_relation_kept_beside_canonical() {
    let f = fx();
    let c = f.doc("Alice utilizes GraphRAG.");
    f.ok(
        c[0],
        "r",
        vec![ent("Alice", "person"), ent("GraphRAG", "software")],
        vec![rel("Alice", "GraphRAG", "utilizes")],
    );
    assert!(f.relation_id("Alice", "uses", "GraphRAG").is_some());
    assert_eq!(
        f.count("SELECT COUNT(*) FROM attempt_relations WHERE raw_relation = 'utilizes'"),
        1
    );
}

#[test]
fn synonym_relations_share_support() {
    let f = fx();
    let c = f.doc(&two_sections());
    f.ok(
        c[0],
        "a",
        vec![ent("Alice", "person"), ent("GraphRAG", "software")],
        vec![rel("Alice", "GraphRAG", "uses")],
    );
    f.ok(
        c[1],
        "b",
        vec![ent("Alice", "person"), ent("GraphRAG", "software")],
        vec![rel("Alice", "GraphRAG", "leverages")],
    );
    let r = f.relation_id("Alice", "uses", "GraphRAG").unwrap();
    assert_eq!(f.db.relation_support(r).unwrap(), 2);
    assert_eq!(f.count("SELECT COUNT(*) FROM relations"), 1);
}

#[test]
fn identical_replay_is_noop() {
    let f = fx();
    let c = f.doc("Alice uses GraphRAG.");
    let out = || AttemptOutcome::Ok {
        entities: vec![ent("Alice", "person"), ent("GraphRAG", "software")],
        relations: vec![rel("Alice", "GraphRAG", "uses")],
    };
    let first = f.commit(c[0], "v1", "same raw", out());
    assert!(!first.replay);
    let uses_before = f.count("SELECT COUNT(*) FROM uses");
    let second = f.commit(c[0], "v1", "same raw", out());
    assert!(second.replay);
    assert_eq!(f.count("SELECT COUNT(*) FROM attempts"), 1);
    assert_eq!(f.count("SELECT COUNT(*) FROM uses"), uses_before);
}

#[test]
fn superseding_removes_old_support() {
    let f = fx();
    let c = f.doc("Alice uses GraphRAG.");
    f.ok(
        c[0],
        "r1",
        vec![ent("Alice", "person"), ent("GraphRAG", "software")],
        vec![rel("Alice", "GraphRAG", "uses")],
    );
    let r = f.relation_id("Alice", "uses", "GraphRAG").unwrap();
    let rep = f.commit(
        c[0],
        "v2",
        "r2",
        AttemptOutcome::Ok {
            entities: vec![ent("Alice", "person"), ent("Rust", "software")],
            relations: vec![rel("Alice", "Rust", "uses")],
        },
    );
    assert!(rep.superseded.is_some());
    assert_eq!(f.count("SELECT COUNT(*) FROM attempts"), 1);
    assert!(
        f.relation_id("Alice", "uses", "GraphRAG").is_none(),
        "relation {r} GC'd"
    );
}

#[test]
fn zero_support_relation_is_gc_ed() {
    let f = fx();
    let c = f.doc("Alice uses GraphRAG.");
    f.ok(
        c[0],
        "r1",
        vec![ent("Alice", "person"), ent("GraphRAG", "software")],
        vec![rel("Alice", "GraphRAG", "uses")],
    );
    let rep = f.commit(c[0], "v2", "{}", AttemptOutcome::Empty);
    assert_eq!(rep.gc_relations, 1);
    assert_eq!(rep.gc_entities, 2);
    assert_eq!(f.count("SELECT COUNT(*) FROM relations"), 0);
    assert_eq!(
        f.count("SELECT COUNT(*) FROM entities WHERE kind = 'entity'"),
        0
    );
}

#[test]
fn asserted_edge_survives_gc() {
    let f = fx();
    let c = f.doc("Alice uses GraphRAG.");
    f.ok(
        c[0],
        "r1",
        vec![ent("Alice", "person"), ent("GraphRAG", "software")],
        vec![],
    );
    let g = f.db.get_entity_by_name("memory", "GraphRAG").unwrap().id;
    let grp = f.db.get_or_create_group("memory", "papers").unwrap();
    f.db.assert_edge(
        g,
        "part_of",
        grp,
        &Asserter::new("human:norman").unwrap(),
        None,
        None,
    )
    .unwrap();
    let rep = f.commit(c[0], "v2", "{}", AttemptOutcome::Empty);
    assert_eq!(rep.gc_entities, 1, "only Alice goes");
    assert!(f.db.get_entity_by_name("memory", "GraphRAG").is_ok());
}

#[test]
fn empty_outcome_is_terminal() {
    let f = fx();
    let c = f.doc("nothing here");
    let rep = f.commit(
        c[0],
        "v1",
        "{\"entities\":[],\"relationships\":[]}",
        AttemptOutcome::Empty,
    );
    assert!(!rep.replay);
    assert_eq!(
        f.count("SELECT COUNT(*) FROM attempts WHERE outcome = 'empty'"),
        1
    );
    assert_eq!(
        f.count("SELECT COUNT(*) FROM entities WHERE kind = 'entity'"),
        0
    );
    assert_eq!(
        f.count("SELECT COUNT(*) FROM step_status WHERE subject_kind='chunk' AND step='extract' AND state='current'"),
        1
    );
}

#[test]
fn implicit_endpoint_created_as_other() {
    let f = fx();
    let c = f.doc("Alice uses Ghost.");
    let rep = f.commit(
        c[0],
        "v1",
        "r",
        AttemptOutcome::Ok {
            entities: vec![ent("Alice", "person")],
            relations: vec![rel("Alice", "Ghost", "uses")],
        },
    );
    assert_eq!(rep.implicit_endpoints, 1);
    assert_eq!(rep.relations, 1);
    let g = f.db.get_entity_by_name("memory", "Ghost").unwrap();
    assert_eq!(g.entity_type.as_deref(), Some("other"));
}

#[test]
fn unmapped_relation_is_counted() {
    let f = fx();
    let c = f.doc("Alice frobs Bob.");
    let rep = f.commit(
        c[0],
        "v1",
        "r",
        AttemptOutcome::Ok {
            entities: vec![ent("Alice", "person"), ent("Bob", "person")],
            relations: vec![rel("Alice", "Bob", "frobs")],
        },
    );
    assert_eq!(rep.unmapped_relations, 1);
    assert!(f.relation_id("Alice", "frobs", "Bob").is_some());
    assert_eq!(f.count("SELECT unmapped_relations FROM attempts"), 1);
}

#[test]
fn stale_attempt_fp_is_rejected() {
    let f = fx();
    let c = f.doc("Alice uses GraphRAG.");
    let err =
        f.db.commit_attempt(&AttemptRecord {
            chunk_id: c[0],
            attempt_fp: "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            model: "m",
            profile: "paper-json",
            prompt_version: "v1",
            schema_version: "graph-v1",
            glean_max_rounds: 0,
            raw_response: "r",
            outcome: AttemptOutcome::Empty,
            rejected_entities: 0,
        })
        .unwrap_err();
    assert!(err.to_string().contains("fingerprint"), "{err}");
    assert_eq!(f.count("SELECT COUNT(*) FROM attempts"), 0);
}

#[test]
fn majority_type_wins() {
    let f = fx();
    let c = f.doc("# A\njj here\n# B\njj there\n# C\njj everywhere\n");
    f.ok(c[0], "a", vec![ent("jj", "software")], vec![]);
    f.ok(c[1], "b", vec![ent("jj", "person")], vec![]);
    f.ok(c[2], "c", vec![ent("jj", "Software")], vec![]);
    let jj = f.db.get_entity_by_name("memory", "jj").unwrap();
    assert_eq!(f.db.canonical_entity_type(jj.id).unwrap(), "software");
    assert_eq!(jj.entity_type.as_deref(), Some("software"));
}

#[test]
fn kind_becomes_isa_edge() {
    let f = fx();
    let c = f.doc("jj is a VCS.");
    let mut jj = ent("jj", "software");
    jj.kind = Some("version control system".into());
    f.ok(c[0], "r", vec![jj], vec![]);
    let isa = f
        .relation_id("jj", "isa", "version control system")
        .unwrap();
    assert_eq!(f.db.relation_support(isa).unwrap(), 1);
    let class =
        f.db.get_entity_by_name("memory", "version control system")
            .unwrap();
    assert_eq!(class.entity_type.as_deref(), Some("collection"));
}

#[test]
fn broader_becomes_genls_edge() {
    let f = fx();
    let c = f.doc("jj is a VCS.");
    let mut jj = ent("jj", "software");
    jj.kind = Some("version control system".into());
    jj.broader = Some("software tool".into());
    f.ok(c[0], "r", vec![jj], vec![]);
    assert!(
        f.relation_id("version control system", "genls", "software tool")
            .is_some()
    );
}

#[test]
fn class_entity_is_itself_the_class() {
    let f = fx();
    let c = f.doc("Databases store data.");
    let mut db_class = ent("databases", "concept");
    db_class.is_class = true;
    db_class.kind = Some("database".into());
    db_class.broader = Some("software".into());
    f.ok(c[0], "r", vec![db_class], vec![]);
    let e = f.db.get_entity_by_name("memory", "databases").unwrap();
    assert_eq!(e.entity_type.as_deref(), Some("collection"));
    assert!(f.relation_id("databases", "genls", "software").is_some());
    assert!(f.relation_id("databases", "isa", "database").is_none());
}

#[test]
fn mention_logged_once_per_evidence_unit() {
    let f = fx();
    let long: String = (0..700).map(|i| format!("w{i} ")).collect();
    let c = f.doc(&format!("# Only\n{long}\n# Second\nAlice again\n"));
    assert!(c.len() >= 3);
    let alice_uses = |f: &Fx| {
        let a = f.db.get_entity_by_name("memory", "Alice").unwrap().id;
        f.count(&format!(
            "SELECT COUNT(*) FROM uses WHERE item_kind = 'entity' AND item_id = {a} AND kind = 'mention'"
        ))
    };
    f.ok(c[0], "a", vec![ent("Alice", "person")], vec![]);
    f.ok(c[1], "b", vec![ent("Alice", "person")], vec![]);
    assert_eq!(alice_uses(&f), 1, "two chunks of one section");
    let last = *c.last().unwrap();
    f.ok(last, "c", vec![ent("Alice", "person")], vec![]);
    assert_eq!(alice_uses(&f), 2, "a second section");
}

#[test]
fn replay_logs_no_use() {
    let f = fx();
    let c = f.doc("Alice uses GraphRAG.");
    f.ok(c[0], "r", vec![ent("Alice", "person")], vec![]);
    let n = f.count("SELECT COUNT(*) FROM uses");
    f.ok(c[0], "r", vec![ent("Alice", "person")], vec![]);
    assert_eq!(f.count("SELECT COUNT(*) FROM uses"), n);
}

#[test]
fn reextraction_of_unchanged_text_logs_no_use() {
    let f = fx();
    let c = f.doc("Alice uses GraphRAG.");
    f.ok(c[0], "r1", vec![ent("Alice", "person")], vec![]);
    let n = f.count("SELECT COUNT(*) FROM uses");
    f.commit(
        c[0],
        "v2",
        "r2",
        AttemptOutcome::Ok {
            entities: vec![ent("Alice", "person")],
            relations: vec![],
        },
    );
    assert_eq!(f.count("SELECT COUNT(*) FROM uses"), n);
}
