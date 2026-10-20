//! Contract tests for the map-reduce plan/apply split.
//!
//! `plan` emits pending work units; an external executor runs them; `apply`
//! ingests results. Both halves are pure data transforms — no model required.
use graphrag_core::Database;
use graphrag_core::db::EntityInput;
use graphrag_core::mr::{
    EmbedResult, ExtractPrompt, ExtractResult, ExtractionStrategy, SummaryResult, WorkUnit,
    apply_embed, apply_extract, apply_extracted_entities, apply_summarize,
    build_community_hierarchy, extraction_provenance_nonce, pipeline_status, plan_embed,
    plan_extract, plan_summarize,
};
use hegel::TestCase;
use hegel::generators as gs;
use rusqlite::Connection;
use tempfile::TempDir;

/// Minimal strategy so the contract is exercised with no model and no
/// graphrag-llm dependency. Parses `A (Type) -[rel]-> B (Type)`.
struct FakeStrategy;

impl ExtractionStrategy for FakeStrategy {
    fn prompt(&self, chunk: &str, idx: usize, total: usize) -> ExtractPrompt {
        ExtractPrompt {
            system: "extract triples".into(),
            user: format!("[{}/{}] {}", idx + 1, total, chunk),
            format: None,
        }
    }
    fn parse(&self, response: &str) -> Vec<EntityInput> {
        response
            .lines()
            .filter_map(|l| {
                let (head, rest) = l.split_once("-[")?;
                let (rel, tail) = rest.split_once("]->")?;
                let split = |s: &str| {
                    let s = s.trim();
                    match s.split_once(" (") {
                        Some((n, t)) => (
                            n.trim().to_string(),
                            Some(t.trim_end_matches(')').trim().to_string()),
                        ),
                        None => (s.to_string(), None),
                    }
                };
                let (h, ht) = split(head);
                let (tl, tt) = split(tail);
                if h.is_empty() || tl.is_empty() {
                    return None;
                }
                Some(EntityInput {
                    head: h,
                    head_type: ht,
                    relation: rel.trim().to_string(),
                    tail: tl,
                    tail_type: tt,
                    properties: None,
                })
            })
            .collect()
    }
}

const S: &FakeStrategy = &FakeStrategy;

fn db_with_chunks(n: usize) -> (TempDir, Database) {
    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("t.db")).unwrap();
    db.create_store("s", 3).unwrap();
    for i in 0..n {
        db.add_chunk("s", &format!("chunk {i}: Alice uses GraphRAG."), None, None)
            .unwrap();
    }
    (dir, db)
}

#[test]
fn legacy_checkpoint_migrates_without_fabricating_exact_provenance() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("legacy.db");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
        "CREATE TABLE stores (
             name TEXT PRIMARY KEY, dim INTEGER NOT NULL, created_at TEXT
         );
         CREATE TABLE chunks (
             id INTEGER PRIMARY KEY, store TEXT NOT NULL, content TEXT NOT NULL,
             source TEXT, metadata TEXT, created_at TEXT
         );
         CREATE TABLE extractions (
             chunk_id INTEGER PRIMARY KEY, model TEXT NOT NULL,
             triple_count INTEGER NOT NULL, extracted_at TEXT NOT NULL
         );
         INSERT INTO stores VALUES ('s', 3, datetime('now'));
         INSERT INTO chunks VALUES (1, 's', 'legacy chunk', NULL, NULL, datetime('now'));
         INSERT INTO extractions VALUES (1, 'legacy-model', 2, datetime('now'));",
    )
    .unwrap();
    drop(conn);

    let db = Database::open(&path).unwrap();
    assert!(plan_extract(&db, "s", "m", S, None).unwrap().is_empty());
    assert!(
        db.get_extraction_provenance("s", "legacy:chunk:1")
            .unwrap()
            .is_none()
    );
    let nonce = extraction_provenance_nonce("s", 1, "m", "legacy chunk");
    assert!(
        apply_extract(
            &db,
            "s",
            S,
            &[ExtractResult {
                chunk_id: 1,
                model: "m".into(),
                provenance_nonce: nonce,
                response: "A (Thing) -[links]-> B (Thing)".into(),
            }],
        )
        .is_err(),
        "legacy checkpoint must conflict rather than acquire fabricated provenance"
    );
    assert!(db.list_entities("s").unwrap().is_empty());
    drop(db);

    let conn = Connection::open(path).unwrap();
    for column in ["provenance_nonce", "response_sha256"] {
        let present: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('extractions') WHERE name = ?1",
                [column],
                |row| row.get(0),
            )
            .unwrap();
        assert!(present, "migration adds {column}");
    }
    let links: i64 = conn
        .query_row("SELECT COUNT(*) FROM extraction_relations", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(
        links, 0,
        "legacy edges are unknowable and must not be inferred"
    );
}

// --- plan -----------------------------------------------------------------

#[test]
fn plan_emits_one_unit_per_pending_chunk() {
    let (_d, db) = db_with_chunks(3);
    let units = plan_extract(&db, "s", "gemma4:31b", S, None).unwrap();
    assert_eq!(units.len(), 3);
    assert!(units[0].unit_id.contains("chunk"));
    assert_eq!(units[0].model, "gemma4:31b");
    assert!(
        !units[0].user.is_empty(),
        "prompt must be built by the strategy"
    );
}

#[test]
fn plan_is_ordered_and_stable() {
    let (_d, db) = db_with_chunks(3);
    let a = plan_extract(&db, "s", "gemma4:31b", S, None).unwrap();
    let b = plan_extract(&db, "s", "gemma4:31b", S, None).unwrap();
    let ids: Vec<_> = a.iter().map(|u| u.chunk_id).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted, "units ascend by chunk id");
    assert_eq!(ids, b.iter().map(|u| u.chunk_id).collect::<Vec<_>>());
    assert!(a.iter().all(|u| u.provenance_nonce.is_some()));
    assert_eq!(
        a.iter()
            .map(|u| u.provenance_nonce.as_deref())
            .collect::<Vec<_>>(),
        b.iter()
            .map(|u| u.provenance_nonce.as_deref())
            .collect::<Vec<_>>(),
        "pure replanning must emit the same correlation nonce"
    );
}

#[test]
fn plan_excludes_already_extracted_chunks() {
    let (_d, db) = db_with_chunks(3);
    let units = plan_extract(&db, "s", "gemma4:31b", S, None).unwrap();
    apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: units[0].chunk_id,
            model: "gemma4:31b".into(),
            provenance_nonce: units[0].provenance_nonce.clone().unwrap(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
        }],
    )
    .unwrap();
    let after = plan_extract(&db, "s", "gemma4:31b", S, None).unwrap();
    assert_eq!(after.len(), 2, "extracted chunk must not be re-planned");
}

#[test]
fn plan_limit_bounds_output() {
    let (_d, db) = db_with_chunks(5);
    assert_eq!(plan_extract(&db, "s", "m", S, Some(2)).unwrap().len(), 2);
}

#[test]
fn plan_unknown_store_errors() {
    let (_d, db) = db_with_chunks(1);
    assert!(plan_extract(&db, "nope", "m", S, None).is_err());
}

// --- apply ----------------------------------------------------------------

#[test]
fn apply_persists_triples_and_checkpoint() {
    let (_d, db) = db_with_chunks(1);
    let u = &plan_extract(&db, "s", "gemma4:31b", S, None).unwrap()[0];
    let n = apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: u.chunk_id,
            model: "gemma4:31b".into(),
            provenance_nonce: u.provenance_nonce.clone().unwrap(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
        }],
    )
    .unwrap();
    assert_eq!(n, 1, "one triple persisted");
    assert!(!db.list_entities("s").unwrap().is_empty());
    assert!(
        plan_extract(&db, "s", "gemma4:31b", S, None)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn zero_yield_is_terminal_not_pending_forever() {
    // The reason we need an explicit checkpoint: a chunk yielding no triples
    // would otherwise look pending on every run.
    let (_d, db) = db_with_chunks(1);
    let u = &plan_extract(&db, "s", "m", S, None).unwrap()[0];
    let n = apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: u.chunk_id,
            model: "m".into(),
            provenance_nonce: u.provenance_nonce.clone().unwrap(),
            response: "".into(),
        }],
    )
    .unwrap();
    assert_eq!(n, 0);
    let provenance = db
        .get_extraction_provenance("s", u.provenance_nonce.as_deref().unwrap())
        .unwrap()
        .expect("zero-yield extraction remains resolvable");
    assert_eq!(provenance.triple_count, 0);
    assert!(provenance.entities.is_empty());
    assert!(provenance.relations.is_empty());
    assert!(
        plan_extract(&db, "s", "m", S, None).unwrap().is_empty(),
        "zero-yield chunk must be marked done, not replanned"
    );
}

#[test]
fn apply_is_idempotent() {
    let (_d, db) = db_with_chunks(1);
    let u = &plan_extract(&db, "s", "m", S, None).unwrap()[0];
    let r = ExtractResult {
        chunk_id: u.chunk_id,
        model: "m".into(),
        provenance_nonce: u.provenance_nonce.clone().unwrap(),
        response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
    };
    apply_extract(&db, "s", S, std::slice::from_ref(&r)).unwrap();
    let entities_before = db.list_entities("s").unwrap().len();
    let relations_before = db.list_relations("s").unwrap().len();
    apply_extract(&db, "s", S, std::slice::from_ref(&r)).unwrap();
    assert_eq!(db.list_entities("s").unwrap().len(), entities_before);
    assert_eq!(
        db.list_relations("s").unwrap().len(),
        relations_before,
        "re-applying one nonce must not duplicate edges"
    );
}

#[test]
fn replayed_nonce_with_changed_response_is_rejected() {
    let (_d, db) = db_with_chunks(1);
    let unit = plan_extract(&db, "s", "m", S, None).unwrap().remove(0);
    let first = ExtractResult {
        chunk_id: unit.chunk_id,
        model: unit.model.clone(),
        provenance_nonce: unit.provenance_nonce.clone().unwrap(),
        response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
    };
    apply_extract(&db, "s", S, std::slice::from_ref(&first)).unwrap();
    let changed = ExtractResult {
        response: "Alice (Person) -[abandons]-> GraphRAG (Software)".into(),
        ..first
    };
    assert!(apply_extract(&db, "s", S, &[changed]).is_err());
    assert_eq!(db.list_relations("s").unwrap().len(), 1);
}

#[test]
fn provenance_nonce_resolves_exact_entities_and_edges() {
    let (_d, db) = db_with_chunks(1);
    let unrelated_head = db
        .get_or_create_entity("s", "Unrelated", Some("Concept"), None)
        .unwrap();
    let unrelated_tail = db
        .get_or_create_entity("s", "Noise", Some("Concept"), None)
        .unwrap();
    db.add_relation("s", unrelated_head, unrelated_tail, "distracts", None)
        .unwrap();

    let unit = plan_extract(&db, "s", "m", S, None).unwrap().remove(0);
    let nonce = unit.provenance_nonce.clone().unwrap();
    apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: unit.chunk_id,
            model: unit.model,
            provenance_nonce: nonce.clone(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)\nAlice (Person) -[maintains]-> Rust (Software)".into(),
        }],
    )
    .unwrap();

    db.create_store("other", 3).unwrap();
    assert!(
        db.get_extraction_provenance("other", &nonce)
            .unwrap()
            .is_none(),
        "nonce lookup is store-scoped"
    );
    let provenance = db
        .get_extraction_provenance("s", &nonce)
        .unwrap()
        .expect("nonce resolves");
    assert_eq!(provenance.chunk_id, unit.chunk_id);
    assert_eq!(provenance.triple_count, 2);
    assert_eq!(provenance.relations.len(), 2);
    let names: Vec<_> = provenance
        .entities
        .iter()
        .map(|entity| entity.name.as_str())
        .collect();
    assert_eq!(names, vec!["Alice", "GraphRAG", "Rust"]);
    assert!(
        provenance
            .relations
            .iter()
            .all(|relation| relation.relation != "distracts"),
        "nonce must not induce unrelated edges from the chunk/global graph"
    );

    let mut exported = Vec::new();
    graphrag_core::export::export_store(&db, "s", &mut exported).unwrap();
    let record = String::from_utf8(exported)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .find(|value| value["record"] == "extraction_provenance")
        .expect("durable export includes exact provenance");
    assert_eq!(record["provenance_nonce"], nonce);
    assert_eq!(record["entity_ids"].as_array().unwrap().len(), 3);
    assert_eq!(record["relation_ids"].as_array().unwrap().len(), 2);
}

#[test]
fn entity_merge_keeps_extraction_provenance_and_resolves() {
    let (_d, db) = db_with_chunks(1);
    let unit = plan_extract(&db, "s", "m", S, None).unwrap().remove(0);
    let nonce = unit.provenance_nonce.clone().unwrap();
    apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: unit.chunk_id,
            model: unit.model,
            provenance_nonce: nonce.clone(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
        }],
    )
    .unwrap();
    let before = db.get_extraction_provenance("s", &nonce).unwrap().unwrap();
    let alice = before
        .entities
        .iter()
        .find(|entity| entity.name == "Alice")
        .unwrap()
        .id;
    let graph = before
        .entities
        .iter()
        .find(|entity| entity.name == "GraphRAG")
        .unwrap()
        .id;

    db.merge_entities(graph, alice).unwrap();

    // Merges are same_as assertions now: provenance rows are untouched and
    // reads resolve the merged entity to its representative.
    let after = db.get_extraction_provenance("s", &nonce).unwrap().unwrap();
    assert_eq!(after.entities.len(), 2);
    assert_eq!(after.relations.len(), 1);
    assert_eq!(
        (after.relations[0].head_id, after.relations[0].tail_id),
        (alice, graph)
    );
    assert_eq!(db.resolve_entity(alice).unwrap(), graph);
}

#[test]
fn mid_write_failure_rolls_back_checkpoint_entities_relations_and_links() {
    let (dir, db) = db_with_chunks(1);
    let unit = plan_extract(&db, "s", "m", S, None).unwrap().remove(0);
    let nonce = unit.provenance_nonce.clone().unwrap();
    let conn = Connection::open(dir.path().join("t.db")).unwrap();
    conn.execute_batch(
        "CREATE TRIGGER fail_second_provenance_relation
         BEFORE INSERT ON extraction_relations
         WHEN (SELECT COUNT(*) FROM extraction_relations) > 0
         BEGIN SELECT RAISE(ABORT, 'injected mid-write failure'); END;",
    )
    .unwrap();
    drop(conn);

    let result = ExtractResult {
        chunk_id: unit.chunk_id,
        model: unit.model,
        provenance_nonce: nonce.clone(),
        response: "Alice (Person) -[uses]-> GraphRAG (Software)\nAlice (Person) -[maintains]-> Rust (Software)".into(),
    };
    assert!(apply_extract(&db, "s", S, &[result]).is_err());
    assert!(db.list_entities("s").unwrap().is_empty());
    assert!(db.list_relations("s").unwrap().is_empty());
    assert!(db.get_extraction_provenance("s", &nonce).unwrap().is_none());
    assert_eq!(plan_extract(&db, "s", "m", S, None).unwrap().len(), 1);
}

#[test]
fn invalid_provenance_nonce_rolls_back_everything() {
    let (_d, db) = db_with_chunks(1);
    let unit = plan_extract(&db, "s", "m", S, None).unwrap().remove(0);
    let result = ExtractResult {
        chunk_id: unit.chunk_id,
        model: unit.model,
        provenance_nonce: "wrong".into(),
        response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
    };
    assert!(apply_extract(&db, "s", S, &[result]).is_err());
    assert!(db.list_entities("s").unwrap().is_empty());
    assert!(db.list_relations("s").unwrap().is_empty());
    assert_eq!(plan_extract(&db, "s", "m", S, None).unwrap().len(), 1);
    assert!(
        db.get_extraction_provenance("s", "wrong")
            .unwrap()
            .is_none()
    );
}

#[test]
fn repeated_triple_in_one_response_is_one_relationship_instance() {
    let (_d, db) = db_with_chunks(1);
    let unit = plan_extract(&db, "s", "m", S, None).unwrap().remove(0);
    let nonce = unit.provenance_nonce.clone().unwrap();
    apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: unit.chunk_id,
            model: unit.model,
            provenance_nonce: nonce.clone(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)\nAlice (Person) -[uses]-> GraphRAG (Software)".into(),
        }],
    )
    .expect("a repeated triple must not abort the chunk");
    assert_eq!(db.list_relations("s").unwrap().len(), 1);
    let provenance = db.get_extraction_provenance("s", &nonce).unwrap().unwrap();
    assert_eq!(provenance.relations.len(), 1);
    assert!(plan_extract(&db, "s", "m", S, None).unwrap().is_empty());
}

#[test]
fn relation_properties_from_parsed_triples_are_persisted() {
    let (_d, db) = db_with_chunks(1);
    let unit = plan_extract(&db, "s", "m", S, None).unwrap().remove(0);
    let result = ExtractResult {
        chunk_id: unit.chunk_id,
        model: unit.model,
        provenance_nonce: unit.provenance_nonce.clone().unwrap(),
        response: String::new(),
    };
    let triple = EntityInput {
        head: "Alice".into(),
        head_type: Some("Person".into()),
        relation: "uses".into(),
        tail: "GraphRAG".into(),
        tail_type: Some("Software".into()),
        properties: Some(serde_json::json!({"since": 2024})),
    };
    apply_extracted_entities(&db, "s", &result, &[triple]).unwrap();
    let relations = db.list_relations("s").unwrap();
    assert_eq!(relations.len(), 1);
    let stored: serde_json::Value =
        serde_json::from_str(relations[0].properties.as_deref().expect("properties kept")).unwrap();
    assert_eq!(stored, serde_json::json!({"since": 2024}));
}

#[test]
fn reused_entity_is_linked_to_each_exact_extraction() {
    let (_d, db) = db_with_chunks(2);
    let units = plan_extract(&db, "s", "m", S, None).unwrap();
    let make = |unit: &WorkUnit, response: &str| ExtractResult {
        chunk_id: unit.chunk_id,
        model: unit.model.clone(),
        provenance_nonce: unit.provenance_nonce.clone().unwrap(),
        response: response.into(),
    };
    apply_extract(
        &db,
        "s",
        S,
        &[make(
            &units[0],
            "Alice (Person) -[uses]-> GraphRAG (Software)",
        )],
    )
    .unwrap();
    apply_extract(
        &db,
        "s",
        S,
        &[make(
            &units[1],
            "GraphRAG (Software) -[indexes]-> Rust (Language)",
        )],
    )
    .unwrap();

    let first = db
        .get_extraction_provenance("s", units[0].provenance_nonce.as_deref().unwrap())
        .unwrap()
        .unwrap();
    let second = db
        .get_extraction_provenance("s", units[1].provenance_nonce.as_deref().unwrap())
        .unwrap()
        .unwrap();
    let first_graph = first
        .entities
        .iter()
        .find(|entity| entity.name == "GraphRAG")
        .unwrap()
        .id;
    let second_graph = second
        .entities
        .iter()
        .find(|entity| entity.name == "GraphRAG")
        .unwrap()
        .id;
    assert_eq!(
        first_graph, second_graph,
        "preexisting entity is linked, not copied"
    );
    assert_eq!(second.relations.len(), 1);
    assert_eq!(
        second
            .entities
            .iter()
            .map(|entity| entity.name.as_str())
            .collect::<Vec<_>>(),
        vec!["GraphRAG", "Rust"]
    );
}

#[test]
fn repeated_global_edge_is_deduplicated_but_linked_to_each_extraction() {
    let (_d, db) = db_with_chunks(2);
    let units = plan_extract(&db, "s", "m", S, None).unwrap();
    let results = units
        .iter()
        .map(|unit| ExtractResult {
            chunk_id: unit.chunk_id,
            model: unit.model.clone(),
            provenance_nonce: unit.provenance_nonce.clone().unwrap(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
        })
        .collect::<Vec<_>>();
    apply_extract(&db, "s", S, &results).unwrap();

    assert_eq!(db.list_relations("s").unwrap().len(), 1);
    let first = db
        .get_extraction_provenance("s", &results[0].provenance_nonce)
        .unwrap()
        .unwrap();
    let second = db
        .get_extraction_provenance("s", &results[1].provenance_nonce)
        .unwrap()
        .unwrap();
    assert_eq!(first.relations.len(), 1);
    assert_eq!(second.relations.len(), 1);
    assert_eq!(first.relations[0].id, second.relations[0].id);
}

#[test]
fn apply_accepts_partial_and_out_of_order_results() {
    let (_d, db) = db_with_chunks(4);
    let units = plan_extract(&db, "s", "m", S, None).unwrap();
    let mk = |u: &WorkUnit| ExtractResult {
        chunk_id: u.chunk_id,
        model: "m".into(),
        provenance_nonce: u.provenance_nonce.clone().unwrap(),
        response: "A (Person) -[knows]-> B (Person)".into(),
    };
    // only units 3 and 1, reversed — a crashed/partial executor run
    apply_extract(&db, "s", S, &[mk(&units[3]), mk(&units[1])]).unwrap();
    let pending: Vec<_> = plan_extract(&db, "s", "m", S, None)
        .unwrap()
        .iter()
        .map(|u| u.chunk_id)
        .collect();
    assert_eq!(pending, vec![units[0].chunk_id, units[2].chunk_id]);
}

#[test]
fn apply_tolerates_malformed_response() {
    let (_d, db) = db_with_chunks(1);
    let u = &plan_extract(&db, "s", "m", S, None).unwrap()[0];
    let n = apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: u.chunk_id,
            model: "m".into(),
            provenance_nonce: u.provenance_nonce.clone().unwrap(),
            response: "%%% not a triple at all %%%".into(),
        }],
    )
    .unwrap();
    assert_eq!(n, 0, "garbage yields no triples but must not error");
    assert!(
        plan_extract(&db, "s", "m", S, None).unwrap().is_empty(),
        "still checkpointed"
    );
}

#[test]
fn work_unit_is_json_roundtrippable() {
    let (_d, db) = db_with_chunks(1);
    let u = &plan_extract(&db, "s", "m", S, None).unwrap()[0];
    let s = serde_json::to_string(u).unwrap();
    let back: WorkUnit = serde_json::from_str(&s).unwrap();
    assert_eq!(back.chunk_id, u.chunk_id);
    assert_eq!(back.user, u.user);
}

// --- stage 2: community -> summary ----------------------------------------

fn db_with_community() -> (TempDir, Database, i64) {
    let (d, db) = db_with_chunks(1);
    let units = plan_extract(&db, "s", "m", S, None).unwrap();
    apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: units[0].chunk_id,
            model: "m".into(),
            provenance_nonce: units[0].provenance_nonce.clone().unwrap(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
        }],
    )
    .unwrap();
    let cid = db.create_community("s", 0, 0.0, None).unwrap();
    for e in db.list_entities("s").unwrap() {
        db.link_entity_community(e.id, cid).unwrap();
    }
    (d, db, cid)
}

#[test]
fn plan_summarize_emits_units_for_unsummarized_communities() {
    let (_d, db, cid) = db_with_community();
    let units = plan_summarize(&db, "s", "m", None).unwrap();
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].kind, "summarize");
    assert!(units[0].unit_id.contains(&cid.to_string()));
    assert!(
        units[0].user.contains("Alice"),
        "member entities appear in the prompt"
    );
}

#[test]
fn apply_summarize_persists_and_checkpoints() {
    let (_d, db, cid) = db_with_community();
    let n = apply_summarize(
        &db,
        "s",
        &[SummaryResult {
            community_id: cid,
            model: "m".into(),
            response: "A community about GraphRAG usage.".into(),
        }],
    )
    .unwrap();
    assert_eq!(n, 1);
    let c = db.list_communities("s").unwrap();
    assert_eq!(
        c[0].summary.as_deref(),
        Some("A community about GraphRAG usage.")
    );
    assert!(
        plan_summarize(&db, "s", "m", None).unwrap().is_empty(),
        "summarized community must not be replanned"
    );
}

#[test]
fn empty_summary_is_terminal_not_replanned() {
    let (_d, db, cid) = db_with_community();
    apply_summarize(
        &db,
        "s",
        &[SummaryResult {
            community_id: cid,
            model: "m".into(),
            response: "   ".into(),
        }],
    )
    .unwrap();
    assert!(
        plan_summarize(&db, "s", "m", None).unwrap().is_empty(),
        "a blank reply still terminates the unit"
    );
}

#[test]
fn summarize_accepts_partial_and_out_of_order() {
    let (_d, db, _c) = db_with_community();
    let c2 = db.create_community("s", 0, 0.0, None).unwrap();
    let c3 = db.create_community("s", 0, 0.0, None).unwrap();
    let pending: Vec<i64> = plan_summarize(&db, "s", "m", None)
        .unwrap()
        .iter()
        .map(|u| u.community_id.unwrap())
        .collect();
    assert_eq!(pending.len(), 3);
    apply_summarize(
        &db,
        "s",
        &[
            SummaryResult {
                community_id: c3,
                model: "m".into(),
                response: "third".into(),
            },
            SummaryResult {
                community_id: c2,
                model: "m".into(),
                response: "second".into(),
            },
        ],
    )
    .unwrap();
    let after: Vec<i64> = plan_summarize(&db, "s", "m", None)
        .unwrap()
        .iter()
        .map(|u| u.community_id.unwrap())
        .collect();
    assert_eq!(after, vec![pending[0]]);
}

// --- stage 2b: hierarchical summarization (GraphRAG paper, §3.1.5) ---------
//
// The paper builds community summaries bottom-up: leaf communities summarize
// their elements, and higher-level communities substitute their SUB-COMMUNITY
// summaries for element summaries. That is what lets a root summary describe
// thousands of entities within a context window, and it is why root-level
// summaries answer global questions at a fraction of the token cost.

#[test]
fn parent_is_not_plannable_until_children_are_summarized() {
    let (_d, db) = db_with_chunks(1);
    let parent = db.create_community("s", 1, 0.0, None).unwrap();
    let child_a = db.create_community("s", 0, 0.0, Some(parent)).unwrap();
    let child_b = db.create_community("s", 0, 0.0, Some(parent)).unwrap();

    // only leaves are plannable at first
    let ids: Vec<i64> = plan_summarize(&db, "s", "m", None)
        .unwrap()
        .iter()
        .map(|u| u.community_id.unwrap())
        .collect();
    assert_eq!(
        ids,
        vec![child_a, child_b],
        "parent must wait for its children"
    );

    apply_summarize(
        &db,
        "s",
        &[SummaryResult {
            community_id: child_a,
            model: "m".into(),
            response: "child A topic".into(),
        }],
    )
    .unwrap();
    let ids: Vec<i64> = plan_summarize(&db, "s", "m", None)
        .unwrap()
        .iter()
        .map(|u| u.community_id.unwrap())
        .collect();
    assert_eq!(ids, vec![child_b], "one child still outstanding");

    apply_summarize(
        &db,
        "s",
        &[SummaryResult {
            community_id: child_b,
            model: "m".into(),
            response: "child B topic".into(),
        }],
    )
    .unwrap();
    let ids: Vec<i64> = plan_summarize(&db, "s", "m", None)
        .unwrap()
        .iter()
        .map(|u| u.community_id.unwrap())
        .collect();
    assert_eq!(
        ids,
        vec![parent],
        "parent becomes plannable once children are done"
    );
}

#[test]
fn parent_prompt_substitutes_child_summaries_for_entities() {
    let (_d, db) = db_with_chunks(1);
    let units = plan_extract(&db, "s", "m", S, None).unwrap();
    apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: units[0].chunk_id,
            model: "m".into(),
            provenance_nonce: units[0].provenance_nonce.clone().unwrap(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
        }],
    )
    .unwrap();

    let parent = db.create_community("s", 1, 0.0, None).unwrap();
    let child = db.create_community("s", 0, 0.0, Some(parent)).unwrap();
    // both levels carry the entities, as hierarchical Leiden persists them
    for e in db.list_entities("s").unwrap() {
        db.link_entity_community(e.id, child).unwrap();
        db.link_entity_community(e.id, parent).unwrap();
    }
    apply_summarize(
        &db,
        "s",
        &[SummaryResult {
            community_id: child,
            model: "m".into(),
            response: "Alice's use of GraphRAG tooling".into(),
        }],
    )
    .unwrap();

    let u = plan_summarize(&db, "s", "m", None).unwrap();
    assert_eq!(u.len(), 1);
    let p = &u[0];
    assert_eq!(p.community_id, Some(parent));
    assert!(
        p.user.contains("Alice's use of GraphRAG tooling"),
        "parent prompt must carry the child summary, got: {}",
        p.user
    );
    assert!(
        !p.user.contains("Alice (Person)"),
        "parent prompt must NOT fall back to the raw entity list"
    );
}

// --- pipeline status: which phases exist, what is ready, what went stale ----

#[test]
fn status_reports_every_phase_with_counts_and_readiness() {
    let (_d, db) = db_with_chunks(3);
    let st = pipeline_status(&db, "s").unwrap();
    let names: Vec<&str> = st.phases.iter().map(|p| p.phase.as_str()).collect();
    assert_eq!(names, vec!["extract", "embed", "communities", "summarize"]);

    let ex = &st.phases[0];
    assert_eq!((ex.total, ex.done, ex.pending), (3, 0, 3));
    assert!(ex.ready, "extract is ready as soon as chunks exist");

    // downstream phases cannot start yet and must say why
    let comm = &st.phases[1];
    assert!(!comm.ready);
    assert!(
        comm.blocked_by.as_deref() == Some("extract"),
        "{:?}",
        comm.blocked_by
    );
    assert!(!st.next.is_empty(), "status names a concrete next action");
}

#[test]
fn status_detects_entities_that_appeared_after_community_detection() {
    let (_d, db) = db_with_chunks(2);
    let units = plan_extract(&db, "s", "m", S, None).unwrap();
    apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: units[0].chunk_id,
            model: "m".into(),
            provenance_nonce: units[0].provenance_nonce.clone().unwrap(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
        }],
    )
    .unwrap();

    // detect communities over what exists now
    let c = db.create_community("s", 0, 0.0, None).unwrap();
    for e in db.list_entities("s").unwrap() {
        db.link_entity_community(e.id, c).unwrap();
    }
    let st = pipeline_status(&db, "s").unwrap();
    assert_eq!(
        st.phase("communities").unwrap().pending,
        0,
        "nothing new yet"
    );

    // a later extraction introduces entities the detection never saw
    apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: units[1].chunk_id,
            model: "m".into(),
            provenance_nonce: units[1].provenance_nonce.clone().unwrap(),
            response: "Bruce (Person) -[reviews]-> leit (Software)".into(),
        }],
    )
    .unwrap();
    let st = pipeline_status(&db, "s").unwrap();
    let comm = st.phase("communities").unwrap();
    assert!(
        comm.pending > 0,
        "unclustered entities must surface as pending"
    );
    assert!(
        comm.guidance.contains("enrich"),
        "guidance names the command: {}",
        comm.guidance
    );
}

#[test]
fn status_summarize_reports_hierarchy_and_blocked_parents() {
    let (_d, db) = db_with_chunks(1);
    let parent = db.create_community("s", 1, 0.0, None).unwrap();
    let child = db.create_community("s", 0, 0.0, Some(parent)).unwrap();
    let st = pipeline_status(&db, "s").unwrap();
    let s = st.phase("summarize").unwrap();
    assert_eq!(s.total, 2);
    assert_eq!(s.pending, 1, "only the leaf is ready");
    assert_eq!(s.blocked, 1, "the parent waits on its child");
    assert_eq!(
        s.levels,
        vec![(0, 1, 0), (1, 1, 0)],
        "(level, count, summarized)"
    );

    apply_summarize(
        &db,
        "s",
        &[SummaryResult {
            community_id: child,
            model: "m".into(),
            response: "leaf".into(),
        }],
    )
    .unwrap();
    let st = pipeline_status(&db, "s").unwrap();
    let s = st.phase("summarize").unwrap();
    assert_eq!(
        (s.pending, s.blocked),
        (1, 0),
        "parent unblocks once the child is done"
    );
}

// --- structural invariants of the community hierarchy ----------------------
//
// Norman's requirement: no orphaned nodes, and every community must chain up
// to a single root that names the high-level concepts beneath it. A forest of
// hundreds of parentless communities cannot answer a global question, because
// there is no vantage point that sees everything.

fn seed_connected_graph(db: &Database) {
    // two dense clusters joined by one bridge — a graph with real structure
    let pairs = [
        ("Leiden", "modularity"),
        ("Leiden", "community"),
        ("modularity", "community"),
        ("community", "summary"),
        ("Leiden", "summary"),
        ("BM25", "lexical"),
        ("BM25", "scoring"),
        ("lexical", "scoring"),
        ("scoring", "ranking"),
        ("BM25", "ranking"),
        ("community", "ranking"), // the bridge
    ];
    for (h, t) in pairs {
        let hid = db
            .get_or_create_entity("s", h, Some("Concept"), None)
            .unwrap();
        let tid = db
            .get_or_create_entity("s", t, Some("Concept"), None)
            .unwrap();
        db.add_relation("s", hid, tid, "relates_to", None).unwrap();
    }
}

#[test]
fn every_entity_belongs_to_a_community() {
    let (_d, db) = db_with_chunks(1);
    seed_connected_graph(&db);
    build_community_hierarchy(&db, "s", Default::default()).unwrap();
    assert_eq!(
        db.unclustered_entity_count("s").unwrap(),
        0,
        "no entity may be orphaned from the hierarchy"
    );
}

#[test]
fn hierarchy_converges_to_exactly_one_root() {
    let (_d, db) = db_with_chunks(1);
    seed_connected_graph(&db);
    build_community_hierarchy(&db, "s", Default::default()).unwrap();
    let roots: Vec<_> = db
        .list_communities("s")
        .unwrap()
        .into_iter()
        .filter(|c| c.parent_id.is_none())
        .collect();
    assert_eq!(
        roots.len(),
        1,
        "expected a single root, got {}",
        roots.len()
    );
}

#[test]
fn every_community_reaches_the_root_by_following_parents() {
    let (_d, db) = db_with_chunks(1);
    seed_connected_graph(&db);
    build_community_hierarchy(&db, "s", Default::default()).unwrap();
    let all = db.list_communities("s").unwrap();
    let by_id: std::collections::HashMap<i64, Option<i64>> =
        all.iter().map(|c| (c.id, c.parent_id)).collect();
    for c in &all {
        let (mut cur, mut hops) = (c.id, 0);
        while let Some(Some(p)) = by_id.get(&cur) {
            cur = *p;
            hops += 1;
            assert!(hops < 100, "cycle in parent chain from community {}", c.id);
        }
        assert!(
            by_id.contains_key(&cur),
            "community {} chain left the store",
            c.id
        );
        assert!(
            by_id[&cur].is_none(),
            "chain from {} did not end at a root",
            c.id
        );
    }
}

#[test]
fn root_is_coarser_than_its_children() {
    let (_d, db) = db_with_chunks(1);
    seed_connected_graph(&db);
    build_community_hierarchy(&db, "s", Default::default()).unwrap();
    let all = db.list_communities("s").unwrap();
    let root = all.iter().find(|c| c.parent_id.is_none()).unwrap();
    let kids = all.iter().filter(|c| c.parent_id == Some(root.id)).count();
    assert!(
        kids >= 2,
        "a root that groups <2 children conveys nothing; got {kids}"
    );
    let root_members = db.get_community_entities(root.id).unwrap().len();
    for c in all.iter().filter(|c| c.parent_id == Some(root.id)) {
        assert!(
            db.get_community_entities(c.id).unwrap().len() <= root_members,
            "child community larger than its parent"
        );
    }
}

// --- phase: entity embedding ------------------------------------------------
//
// Entities need their own vectors on the RAG side. Lexical variants that stay
// distinct nodes ("Leiden" / "Leiden algorithm" / "hierarchical Leiden") embed
// at cosine 0.74-1.00 against each other and 0.24-0.31 against unrelated terms,
// so with vectors present a query for any one surfaces all of them. Without
// them the merge-candidate tooling is inert and variants never co-surface.

#[test]
fn plan_embed_lists_entities_without_vectors() {
    let (_d, db) = db_with_chunks(1);
    let u = plan_extract(&db, "s", "m", S, None).unwrap();
    apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: u[0].chunk_id,
            model: "m".into(),
            provenance_nonce: u[0].provenance_nonce.clone().unwrap(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
        }],
    )
    .unwrap();

    let units = plan_embed(&db, "s", None).unwrap();
    assert_eq!(units.len(), 2, "both entities need vectors");
    assert!(
        units.iter().any(|e| e.text == "Alice (Person)"),
        "text is type-qualified for embedding"
    );
    assert!(units.iter().all(|e| e.entity_id > 0));
}

#[test]
fn apply_embed_persists_and_is_terminal() {
    let (_d, db) = db_with_chunks(1);
    let u = plan_extract(&db, "s", "m", S, None).unwrap();
    apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: u[0].chunk_id,
            model: "m".into(),
            provenance_nonce: u[0].provenance_nonce.clone().unwrap(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
        }],
    )
    .unwrap();

    let units = plan_embed(&db, "s", None).unwrap();
    let results: Vec<EmbedResult> = units
        .iter()
        .map(|e| EmbedResult {
            entity_id: e.entity_id,
            vector: vec![0.1, 0.2, 0.3],
        })
        .collect();
    assert_eq!(apply_embed(&db, "s", &results).unwrap(), 2);
    assert!(
        plan_embed(&db, "s", None).unwrap().is_empty(),
        "embedded entities are terminal"
    );
    assert_eq!(
        db.get_entity_embedding(units[0].entity_id)
            .unwrap()
            .unwrap(),
        vec![0.1, 0.2, 0.3]
    );
}

#[test]
fn status_reports_embed_as_a_phase_between_extract_and_communities() {
    let (_d, db) = db_with_chunks(1);
    let u = plan_extract(&db, "s", "m", S, None).unwrap();
    apply_extract(
        &db,
        "s",
        S,
        &[ExtractResult {
            chunk_id: u[0].chunk_id,
            model: "m".into(),
            provenance_nonce: u[0].provenance_nonce.clone().unwrap(),
            response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
        }],
    )
    .unwrap();

    let st = pipeline_status(&db, "s").unwrap();
    let names: Vec<&str> = st.phases.iter().map(|p| p.phase.as_str()).collect();
    assert_eq!(names, vec!["extract", "embed", "communities", "summarize"]);
    let e = st.phase("embed").unwrap();
    assert_eq!((e.total, e.done, e.pending), (2, 0, 2));
    assert!(e.ready);
    assert!(
        e.guidance.contains("embed"),
        "guidance names the command: {}",
        e.guidance
    );
}

#[hegel::test]
fn provenance_nonce_is_stable_and_binds_every_planned_dimension(tc: TestCase) {
    let store = tc.draw(gs::text().max_size(64));
    let model = tc.draw(gs::text().max_size(64));
    let content = tc.draw(gs::text().max_size(512));
    let chunk_id = tc.draw(gs::integers::<i64>());
    let nonce = extraction_provenance_nonce(&store, chunk_id, &model, &content);
    assert_eq!(
        nonce,
        extraction_provenance_nonce(&store, chunk_id, &model, &content)
    );
    assert_ne!(
        nonce,
        extraction_provenance_nonce(&(store.clone() + "\0"), chunk_id, &model, &content)
    );
    assert_ne!(
        nonce,
        extraction_provenance_nonce(&store, chunk_id.wrapping_add(1), &model, &content)
    );
    assert_ne!(
        nonce,
        extraction_provenance_nonce(&store, chunk_id, &(model.clone() + "\0"), &content)
    );
    assert_ne!(
        nonce,
        extraction_provenance_nonce(&store, chunk_id, &model, &(content + "\0"))
    );
}
