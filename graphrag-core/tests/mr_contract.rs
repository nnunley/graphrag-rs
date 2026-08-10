//! Contract tests for the map-reduce plan/apply split.
//!
//! `plan` emits pending work units; an external executor runs them; `apply`
//! ingests results. Both halves are pure data transforms — no model required.
use graphrag_core::Database;
use graphrag_core::db::EntityInput;
use graphrag_core::mr::{
    ExtractPrompt, ExtractResult, ExtractionStrategy, SummaryResult, WorkUnit, apply_extract,
    apply_summarize, plan_extract, plan_summarize,
};
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
            response: "".into(),
        }],
    )
    .unwrap();
    assert_eq!(n, 0);
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
        response: "Alice (Person) -[uses]-> GraphRAG (Software)".into(),
    };
    apply_extract(&db, "s", S, std::slice::from_ref(&r)).unwrap();
    let before = db.list_entities("s").unwrap().len();
    apply_extract(&db, "s", S, std::slice::from_ref(&r)).unwrap();
    assert_eq!(
        db.list_entities("s").unwrap().len(),
        before,
        "re-apply must not duplicate"
    );
}

#[test]
fn apply_accepts_partial_and_out_of_order_results() {
    let (_d, db) = db_with_chunks(4);
    let units = plan_extract(&db, "s", "m", S, None).unwrap();
    let mk = |c: i64| ExtractResult {
        chunk_id: c,
        model: "m".into(),
        response: "A (Person) -[knows]-> B (Person)".into(),
    };
    // only units 3 and 1, reversed — a crashed/partial executor run
    apply_extract(&db, "s", S, &[mk(units[3].chunk_id), mk(units[1].chunk_id)]).unwrap();
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
