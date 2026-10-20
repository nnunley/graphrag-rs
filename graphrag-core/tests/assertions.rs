//! Task 5: node kinds, retractable assertions and same_as merges.

use graphrag_core::Database;
use graphrag_core::assertions::Asserter;
use graphrag_core::nodes::NodeKind;
use tempfile::TempDir;

fn db() -> (TempDir, Database) {
    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("t.db")).unwrap();
    db.create_store("memory", 3).unwrap();
    (dir, db)
}

fn by(s: &str) -> Asserter {
    Asserter::new(s).unwrap()
}

#[test]
fn assert_is_idempotent_while_active() {
    let (_d, db) = db();
    let doc = db
        .create_node("memory", NodeKind::Document, "doc:1")
        .unwrap();
    let g = db.get_or_create_group("memory", "papers").unwrap();
    let a = db
        .assert_edge(doc, "part_of", g, &by("agent:cli"), None, None)
        .unwrap();
    let b = db
        .assert_edge(doc, "part_of", g, &by("agent:cli"), None, None)
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(db.active_assertions(doc).unwrap().len(), 1);
}

#[test]
fn group_nodes_are_reused() {
    let (_d, db) = db();
    let a = db.get_or_create_group("memory", "papers").unwrap();
    let b = db.get_or_create_group("memory", "papers").unwrap();
    assert_eq!(a, b);
    let e = db.get_entity_by_id(a).unwrap();
    assert_eq!(e.entity_type.as_deref(), Some("group"));
    assert_eq!(db.node_kind(a).unwrap(), NodeKind::Group);
}

#[test]
fn retract_keeps_record_and_hides_edge() {
    let (_d, db) = db();
    let doc = db
        .create_node("memory", NodeKind::Document, "doc:1")
        .unwrap();
    let g = db.get_or_create_group("memory", "papers").unwrap();
    let a = db
        .assert_edge(
            doc,
            "part_of",
            g,
            &by("human:norman"),
            Some("wrong group"),
            None,
        )
        .unwrap();
    db.retract_edge(a, &by("human:norman")).unwrap();
    assert!(db.active_assertions(doc).unwrap().is_empty());
    let rec = db.assertion(a).unwrap();
    assert!(rec.retracted_at.is_some());
    assert_eq!(rec.reason.as_deref(), Some("wrong group"));
}

#[test]
fn reassert_after_retract_creates_new_record() {
    let (_d, db) = db();
    let doc = db
        .create_node("memory", NodeKind::Document, "doc:1")
        .unwrap();
    let g = db.get_or_create_group("memory", "papers").unwrap();
    let a = db
        .assert_edge(doc, "part_of", g, &by("agent:cli"), None, None)
        .unwrap();
    db.retract_edge(a, &by("agent:cli")).unwrap();
    let b = db
        .assert_edge(doc, "part_of", g, &by("agent:cli"), None, None)
        .unwrap();
    assert_ne!(a, b);
    assert_eq!(db.active_assertions(doc).unwrap().len(), 1);
}

#[test]
fn groups_are_transitive() {
    let (_d, db) = db();
    let doc = db
        .create_node("memory", NodeKind::Document, "doc:1")
        .unwrap();
    let graphrag = db.get_or_create_group("memory", "graphrag").unwrap();
    let papers = db.get_or_create_group("memory", "papers").unwrap();
    db.assert_edge(doc, "part_of", graphrag, &by("agent:cli"), None, None)
        .unwrap();
    db.assert_edge(graphrag, "part_of", papers, &by("agent:cli"), None, None)
        .unwrap();
    let mut g = db.groups_of(doc).unwrap();
    g.sort();
    assert_eq!(g, vec!["graphrag".to_string(), "papers".to_string()]);
    let mut members = db.members_of_group("memory", "papers").unwrap();
    members.sort();
    assert_eq!(members, vec![doc]);
}

#[test]
fn document_in_two_groups() {
    let (_d, db) = db();
    let doc = db
        .create_node("memory", NodeKind::Document, "doc:1")
        .unwrap();
    for g in ["papers", "conversations"] {
        let id = db.get_or_create_group("memory", g).unwrap();
        db.assert_edge(doc, "part_of", id, &by("agent:cli"), None, None)
            .unwrap();
    }
    assert_eq!(db.groups_of(doc).unwrap().len(), 2);
    assert_eq!(db.members_of_group("memory", "papers").unwrap(), vec![doc]);
    assert_eq!(
        db.members_of_group("memory", "conversations").unwrap(),
        vec![doc]
    );
}

#[test]
fn group_cycles_terminate() {
    let (_d, db) = db();
    let a = db.get_or_create_group("memory", "a").unwrap();
    let b = db.get_or_create_group("memory", "b").unwrap();
    db.assert_edge(a, "part_of", b, &by("agent:cli"), None, None)
        .unwrap();
    db.assert_edge(b, "part_of", a, &by("agent:cli"), None, None)
        .unwrap();
    let mut g = db.groups_of(a).unwrap();
    g.sort();
    assert_eq!(g, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn asserter_must_be_prefixed() {
    assert!(Asserter::new("norman").is_err());
    assert!(Asserter::new("human:").is_err());
    assert!(Asserter::new("robot:x").is_err());
    assert!(Asserter::new("human:norman").is_ok());
    assert!(Asserter::new("agent:structure").is_ok());
}

#[test]
fn context_is_stored_and_optional() {
    let (_d, db) = db();
    let x = db.create_node("memory", NodeKind::Entity, "x").unwrap();
    let y = db.create_node("memory", NodeKind::Entity, "y").unwrap();
    let a = db
        .assert_edge(x, "subject", y, &by("agent:claude"), None, Some("mt:work"))
        .unwrap();
    let b = db
        .assert_edge(x, "subject", y, &by("agent:claude"), None, None)
        .unwrap();
    assert_ne!(a, b, "a different context is a different assertion");
    assert_eq!(db.assertion(a).unwrap().context.as_deref(), Some("mt:work"));
    assert_eq!(db.assertion(b).unwrap().context, None);
}

#[test]
fn same_as_resolves_and_retract_unmerges() {
    let (_d, db) = db();
    let a = db
        .get_or_create_entity("memory", "GraphRAG", Some("software"), None)
        .unwrap();
    let b = db
        .get_or_create_entity("memory", "Graph RAG", Some("software"), None)
        .unwrap();
    let other = db
        .get_or_create_entity("memory", "Leiden", None, None)
        .unwrap();
    db.add_relation("memory", a, other, "uses", None).unwrap();
    let m = db.merge_as_same(a, b, &by("agent:claude")).unwrap();
    assert_eq!(db.resolve_entity(a).unwrap(), b);
    assert_eq!(db.resolve_entity(b).unwrap(), b);
    db.retract_edge(m, &by("agent:claude")).unwrap();
    assert_eq!(db.resolve_entity(a).unwrap(), a);
    assert_eq!(
        db.get_relations_for_entity(a).unwrap().len(),
        1,
        "a's rows are intact"
    );
}

#[test]
fn same_as_chain_resolves_to_end() {
    let (_d, db) = db();
    let a = db.get_or_create_entity("memory", "a", None, None).unwrap();
    let b = db.get_or_create_entity("memory", "b", None, None).unwrap();
    let c = db.get_or_create_entity("memory", "c", None, None).unwrap();
    db.merge_as_same(a, b, &by("agent:claude")).unwrap();
    db.merge_as_same(b, c, &by("agent:claude")).unwrap();
    assert_eq!(db.resolve_entity(a).unwrap(), c);
}

#[test]
fn same_as_cycle_rejected() {
    let (_d, db) = db();
    let a = db.get_or_create_entity("memory", "a", None, None).unwrap();
    let b = db.get_or_create_entity("memory", "b", None, None).unwrap();
    db.merge_as_same(a, b, &by("agent:claude")).unwrap();
    assert!(db.merge_as_same(b, a, &by("agent:claude")).is_err());
    assert!(db.merge_as_same(a, a, &by("agent:claude")).is_err());
}

#[test]
fn merge_entities_is_a_same_as_wrapper() {
    let (_d, db) = db();
    let a = db.get_or_create_entity("memory", "a", None, None).unwrap();
    let b = db.get_or_create_entity("memory", "b", None, None).unwrap();
    db.merge_entities(b, a).unwrap();
    assert_eq!(db.resolve_entity(a).unwrap(), b);
    assert!(
        db.get_entity_by_id(a).is_ok(),
        "merge no longer deletes the source"
    );
}

#[test]
fn list_groups_counts_documents() {
    let (_d, db) = db();
    let a = db
        .create_node("memory", NodeKind::Document, "doc:a")
        .unwrap();
    let b = db
        .create_node("memory", NodeKind::Document, "doc:b")
        .unwrap();
    let papers = db.get_or_create_group("memory", "papers").unwrap();
    let notes = db.get_or_create_group("memory", "notes").unwrap();
    for d in [a, b] {
        db.assert_edge(d, "part_of", papers, &by("agent:cli"), None, None)
            .unwrap();
    }
    db.assert_edge(a, "part_of", notes, &by("agent:cli"), None, None)
        .unwrap();
    assert_eq!(
        db.list_groups("memory").unwrap(),
        vec![("notes".to_string(), 1), ("papers".to_string(), 2)]
    );
}
