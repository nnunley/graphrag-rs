//! Task 8: Dublin Core vocabulary additions and name normalization.

use graphrag_core::Database;
use graphrag_core::synonyms::{
    RelationMapping, canonical_relations, normalize_entity_name, normalize_entity_type,
    normalize_relation,
};
use tempfile::TempDir;

#[test]
fn synonyms_map_to_one_canonical() {
    for raw in [
        "utilizes",
        "leverages",
        "uses",
        "Uses",
        "makes use of",
        "makes-use-of",
    ] {
        assert_eq!(
            normalize_relation(raw),
            RelationMapping::Canonical("uses"),
            "{raw}"
        );
    }
}

#[test]
fn subject_is_canonical() {
    assert!(canonical_relations().contains(&"subject"));
    assert_eq!(
        normalize_relation("about"),
        RelationMapping::Canonical("subject")
    );
    assert_eq!(
        normalize_relation("topic_of"),
        RelationMapping::Canonical("subject")
    );
    assert_eq!(
        normalize_relation("subject"),
        RelationMapping::Canonical("subject")
    );
}

#[test]
fn group_is_its_own_type() {
    assert_eq!(normalize_entity_type("group"), "group");
    assert_eq!(normalize_entity_type("Group"), "group");
    assert_eq!(normalize_entity_type("section"), "section");
}

#[test]
fn collection_is_not_dataset() {
    assert_eq!(normalize_entity_type("collection"), "collection");
    assert_eq!(normalize_entity_type("corpus"), "dataset");
}

#[test]
fn unknown_type_is_other() {
    assert_eq!(normalize_entity_type("spaceship"), "other");
    assert_eq!(normalize_entity_type("Software"), "software");
    assert_eq!(normalize_entity_type("library"), "software");
}

#[test]
fn isa_and_genls_are_distinct() {
    assert_eq!(
        normalize_relation("instance_of"),
        RelationMapping::Canonical("isa")
    );
    assert_eq!(
        normalize_relation("is a"),
        RelationMapping::Canonical("isa")
    );
    assert_eq!(
        normalize_relation("type_of"),
        RelationMapping::Canonical("isa")
    );
    assert_eq!(
        normalize_relation("kind_of"),
        RelationMapping::Canonical("genls")
    );
    assert_eq!(
        normalize_relation("subclass_of"),
        RelationMapping::Canonical("genls")
    );
    assert_eq!(
        normalize_relation("subtype_of"),
        RelationMapping::Canonical("genls")
    );
    assert!(!canonical_relations().contains(&"type_of"));
}

#[test]
fn reserved_relations_are_canonical() {
    for r in ["disjoint_with", "links_to", "same_as"] {
        assert_eq!(normalize_relation(r), RelationMapping::Canonical(r));
        assert!(canonical_relations().contains(&r));
    }
}

#[test]
fn unknown_relation_is_unmapped() {
    assert_eq!(
        normalize_relation("Frobnicates With"),
        RelationMapping::Unmapped("frobnicates_with".into())
    );
}

#[test]
fn spelling_variants_merge() {
    assert_eq!(
        normalize_entity_name("GraphRAG"),
        normalize_entity_name("graphrag ")
    );
    assert_eq!(
        normalize_entity_name("GRAPHRAG"),
        normalize_entity_name("  GraphRag")
    );
    assert_eq!(
        normalize_entity_name("Leiden   algorithm"),
        "leiden algorithm"
    );

    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("t.db")).unwrap();
    db.create_store("memory", 3).unwrap();
    let a = db
        .get_or_create_entity("memory", "GraphRAG", None, None)
        .unwrap();
    let b = db
        .get_or_create_entity("memory", "graphrag ", None, None)
        .unwrap();
    let c = db
        .get_or_create_entity("memory", "GRAPHRAG", None, None)
        .unwrap();
    let d = db
        .get_or_create_entity("memory", "Leiden  algorithm", None, None)
        .unwrap();
    let e = db
        .get_or_create_entity("memory", "leiden algorithm", None, None)
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert_eq!(d, e);
    assert_eq!(
        db.get_entity_by_id(a).unwrap().name,
        "GraphRAG",
        "first display form kept"
    );
    assert_eq!(db.get_entity_by_id(d).unwrap().name, "Leiden algorithm");
}

#[test]
fn extracted_mentions_still_fold_into_references() {
    assert_eq!(
        normalize_relation("mentions"),
        RelationMapping::Canonical("references")
    );
}
