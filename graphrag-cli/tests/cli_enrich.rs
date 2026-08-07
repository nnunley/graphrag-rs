use assert_cmd::Command;
use graphrag_core::Database;
use predicates::str::contains;
use std::path::{Path, PathBuf};

fn temp_data_dir(label: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("graphrag-cli-test-{}-{}-{}", label, pid, nanos));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn seed_graph(data_dir: &Path) -> Database {
    let db = Database::open(&data_dir.join("graphrag.db")).expect("db opens");
    db.create_store("conversations", 768)
        .expect("store created");
    let chunk_id = db
        .add_chunk(
            "conversations",
            "Alice works with Bob on GraphRAG.",
            Some("test"),
            None,
        )
        .expect("chunk added");
    let alice_id = db
        .get_or_create_entity("conversations", "Alice", Some("person"), None)
        .expect("alice entity");
    let bob_id = db
        .get_or_create_entity("conversations", "Bob", Some("person"), None)
        .expect("bob entity");
    let graph_id = db
        .get_or_create_entity("conversations", "GraphRAG", Some("software"), None)
        .expect("graphrag entity");
    db.add_relation("conversations", alice_id, bob_id, "works_with", None)
        .expect("alice bob relation");
    db.add_relation("conversations", bob_id, graph_id, "uses", None)
        .expect("bob graph relation");
    db.link_chunk_entity(chunk_id, alice_id).unwrap();
    db.link_chunk_entity(chunk_id, bob_id).unwrap();
    db.link_chunk_entity(chunk_id, graph_id).unwrap();
    db
}

#[test]
fn enrich_persists_communities_for_existing_entity_graph() {
    let data_dir = temp_data_dir("enrich_persists_communities");
    seed_graph(&data_dir);

    Command::cargo_bin("graphrag")
        .expect("binary exists")
        .env("GRAPHRAG_DATA_DIR", &data_dir)
        .args(["enrich", "--clear"])
        .assert()
        .success()
        .stdout(contains("store=conversations"))
        .stdout(contains("communities="))
        .stdout(contains("entities=3"))
        .stdout(contains("relations=2"));

    let db = Database::open(&data_dir.join("graphrag.db")).expect("db opens");
    let communities = db
        .list_communities("conversations")
        .expect("communities listed");
    assert!(!communities.is_empty(), "expected communities after enrich");
    let linked_entities = db
        .get_community_entities(communities[0].id)
        .expect("community entities listed");
    assert!(
        !linked_entities.is_empty(),
        "expected persisted community links"
    );
}

#[test]
fn enrich_clear_replaces_stale_communities() {
    let data_dir = temp_data_dir("enrich_clear_replaces_stale");
    let db = seed_graph(&data_dir);
    db.create_community("conversations", 42, 0.0, None)
        .expect("stale community");

    Command::cargo_bin("graphrag")
        .expect("binary exists")
        .env("GRAPHRAG_DATA_DIR", &data_dir)
        .args(["enrich", "--clear"])
        .assert()
        .success();

    let db = Database::open(&data_dir.join("graphrag.db")).expect("db opens");
    let communities = db
        .list_communities("conversations")
        .expect("communities listed");
    assert!(
        communities.iter().all(|community| community.level != 42),
        "expected --clear to remove stale communities"
    );
}
