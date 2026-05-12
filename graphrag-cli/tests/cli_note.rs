use assert_cmd::Command;
use graphrag_core::Database;
use std::path::PathBuf;

/// Per-test data dir: $TMPDIR/graphrag-cli-test-<test>-<rand>
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

#[test]
fn note_writes_chunk_to_db() {
    let data_dir = temp_data_dir("note_writes_chunk");

    Command::cargo_bin("graphrag")
        .expect("binary exists")
        .env("GRAPHRAG_DATA_DIR", &data_dir)
        .args(["note", "remember to drink water"])
        .assert()
        .success();

    let db = Database::open(&data_dir.join("graphrag.db")).expect("db opens");
    let stats = db
        .store_stats("conversations")
        .expect("default store exists");
    // store_stats returns (chunk_count, entity_count, relation_count)
    assert_eq!(stats.0, 1, "expected exactly one chunk after one note");
}
