use assert_cmd::Command;
use predicates::str::contains;
use std::path::PathBuf;

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
fn log_shows_recent_notes() {
    let data_dir = temp_data_dir("log_shows_recent");

    Command::cargo_bin("graphrag")
        .unwrap()
        .env("GRAPHRAG_DATA_DIR", &data_dir)
        .args([
            "note",
            "the auto_dream cycle clusters bag-of-words on shell exhaust",
        ])
        .assert()
        .success();

    Command::cargo_bin("graphrag")
        .unwrap()
        .env("GRAPHRAG_DATA_DIR", &data_dir)
        .arg("log")
        .assert()
        .success()
        .stdout(contains("auto_dream cycle"));
}
