use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn status_header_columns_are_separated() {
    let data_dir = std::env::temp_dir().join(format!(
        "graphrag-cli-test-status-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&data_dir).unwrap();
    Command::cargo_bin("graphrag")
        .unwrap()
        .env("GRAPHRAG_DATA_DIR", &data_dir)
        .arg("status")
        .assert()
        .success()
        .stdout(contains("cur/stl/mis/fl cur/stl/mis/fl"));
}
