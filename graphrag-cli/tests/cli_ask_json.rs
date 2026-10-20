//! Task 19A: machine-readable recall with lane selection, for the eval harness.

use assert_cmd::Command;
use std::path::{Path, PathBuf};

fn temp_dir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let d = std::env::temp_dir().join(format!(
        "graphrag-cli-askjson-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cli(dir: &Path) -> Command {
    let mut c = Command::cargo_bin("graphrag").unwrap();
    c.env("GRAPHRAG_DATA_DIR", dir)
        .env("GRAPHRAG_BIN", "/usr/bin/true");
    c
}

#[test]
fn ask_json_reports_lanes_and_ids() {
    let dir = temp_dir("lanes");
    cli(&dir)
        .args(["note", "the capybara sanctuary opened", "--source", "s1"])
        .assert()
        .success();
    cli(&dir)
        .args([
            "note",
            "an unrelated remark about weather",
            "--source",
            "s2",
        ])
        .assert()
        .success();
    let out = cli(&dir)
        .args(["ask", "capybara sanctuary", "--json", "--lanes", "lexical"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let hits: Vec<serde_json::Value> = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(hits.len(), 1, "only the lexical match: {hits:?}");
    let h = &hits[0];
    assert!(h["chunk_id"].is_i64() && h["document_id"].is_i64());
    assert_eq!(h["source"], "s1");
    assert!(h["lanes"]["lexical"].is_number());
    assert!(h["lanes"]["vector"].is_null() && h["lanes"]["graph"].is_null());
    assert!(h["content"].as_str().unwrap().contains("capybara"));

    let out = cli(&dir)
        .args(["ask", "capybara sanctuary", "--json"])
        .output()
        .unwrap();
    let hits: Vec<serde_json::Value> = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(hits.len(), 2, "the vector lane returns both");
    assert!(hits[0]["lanes"]["vector"].is_number());
}

#[test]
fn unknown_lane_is_rejected() {
    let dir = temp_dir("bad-lane");
    cli(&dir)
        .args(["ask", "x", "--lanes", "telepathy"])
        .assert()
        .failure();
}
