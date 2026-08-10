//! CLI contract for the plan/apply map-reduce surface.
//!
//! These prove the notebook-facing shape: JSONL out of `plan`, JSONL into
//! `apply`, resumable across runs. No model is involved.
use assert_cmd::Command;
use graphrag_core::Database;
use predicates::str::contains;
use serde_json::Value;
use tempfile::TempDir;

fn seeded(n: usize) -> TempDir {
    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("graphrag.db")).unwrap();
    db.create_store("conversations", 3).unwrap();
    for i in 0..n {
        db.add_chunk(
            "conversations",
            &format!("chunk {i}: Alice uses GraphRAG."),
            None,
            None,
        )
        .unwrap();
    }
    dir
}

fn cli(dir: &TempDir) -> Command {
    let mut c = Command::cargo_bin("graphrag").unwrap();
    c.env("GRAPHRAG_DATA_DIR", dir.path());
    c
}

fn lines(out: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(out)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("each stdout line is JSON"))
        .collect()
}

#[test]
fn plan_emits_one_jsonl_work_unit_per_pending_chunk() {
    let dir = seeded(3);
    let out = cli(&dir)
        .args([
            "plan",
            "extract",
            "--store",
            "conversations",
            "--model",
            "gemma4:31b",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let units = lines(&out);
    assert_eq!(units.len(), 3);
    for u in &units {
        assert_eq!(u["kind"], "extract");
        assert_eq!(u["model"], "gemma4:31b");
        assert!(u["chunk_id"].is_i64());
        assert!(
            !u["user"].as_str().unwrap().is_empty(),
            "prompt is populated"
        );
        assert!(u["unit_id"].as_str().unwrap().starts_with("extract:chunk:"));
    }
}

#[test]
fn plan_limit_bounds_output() {
    let dir = seeded(5);
    let out = cli(&dir)
        .args([
            "plan",
            "extract",
            "--store",
            "conversations",
            "--model",
            "m",
            "--limit",
            "2",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(lines(&out).len(), 2);
}

#[test]
fn apply_reads_jsonl_from_stdin_and_checkpoints() {
    let dir = seeded(2);
    let out = cli(&dir)
        .args([
            "plan",
            "extract",
            "--store",
            "conversations",
            "--model",
            "m",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let units = lines(&out);
    let first = units[0]["chunk_id"].as_i64().unwrap();

    let results = format!(
        r#"{{"chunk_id":{first},"model":"m","response":"Alice (Person) -[uses]-> GraphRAG (Software)"}}"#
    );
    cli(&dir)
        .args(["apply", "extract", "--store", "conversations"])
        .write_stdin(results)
        .assert()
        .success()
        .stderr(contains("1"));

    // the applied chunk must not be replanned
    let out2 = cli(&dir)
        .args([
            "plan",
            "extract",
            "--store",
            "conversations",
            "--model",
            "m",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let pending = lines(&out2);
    assert_eq!(pending.len(), 1);
    assert_ne!(pending[0]["chunk_id"].as_i64().unwrap(), first);
}

#[test]
fn plan_apply_roundtrip_drains_to_empty() {
    let dir = seeded(4);
    for _ in 0..4 {
        let out = cli(&dir)
            .args([
                "plan",
                "extract",
                "--store",
                "conversations",
                "--model",
                "m",
                "--limit",
                "1",
            ])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let units = lines(&out);
        if units.is_empty() {
            break;
        }
        let id = units[0]["chunk_id"].as_i64().unwrap();
        let r = format!(
            r#"{{"chunk_id":{id},"model":"m","response":"A (Person) -[knows]-> B (Person)"}}"#
        );
        cli(&dir)
            .args(["apply", "extract", "--store", "conversations"])
            .write_stdin(r)
            .assert()
            .success();
    }
    let out = cli(&dir)
        .args([
            "plan",
            "extract",
            "--store",
            "conversations",
            "--model",
            "m",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert!(lines(&out).is_empty(), "all chunks drained");
}

#[test]
fn apply_tolerates_blank_lines_and_malformed_response_text() {
    let dir = seeded(1);
    let out = cli(&dir)
        .args([
            "plan",
            "extract",
            "--store",
            "conversations",
            "--model",
            "m",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let id = lines(&out)[0]["chunk_id"].as_i64().unwrap();
    let payload =
        format!("\n{{\"chunk_id\":{id},\"model\":\"m\",\"response\":\"%%% junk %%%\"}}\n\n");
    cli(&dir)
        .args(["apply", "extract", "--store", "conversations"])
        .write_stdin(payload)
        .assert()
        .success();
    let out2 = cli(&dir)
        .args([
            "plan",
            "extract",
            "--store",
            "conversations",
            "--model",
            "m",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert!(lines(&out2).is_empty(), "malformed reply still checkpoints");
}

#[test]
fn plan_unknown_store_fails_cleanly() {
    let dir = seeded(1);
    cli(&dir)
        .args(["plan", "extract", "--store", "nope", "--model", "m"])
        .assert()
        .failure();
}
