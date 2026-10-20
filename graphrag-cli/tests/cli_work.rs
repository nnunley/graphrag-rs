//! Task 16: work/status commands, group-scoped note and ask.

use assert_cmd::Command;
use graphrag_core::Database;
use graphrag_core::attempts::{AttemptOutcome, AttemptRecord, ExtractedEntity};
use graphrag_core::fingerprints::extract_fp;
use graphrag_core::paper::ExtractionAttemptIdentityParts;
use predicates::str::contains;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const STORE: &str = "memory";

fn temp_dir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "graphrag-cli-work-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A stand-in worker binary that records its argv.
fn stub_bin(dir: &Path) -> PathBuf {
    let bin = dir.join("stub-graphrag.sh");
    let log = dir.join("spawned.txt");
    std::fs::write(
        &bin,
        format!("#!/bin/sh\necho \"$@\" >> '{}'\n", log.display()),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    bin
}

fn cli(dir: &Path) -> Command {
    let mut c = Command::cargo_bin("graphrag").unwrap();
    c.env("GRAPHRAG_DATA_DIR", dir)
        .env("GRAPHRAG_BIN", stub_bin(dir));
    c
}

fn db(dir: &Path) -> Database {
    Database::open(&dir.join("graphrag.db")).unwrap()
}

fn groups_of_only_doc(dir: &Path) -> Vec<String> {
    let d = db(dir);
    let docs = d.active_documents(STORE).unwrap();
    assert_eq!(docs.len(), 1);
    d.groups_of(d.document_node(docs[0]).unwrap()).unwrap()
}

#[test]
fn note_defaults_to_conversations_group() {
    let dir = temp_dir("note-default");
    cli(&dir)
        .args(["note", "remember to drink water"])
        .assert()
        .success();
    assert_eq!(groups_of_only_doc(&dir), vec!["conversations".to_string()]);
}

#[test]
fn note_with_two_groups() {
    let dir = temp_dir("note-groups");
    cli(&dir)
        .args([
            "note",
            "a paper note",
            "--group",
            "papers",
            "--group",
            "wiki",
        ])
        .assert()
        .success();
    assert_eq!(
        groups_of_only_doc(&dir),
        vec!["papers".to_string(), "wiki".to_string()]
    );
}

#[test]
fn note_spawns_worker() {
    let dir = temp_dir("note-spawn");
    cli(&dir)
        .args(["note", "spawn a worker please"])
        .assert()
        .success();
    let log = dir.join("spawned.txt");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !log.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(std::fs::read_to_string(&log).unwrap().contains("work"));
}

#[test]
fn note_is_embedded_inline() {
    let dir = temp_dir("note-embedded");
    cli(&dir)
        .args(["note", "embedded right away"])
        .assert()
        .success();
    let conn = rusqlite::Connection::open(dir.join("graphrag.db")).unwrap();
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM step_status WHERE subject_kind='chunk' AND step='embed' AND state='current'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 1);
}

#[test]
fn ask_in_scopes_results() {
    let dir = temp_dir("ask-scope");
    cli(&dir)
        .args(["note", "zebra crossings in papers", "--group", "papers"])
        .assert()
        .success();
    cli(&dir)
        .args(["note", "zebra crossings in chat"])
        .assert()
        .success();
    let out = cli(&dir)
        .args(["ask", "zebra crossings", "--in", "papers"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success());
    assert!(stdout.contains("in papers"), "{stdout}");
    assert!(!stdout.contains("in chat"), "{stdout}");
    assert!(stdout.contains("papers"), "groups shown");
}

#[test]
fn ask_escalates_incomplete_hits() {
    let dir = temp_dir("ask-escalate");
    cli(&dir)
        .args(["note", "escalate me when recalled"])
        .assert()
        .success();
    cli(&dir)
        .args(["ask", "escalate recalled"])
        .assert()
        .success();
    let conn = rusqlite::Connection::open(dir.join("graphrag.db")).unwrap();
    let p: i64 = conn
        .query_row("SELECT MAX(priority) FROM chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(p, 2);
}

#[test]
fn store_flag_warns_and_aliases() {
    let dir = temp_dir("store-alias");
    cli(&dir)
        .args(["note", "aliased store note", "--store", "papers"])
        .assert()
        .success()
        .stderr(contains("warning: --store is deprecated; use --in"));
    assert_eq!(groups_of_only_doc(&dir), vec!["papers".to_string()]);
}

fn commit(d: &Database, chunk: i64, names: &[&str]) {
    let text = d.chunk_text(chunk).unwrap();
    let fp = extract_fp(&ExtractionAttemptIdentityParts {
        store: STORE,
        chunk_id: chunk,
        chunk_content: &text,
        model: "m",
        extraction_profile: "paper-json",
        prompt_version: "v1",
        response_schema_version: "graph-v1",
        glean_max_rounds: 0,
    })
    .unwrap();
    let entities = names
        .iter()
        .map(|n| ExtractedEntity {
            name: n.to_string(),
            entity_type: "method".into(),
            description: String::new(),
            kind: None,
            is_class: false,
            broader: None,
        })
        .collect();
    d.commit_attempt(&AttemptRecord {
        chunk_id: chunk,
        attempt_fp: &fp,
        model: "m",
        profile: "paper-json",
        prompt_version: "v1",
        schema_version: "graph-v1",
        glean_max_rounds: 0,
        raw_response: &names.join(","),
        outcome: AttemptOutcome::Ok {
            entities,
            relations: vec![],
        },
        rejected_entities: 0,
    })
    .unwrap();
}

#[test]
fn unscoped_ask_reaches_paper_through_shared_entity() {
    let dir = temp_dir("ask-expand");
    cli(&dir)
        .args(["note", "my notebook entry about clustering experiments"])
        .assert()
        .success();
    // Distractors that match the query wording, so the paper is not a seed.
    for i in 0..4 {
        cli(&dir)
            .args(["note", &format!("clustering run {i} finished")])
            .assert()
            .success();
    }
    cli(&dir)
        .args([
            "note",
            "Refinement guarantees well-connected partitions.",
            "--group",
            "papers",
        ])
        .assert()
        .success();
    {
        let d = db(&dir);
        let chunks = d.list_chunks(STORE).unwrap();
        let id_of = |needle: &str| {
            chunks
                .iter()
                .find(|c| c.content.contains(needle))
                .unwrap()
                .id
        };
        commit(&d, id_of("my notebook entry"), &["Leiden"]);
        commit(&d, id_of("well-connected"), &["Leiden"]);
    }
    let out = cli(&dir)
        .args(["ask", "notebook entry clustering experiments"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let paper_line = stdout
        .lines()
        .find(|l| l.contains("well-connected partitions"))
        .unwrap_or_else(|| panic!("paper chunk missing:\n{stdout}"));
    assert!(
        !paper_line.contains("graph=-"),
        "reached through the expansion lane: {paper_line}"
    );

    let out = cli(&dir)
        .args([
            "ask",
            "notebook entry clustering experiments",
            "--in",
            "conversations",
        ])
        .output()
        .unwrap();
    assert!(!String::from_utf8_lossy(&out.stdout).contains("well-connected partitions"));
}

#[test]
fn status_json_has_per_group_counts() {
    let dir = temp_dir("status-json");
    cli(&dir)
        .args(["note", "status note one"])
        .assert()
        .success();
    cli(&dir)
        .args(["note", "status note two", "--group", "papers"])
        .assert()
        .success();
    let out = cli(&dir).args(["status", "--json"]).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["report"]["per_group"]["conversations"].is_object(), "{v}");
    assert!(v["report"]["per_group"]["papers"].is_object());
    assert_eq!(v["report"]["total"]["current"]["embed"], 2);
    assert!(v["worker_running"].is_boolean());
}

#[test]
fn work_exit_code_passthrough() {
    let dir = temp_dir("work-69");
    cli(&dir)
        .args(["note", "needs extraction"])
        .assert()
        .success();
    cli(&dir)
        .env("GRAPHRAG_LLM_BASE_URL", "http://127.0.0.1:1/v1")
        .args(["work"])
        .assert()
        .code(69);
}

#[test]
fn enrich_is_a_deprecated_alias_for_work() {
    let dir = temp_dir("enrich-alias");
    cli(&dir)
        .args(["note", "needs extraction"])
        .assert()
        .success();
    cli(&dir)
        .env("GRAPHRAG_LLM_BASE_URL", "http://127.0.0.1:1/v1")
        .args(["enrich"])
        .assert()
        .code(69)
        .stderr(contains("deprecated"));
}

#[test]
fn work_pause_and_resume() {
    let dir = temp_dir("work-pause");
    cli(&dir).args(["work", "--pause"]).assert().success();
    assert!(dir.join("worker.pause").exists());
    cli(&dir).args(["work", "--resume"]).assert().success();
    assert!(!dir.join("worker.pause").exists());
}
