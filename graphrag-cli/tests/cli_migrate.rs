//! Task 19: `graphrag migrate v2 --plan | --apply`.

use assert_cmd::Command;
use graphrag_core::Database;
use std::path::PathBuf;

fn temp_dir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let d = std::env::temp_dir().join(format!(
        "graphrag-cli-migrate-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn migrate_v2_plans_then_applies() {
    let dir = temp_dir("plan-apply");
    {
        let db = Database::open(&dir.join("graphrag.db")).unwrap();
        db.create_store("conversations", 768).unwrap();
        db.add_chunk("conversations", "an old note", Some("brain"), None)
            .unwrap();
    }
    let paper = dir.join("paper.md");
    std::fs::write(&paper, "# Paper\ntext\n").unwrap();
    let args = |mode: &str| {
        vec![
            "migrate".to_string(),
            "v2".into(),
            mode.into(),
            "--repo-root".into(),
            dir.join("mica").display().to_string(),
            "--dev-root".into(),
            dir.join("dev").display().to_string(),
            "--paper".into(),
            paper.display().to_string(),
        ]
    };
    let out = Command::cargo_bin("graphrag")
        .unwrap()
        .env("GRAPHRAG_DATA_DIR", &dir)
        .args(args("--plan"))
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(plan["per_group"]["conversations"]["single_chunk"], 1);
    assert_eq!(plan["per_group"]["papers"]["from_files"], 1);
    assert_eq!(plan["unresolved"].as_array().unwrap().len(), 0);

    Command::cargo_bin("graphrag")
        .unwrap()
        .env("GRAPHRAG_DATA_DIR", &dir)
        .args(args("--apply"))
        .assert()
        .success();
    let db = Database::open(&dir.join("graphrag.db")).unwrap();
    let stores: Vec<String> = db
        .list_stores()
        .unwrap()
        .into_iter()
        .map(|s| s.name)
        .collect();
    assert_eq!(stores, vec!["memory".to_string()]);
    assert_eq!(db.list_groups("memory").unwrap().len(), 2);
}
