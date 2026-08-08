use graphrag_core::{Database, GraphRagError, export::export_store};
use serde_json::Value;
use tempfile::TempDir;

fn setup_db(dir: &TempDir) -> Database {
    let db = Database::open(&dir.path().join("t.db")).unwrap();
    db.create_store("s", 3).unwrap();
    let c1 = db
        .add_chunk("s", "c1", Some("src1"), Some("meta1"))
        .unwrap();
    db.set_chunk_embedding(c1, &[1.0, 2.0, 3.0]).unwrap();
    let c2 = db.add_chunk("s", "c2", None, None).unwrap();
    db.set_chunk_embedding(c2, &[4.0, 5.0, 6.0]).unwrap();
    let e1 = db
        .get_or_create_entity("s", "e1", Some("t1"), Some("p1"))
        .unwrap();
    let e2 = db
        .get_or_create_entity("s", "e2", Some("t2"), Some("p2"))
        .unwrap();
    db.add_relation("s", e1, e2, "r1", Some("pr1")).unwrap();
    db.create_community("s", 0, 0.5, None).unwrap();
    db
}

#[test]
fn export_record_order_and_tags() {
    let dir = TempDir::new().unwrap();
    let db = setup_db(&dir);
    let mut buf = Vec::new();
    export_store(&db, "s", &mut buf).unwrap();
    let lines: Vec<&str> = std::str::from_utf8(&buf).unwrap().lines().collect();
    let records: Vec<String> = lines
        .iter()
        .map(|l| {
            serde_json::from_str::<Value>(l).unwrap()["record"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(
        records,
        vec![
            "store",
            "chunk",
            "chunk",
            "entity",
            "entity",
            "relation",
            "community"
        ]
    );
    let vals: Vec<Value> = lines
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(vals[0]["name"], "s");
    assert_eq!(vals[0]["dim"], 3);
    assert_eq!(vals[1]["content"], "c1");
    assert_eq!(vals[1]["source"], "src1");
    assert_eq!(vals[1]["metadata"], "meta1");
    assert_eq!(vals[2]["source"], Value::Null);
    assert_eq!(vals[3]["name"], "e1");
    assert_eq!(vals[3]["entity_type"], "t1");
    assert_eq!(vals[5]["relation"], "r1");
    assert!(vals[5]["head_id"].is_i64());
    assert_eq!(vals[6]["level"], 0);
}

#[test]
fn export_is_deterministic() {
    let dir = TempDir::new().unwrap();
    let db = setup_db(&dir);
    let mut buf1 = Vec::new();
    let mut buf2 = Vec::new();
    export_store(&db, "s", &mut buf1).unwrap();
    export_store(&db, "s", &mut buf2).unwrap();
    assert_eq!(buf1, buf2);
}

#[test]
fn export_unknown_store_errors() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("t.db")).unwrap();
    let mut buf = Vec::new();
    let res = export_store(&db, "unknown", &mut buf);
    match res {
        Err(GraphRagError::StoreNotFound(_)) => (),
        _ => panic!("Expected StoreNotFound error"),
    }
}

#[test]
fn export_empty_store() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("t.db")).unwrap();
    db.create_store("s", 3).unwrap();
    let mut buf = Vec::new();
    export_store(&db, "s", &mut buf).unwrap();
    let lines: Vec<&str> = std::str::from_utf8(&buf).unwrap().lines().collect();
    assert_eq!(lines.len(), 1);
    let val: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(val["record"], "store");
}

#[test]
fn export_excludes_embeddings() {
    let dir = TempDir::new().unwrap();
    let db = setup_db(&dir);
    let mut buf = Vec::new();
    export_store(&db, "s", &mut buf).unwrap();
    let s = std::str::from_utf8(&buf).unwrap();
    assert!(!s.contains("embedding"));
}
