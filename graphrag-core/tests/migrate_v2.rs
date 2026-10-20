//! Task 19: merge stores into one graph with groups and sections.

use graphrag_core::Database;
use graphrag_core::migrate_v2::{MigrationSources, apply_v2_migration, plan_v2};
use rusqlite::Connection;
use std::path::Path;
use tempfile::TempDir;

struct Fx {
    dir: TempDir,
    db: Database,
    sources: MigrationSources,
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// A DB with the live source shapes, plus the files they point at.
fn fx() -> Fx {
    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("graphrag.db")).unwrap();
    let mica = dir.path().join("mica");
    let dev = dir.path().join("dev");
    write(
        &mica.join("docs/a.md"),
        "# A1\nalpha one\n# A2\nalpha two\n# A3\nalpha three\n",
    );
    write(&mica.join("docs/b.md"), "# B1\nbeta one\n# B2\nbeta two\n");
    write(
        &dev.join("leit/docs/vision.md"),
        "# Vision\nthe leit vision\n",
    );
    let paper = dir.path().join("paper.md");
    write(
        &paper,
        "# GraphRAG\nThe paper.\n## Method\nLeiden communities.\n",
    );
    for (s, d) in [
        ("conversations", 3),
        ("mica-docs", 3),
        ("operation", 3),
        ("graphrag-paper", 3),
        ("graphrag-paper-dense", 3),
        ("9front-kb", 3),
    ] {
        db.create_store(s, d).unwrap();
    }
    let add = |store: &str, text: &str, source: &str| {
        db.add_chunk(store, text, Some(source), None).unwrap()
    };
    let n1 = add("conversations", "note one about Leiden\n", "brain");
    add("conversations", "note two\n", "brain");
    add(
        "conversations",
        "prime session note",
        "prime-session:019fd75e",
    );
    for (h, t) in [
        ("A1", "alpha one"),
        ("A2", "alpha two"),
        ("A3", "alpha three"),
        ("B1", "beta one"),
        ("B2", "beta two"),
    ] {
        let file = if h.starts_with('A') {
            "docs/a.md"
        } else {
            "docs/b.md"
        };
        add("mica-docs", t, &format!("{file}#{h}"));
    }
    add("operation", "lesson text one", "lesson:retry-budget");
    add("operation", "lesson text two", "lesson:jj-push");
    add(
        "operation",
        "operational text",
        "operational:quota-2026-08-11",
    );
    add("operation", "the leit vision", "leit:vision#0");
    // Out of order on purpose; no file exists for this doc.
    add("operation", "part two", "leit:kernel_handover#2");
    add("operation", "part zero", "leit:kernel_handover#0");
    add("operation", "part one", "leit:kernel_handover#1");
    add(
        "operation",
        "paper copy",
        "graphrag-rs:microsoft-graphrag-paper#0",
    );
    for i in 0..3 {
        add(
            "graphrag-paper",
            &format!("paper chunk {i}"),
            &format!("graphrag-paper:c{i}"),
        );
        add(
            "graphrag-paper-dense",
            &format!("paper chunk {i}"),
            &format!("graphrag-paper:c{i}"),
        );
    }
    add("9front-kb", "plan task", "plan-task-0.3");
    db.set_chunk_embedding(n1, &[1.0, 0.0, 0.0]).unwrap();
    let e = db
        .get_or_create_entity("conversations", "Leiden", Some("method"), None)
        .unwrap();
    let e2 = db
        .get_or_create_entity("conversations", "Louvain", Some("method"), None)
        .unwrap();
    db.add_relation("conversations", e, e2, "refines", None)
        .unwrap();
    db.create_community("conversations", 0, 0.1, None).unwrap();
    let sources = MigrationSources {
        mica_root: mica,
        dev_root: dev,
        paper_path: paper,
        embedder_id: "fake".into(),
        dim: 3,
    };
    Fx { dir, db, sources }
}

impl Fx {
    fn count(&self, sql: &str) -> i64 {
        Connection::open(self.dir.path().join("graphrag.db"))
            .unwrap()
            .query_row(sql, [], |r| r.get(0))
            .unwrap()
    }
    fn docs_in(&self, group: &str) -> Vec<String> {
        let mut out: Vec<String> = self
            .db
            .members_of_group("memory", group)
            .unwrap()
            .into_iter()
            .map(|node| {
                Connection::open(self.dir.path().join("graphrag.db"))
                    .unwrap()
                    .query_row(
                        "SELECT content FROM documents WHERE node_id = ?1",
                        [node],
                        |r| r.get(0),
                    )
                    .unwrap()
            })
            .collect();
        out.sort();
        out
    }
}

#[test]
fn plan_counts_match_fixture() {
    let f = fx();
    let plan = plan_v2(&f.db, &f.sources).unwrap();
    assert!(plan.unresolved.is_empty(), "{:?}", plan.unresolved);
    let g = |name: &str| plan.per_group.get(name).cloned().unwrap_or_default();
    assert_eq!(g("conversations").single_chunk, 3);
    assert_eq!(g("mica-docs").from_files, 2);
    assert_eq!(
        g("operation").single_chunk,
        3,
        "two lessons + one operational"
    );
    assert_eq!(g("operation").from_files, 1, "leit:vision");
    assert_eq!(g("operation").from_chunk_groups, 1, "leit:kernel_handover");
    assert_eq!(g("9front-kb").single_chunk, 1);
    assert_eq!(g("papers").from_files, 1);
    assert_eq!(plan.deletions.paper_chunks, 7, "3 + 3 + the operation copy");
    assert_eq!(plan.deletions.entities, 2);
    assert_eq!(plan.deletions.relations, 1);
    assert_eq!(plan.deletions.communities, 1);
}

#[test]
fn mica_sections_become_section_nodes() {
    let f = fx();
    let plan = plan_v2(&f.db, &f.sources).unwrap();
    apply_v2_migration(&f.db, &plan, &f.sources).unwrap();
    let docs = f.docs_in("mica-docs");
    assert_eq!(docs.len(), 2);
    let ids: Vec<i64> =
        f.db.active_documents("memory")
            .unwrap()
            .into_iter()
            .filter(|d| {
                f.db.document_sections(*d)
                    .map(|s| !s.is_empty())
                    .unwrap_or(false)
            })
            .collect();
    let mica_sections: usize = ids
        .iter()
        .filter(|d| {
            let node = f.db.document_node(**d).unwrap();
            f.db.groups_of(node)
                .unwrap()
                .contains(&"mica-docs".to_string())
        })
        .map(|d| f.db.document_sections(*d).unwrap().len())
        .sum();
    assert_eq!(mica_sections, 5);
}

#[test]
fn operation_doc_without_file_is_reassembled_in_order() {
    let f = fx();
    let plan = plan_v2(&f.db, &f.sources).unwrap();
    apply_v2_migration(&f.db, &plan, &f.sources).unwrap();
    let docs = f.docs_in("operation");
    let handover = docs
        .iter()
        .find(|d| d.contains("part zero"))
        .expect("reassembled doc");
    let (a, b, c) = (
        handover.find("part zero").unwrap(),
        handover.find("part one").unwrap(),
        handover.find("part two").unwrap(),
    );
    assert!(a < b && b < c, "{handover}");
    assert!(
        docs.contains(&"# Vision\nthe leit vision\n".to_string()),
        "file re-imported: {docs:?}"
    );
}

#[test]
fn paper_copies_collapse_to_one_document() {
    let f = fx();
    let plan = plan_v2(&f.db, &f.sources).unwrap();
    apply_v2_migration(&f.db, &plan, &f.sources).unwrap();
    let papers = f.docs_in("papers");
    assert_eq!(papers.len(), 1);
    assert!(papers[0].starts_with("# GraphRAG"));
    assert_eq!(f.count("SELECT COUNT(*) FROM documents WHERE content LIKE 'paper chunk%' OR content = 'paper copy'"), 0);
}

#[test]
fn lessons_stay_single_documents() {
    let f = fx();
    let plan = plan_v2(&f.db, &f.sources).unwrap();
    apply_v2_migration(&f.db, &plan, &f.sources).unwrap();
    let docs = f.docs_in("operation");
    assert!(docs.contains(&"lesson text one".to_string()));
    assert!(docs.contains(&"lesson text two".to_string()));
}

#[test]
fn everything_lands_in_one_store() {
    let f = fx();
    let plan = plan_v2(&f.db, &f.sources).unwrap();
    let report = apply_v2_migration(&f.db, &plan, &f.sources).unwrap();
    assert_eq!(f.count("SELECT COUNT(*) FROM stores"), 1);
    assert_eq!(
        f.count("SELECT COUNT(*) FROM chunks WHERE store != 'memory'"),
        0
    );
    assert_eq!(
        f.count("SELECT COUNT(*) FROM entities WHERE kind = 'entity'"),
        0,
        "clean slate"
    );
    assert_eq!(f.count("SELECT COUNT(*) FROM relations"), 0);
    assert_eq!(f.count("SELECT COUNT(*) FROM communities"), 0);
    assert_eq!(
        f.count("SELECT COUNT(*) FROM documents d WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)"),
        0,
        "no document is left without chunks"
    );
    assert_eq!(f.docs_in("conversations").len(), 3);
    assert!(report.documents > 0);
    assert_eq!(
        f.count("SELECT COUNT(*) FROM assertions WHERE relation = 'part_of' AND asserter = 'agent:migration'"),
        report.documents as i64
    );
}

#[test]
fn unresolved_label_blocks_apply() {
    let f = fx();
    f.db.add_chunk("operation", "mystery", Some("no-pattern-here"), None)
        .unwrap();
    let plan = plan_v2(&f.db, &f.sources).unwrap();
    assert_eq!(
        plan.unresolved,
        vec!["operation: no-pattern-here".to_string()]
    );
    assert!(apply_v2_migration(&f.db, &plan, &f.sources).is_err());
    assert_eq!(f.count("SELECT COUNT(*) FROM stores"), 6, "nothing changed");
}

#[test]
fn stale_plan_is_refused() {
    let f = fx();
    let plan = plan_v2(&f.db, &f.sources).unwrap();
    f.db.add_chunk("conversations", "a late note", Some("brain"), None)
        .unwrap();
    assert!(apply_v2_migration(&f.db, &plan, &f.sources).is_err());
}

#[test]
fn embeddings_reattached_for_unchanged_text() {
    let f = fx();
    let plan = plan_v2(&f.db, &f.sources).unwrap();
    let report = apply_v2_migration(&f.db, &plan, &f.sources).unwrap();
    assert_eq!(report.embeddings_reattached, 1);
    let chunk: i64 = Connection::open(f.dir.path().join("graphrag.db"))
        .unwrap()
        .query_row(
            "SELECT c.id FROM chunk_view c WHERE c.content = 'note one about Leiden'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        f.db.get_chunk_embedding(chunk).unwrap(),
        Some(vec![1.0, 0.0, 0.0])
    );
    assert_eq!(
        f.count(&format!(
            "SELECT COUNT(*) FROM step_status WHERE subject_kind='chunk' AND subject_id={chunk} AND step='embed' AND state='current'"
        )),
        1
    );
}

#[test]
fn apply_is_atomic() {
    let f = fx();
    let plan = plan_v2(&f.db, &f.sources).unwrap();
    let before = (
        f.count("SELECT COUNT(*) FROM stores"),
        f.count("SELECT COUNT(*) FROM chunks"),
        f.count("SELECT COUNT(*) FROM entities"),
        f.count("SELECT COUNT(*) FROM documents"),
    );
    Connection::open(f.dir.path().join("graphrag.db"))
        .unwrap()
        .execute_batch(
            "CREATE TRIGGER fail_paper BEFORE INSERT ON document_adds
             WHEN (SELECT COUNT(*) FROM document_adds) > 5
             BEGIN SELECT RAISE(ABORT, 'injected'); END;",
        )
        .unwrap();
    assert!(apply_v2_migration(&f.db, &plan, &f.sources).is_err());
    let after = (
        f.count("SELECT COUNT(*) FROM stores"),
        f.count("SELECT COUNT(*) FROM chunks"),
        f.count("SELECT COUNT(*) FROM entities"),
        f.count("SELECT COUNT(*) FROM documents"),
    );
    assert_eq!(before, after);
}
