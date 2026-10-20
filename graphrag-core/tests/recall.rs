//! Task 16 (core side): group scoping and the graph expansion lane.

mod common;

use common::*;
use graphrag_core::status::{Step, SubjectKind};

#[test]
fn chunks_in_groups_follow_subgroups() {
    let f = fx();
    let (_, note) = f.doc("a note", &["conversations"]);
    let (_, paper) = f.doc("a paper", &["graphrag"]);
    let g = f.db.get_or_create_group(STORE, "graphrag").unwrap();
    let papers = f.db.get_or_create_group(STORE, "papers").unwrap();
    f.db.assert_edge(
        g,
        "part_of",
        papers,
        &graphrag_core::assertions::Asserter::new("agent:cli").unwrap(),
        None,
        None,
    )
    .unwrap();
    let in_papers = f.db.chunk_ids_in_groups(STORE, &["papers"]).unwrap();
    assert!(in_papers.contains(&paper[0]) && !in_papers.contains(&note[0]));
    let both =
        f.db.chunk_ids_in_groups(STORE, &["papers", "conversations"])
            .unwrap();
    assert_eq!(both.len(), 2);
}

#[test]
fn replaced_documents_drop_out_of_active_chunks() {
    let f = fx();
    let p = graphrag_core::fingerprints::ChunkerProfile::default_markdown();
    let add = |text: &str| {
        f.db.add_document(&graphrag_core::documents::AddDocument {
            store: STORE,
            content: text,
            origin: graphrag_core::documents::Origin::Source,
            source: None,
            doc_key: Some("a.md"),
            title: None,
            groups: &[],
            surface: "cli",
            profile: &p,
        })
        .unwrap()
    };
    add("version one");
    add("version two");
    let active = f.db.active_chunk_ids(STORE).unwrap();
    let texts: Vec<String> = active
        .iter()
        .map(|c| f.db.chunk_text(*c).unwrap())
        .collect();
    assert_eq!(texts, vec!["version two".to_string()]);
}

#[test]
fn expansion_reaches_other_groups_through_shared_entities() {
    let f = fx();
    let (_, note) = f.doc("I tried Leiden clustering today", &["conversations"]);
    let (_, paper) = f.doc("Leiden refines Louvain partitions", &["papers"]);
    let (_, other) = f.doc("Unrelated text about Tokio", &["papers"]);
    f.commit(note[0], "n", vec![ent("Leiden", "method")], vec![]);
    f.commit(
        paper[0],
        "p",
        vec![ent("Leiden", "method"), ent("Louvain", "method")],
        vec![],
    );
    f.commit(other[0], "o", vec![ent("Tokio", "software")], vec![]);
    let exp =
        f.db.expand_recall(STORE, &[note[0]], None, 5, "2026-10-20T12:00:00.000Z")
            .unwrap();
    let ids: Vec<i64> = exp.chunks.iter().map(|c| c.0).collect();
    assert_eq!(ids, vec![paper[0]]);
    assert!(exp.entities.contains(&f.id("Leiden")));
    // A recall use is logged for each entity reached.
    assert_eq!(
        f.count(&format!(
            "SELECT COUNT(*) FROM uses WHERE kind = 'recall' AND item_id = {}",
            f.id("Leiden")
        )),
        1
    );
    // Scoped to conversations, the paper is not offered.
    let scoped = f.db.chunk_ids_in_groups(STORE, &["conversations"]).unwrap();
    let exp =
        f.db.expand_recall(
            STORE,
            &[note[0]],
            Some(&scoped),
            5,
            "2026-10-20T12:00:00.000Z",
        )
        .unwrap();
    assert!(exp.chunks.is_empty());
}

#[test]
fn expansion_skips_dormant_entities() {
    let f = fx();
    let (_, note) = f.doc("Leiden note", &[]);
    let (_, paper) = f.doc("Leiden paper", &[]);
    f.commit(note[0], "n", vec![ent("Leiden", "method")], vec![]);
    f.commit(paper[0], "p", vec![ent("Leiden", "method")], vec![]);
    f.raw()
        .execute("UPDATE entities SET edge_status = 'dormant'", [])
        .unwrap();
    let exp =
        f.db.expand_recall(STORE, &[note[0]], None, 5, "2026-10-20T12:00:00.000Z")
            .unwrap();
    assert!(exp.chunks.is_empty());
}

#[test]
fn rare_shared_entities_rank_above_hubs() {
    let f = fx();
    let (_, seed) = f.doc("seed", &[]);
    f.commit(
        seed[0],
        "s",
        vec![ent("Hub", "concept"), ent("Rare", "concept")],
        vec![],
    );
    let (_, rare) = f.doc("rare match", &[]);
    f.commit(rare[0], "r", vec![ent("Rare", "concept")], vec![]);
    let mut hubs = Vec::new();
    for i in 0..6 {
        let (_, c) = f.doc(&format!("hub {i}"), &[]);
        f.commit(c[0], &format!("h{i}"), vec![ent("Hub", "concept")], vec![]);
        hubs.push(c[0]);
    }
    let exp =
        f.db.expand_recall(STORE, &[seed[0]], None, 3, "2026-10-20T12:00:00.000Z")
            .unwrap();
    assert_eq!(exp.chunks[0].0, rare[0]);
    assert_eq!(exp.chunks.len(), 3);
}

#[test]
fn step_fp_reads_status() {
    let f = fx();
    let (_, c) = f.doc("x", &[]);
    assert_eq!(
        f.db.step_fp(SubjectKind::Chunk, c[0], Step::Embed).unwrap(),
        None
    );
    f.db.record_step(
        SubjectKind::Chunk,
        c[0],
        Step::Embed,
        "sha256:abc",
        graphrag_core::status::StepState::Current,
        None,
    )
    .unwrap();
    assert_eq!(
        f.db.step_fp(SubjectKind::Chunk, c[0], Step::Embed).unwrap(),
        Some(("sha256:abc".to_string(), "current".to_string()))
    );
}

#[test]
fn hybrid_recall_fuses_scopes_and_labels() {
    let f = fx();
    let (_, a) = f.doc("# Topic\nquokka habitats in papers", &["papers"]);
    let (_, b) = f.doc("quokka habitats in chat", &["conversations"]);
    f.db.set_chunk_embedding(a[0], &[1.0, 0.0, 0.0]).unwrap();
    f.db.set_chunk_embedding(b[0], &[0.9, 0.1, 0.0]).unwrap();
    let now = "2026-10-20T12:00:00.000Z";
    let all =
        f.db.hybrid_recall(
            STORE,
            "quokka habitats topic papers",
            &[1.0, 0.0, 0.0],
            &[],
            5,
            None,
            now,
        )
        .unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].chunk_id, a[0]);
    assert!(all[0].vec.is_some() && all[0].lex.is_some());
    assert_eq!(all[0].groups, vec!["papers".to_string()]);
    assert_eq!(all[0].heading_path, vec!["Topic".to_string()]);
    let scoped =
        f.db.hybrid_recall(
            STORE,
            "quokka habitats",
            &[1.0, 0.0, 0.0],
            &["conversations"],
            5,
            None,
            now,
        )
        .unwrap();
    assert_eq!(
        scoped.iter().map(|h| h.chunk_id).collect::<Vec<_>>(),
        vec![b[0]]
    );
}
