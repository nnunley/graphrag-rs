//! Task 12: evidence-weighted Leiden projection.

mod common;

use common::*;
use graphrag_core::assertions::Asserter;
use graphrag_core::fingerprints::CommunityConfig;

fn weight(p: &graphrag_core::weights::Projection, a: i64, b: i64) -> Option<f64> {
    let (x, y) = (a.min(b), a.max(b));
    p.edges.iter().find(|e| (e.0, e.1) == (x, y)).map(|e| e.2)
}

/// Three sections each stating `Alice uses GraphRAG`.
fn three_attempts(f: &Fx) {
    let (_, chunks) = f.doc(&sections(3, "s"), &["papers"]);
    for (i, c) in chunks.iter().enumerate() {
        f.commit(
            *c,
            &format!("r{i}"),
            vec![ent("Alice", "person"), ent("GraphRAG", "software")],
            vec![rel("Alice", "GraphRAG", "uses")],
        );
    }
}

#[test]
fn weight_equals_support() {
    let f = fx();
    three_attempts(&f);
    let p =
        f.db.project_graph(STORE, &CommunityConfig::default())
            .unwrap();
    // Not 1.0: a unit-weight mutant fails here.
    assert_eq!(weight(&p, f.id("Alice"), f.id("GraphRAG")), Some(3.0));
}

#[test]
fn zero_weights_exclude_structural_nodes() {
    let f = fx();
    three_attempts(&f);
    let p =
        f.db.project_graph(STORE, &CommunityConfig::default())
            .unwrap();
    let nodes: Vec<i64> = p
        .edges
        .iter()
        .flat_map(|e| [e.0, e.1])
        .chain(p.isolates.iter().copied())
        .collect();
    let kinds: Vec<_> = nodes.iter().map(|n| f.db.node_kind(*n).unwrap()).collect();
    assert!(
        kinds
            .iter()
            .all(|k| *k == graphrag_core::nodes::NodeKind::Entity),
        "{kinds:?}"
    );
}

#[test]
fn asserted_weight_reaches_adjacency() {
    let f = fx();
    let (d, chunks) = f.doc("a note", &["papers"]);
    f.commit(chunks[0], "r", vec![ent("Alice", "person")], vec![]);
    let cfg = CommunityConfig {
        asserted_weight: 0.5,
        ..CommunityConfig::default()
    };
    let p = f.db.project_graph(STORE, &cfg).unwrap();
    let doc = f.db.document_node(d).unwrap();
    let group = f.db.get_or_create_group(STORE, "papers").unwrap();
    assert_eq!(weight(&p, doc, group), Some(0.5));
}

#[test]
fn mention_and_structure_weights_reach_adjacency() {
    let f = fx();
    let (d, chunks) = f.doc("# Only\nAlice here", &[]);
    f.commit(chunks[0], "r", vec![ent("Alice", "person")], vec![]);
    let cfg = CommunityConfig {
        mention_weight: 2.0,
        structure_weight: 0.25,
        ..CommunityConfig::default()
    };
    let p = f.db.project_graph(STORE, &cfg).unwrap();
    let sec = f.db.document_sections(d).unwrap()[0].node_id;
    let doc = f.db.document_node(d).unwrap();
    assert_eq!(weight(&p, sec, f.id("Alice")), Some(2.0));
    assert_eq!(weight(&p, doc, sec), Some(0.25));
}

#[test]
fn same_as_contracts_nodes() {
    let f = fx();
    let (_, chunks) = f.doc(&sections(2, "s"), &[]);
    f.commit(
        chunks[0],
        "a",
        vec![ent("GraphRAG", "software"), ent("Leiden", "method")],
        vec![rel("GraphRAG", "Leiden", "uses")],
    );
    f.commit(
        chunks[1],
        "b",
        vec![ent("Graph RAG", "software"), ent("Rust", "software")],
        vec![rel("Graph RAG", "Rust", "uses")],
    );
    let (a, b) = (f.id("Graph RAG"), f.id("GraphRAG"));
    f.db.merge_as_same(a, b, &Asserter::new("agent:claude").unwrap())
        .unwrap();
    let p =
        f.db.project_graph(STORE, &CommunityConfig::default())
            .unwrap();
    assert_eq!(weight(&p, b, f.id("Leiden")), Some(1.0));
    assert_eq!(
        weight(&p, b, f.id("Rust")),
        Some(1.0),
        "a's edge moved to b"
    );
    assert!(p.edges.iter().all(|e| e.0 != a && e.1 != a));
    assert!(!p.isolates.contains(&a));
}

#[test]
fn retracted_assertion_not_projected() {
    let f = fx();
    let (d, _) = f.doc("a note", &[]);
    let doc = f.db.document_node(d).unwrap();
    let g = f.db.get_or_create_group(STORE, "papers").unwrap();
    let by = Asserter::new("human:norman").unwrap();
    let a =
        f.db.assert_edge(doc, "part_of", g, &by, None, None)
            .unwrap();
    let cfg = CommunityConfig {
        asserted_weight: 1.0,
        ..CommunityConfig::default()
    };
    assert_eq!(
        weight(&f.db.project_graph(STORE, &cfg).unwrap(), doc, g),
        Some(1.0)
    );
    f.db.retract_edge(a, &by).unwrap();
    assert_eq!(
        weight(&f.db.project_graph(STORE, &cfg).unwrap(), doc, g),
        None
    );
}

#[test]
fn isolates_included() {
    let f = fx();
    let (_, chunks) = f.doc("lonely", &[]);
    f.commit(chunks[0], "r", vec![ent("Hermit", "person")], vec![]);
    let p =
        f.db.project_graph(STORE, &CommunityConfig::default())
            .unwrap();
    assert_eq!(p.isolates, vec![f.id("Hermit")]);
    assert_eq!(p.graph.node_count(), 1);
}

#[test]
fn class_edges_need_class_weight() {
    let f = fx();
    let (_, chunks) = f.doc("jj", &[]);
    f.commit(
        chunks[0],
        "r",
        vec![ent_kind("jj", "software", "version control system")],
        vec![],
    );
    let (jj, vcs) = (f.id("jj"), f.id("version control system"));
    let p =
        f.db.project_graph(STORE, &CommunityConfig::default())
            .unwrap();
    assert_eq!(weight(&p, jj, vcs), None);
    let cfg = CommunityConfig {
        class_weight: 0.5,
        ..CommunityConfig::default()
    };
    assert_eq!(
        weight(&f.db.project_graph(STORE, &cfg).unwrap(), jj, vcs),
        Some(0.5)
    );
}

#[test]
fn dormant_items_are_skipped() {
    let f = fx();
    three_attempts(&f);
    f.raw()
        .execute(
            "UPDATE entities SET edge_status = 'dormant' WHERE name = 'Alice'",
            [],
        )
        .unwrap();
    let p =
        f.db.project_graph(STORE, &CommunityConfig::default())
            .unwrap();
    assert!(p.edges.is_empty());
    assert_eq!(p.isolates, vec![f.id("GraphRAG")]);
}

#[test]
fn communities_fp_follows_projection() {
    let f = fx();
    three_attempts(&f);
    let a =
        f.db.graph_input_fp(STORE, &CommunityConfig::default())
            .unwrap();
    let (_, chunks) = f.doc("# x\nmore", &[]);
    f.commit(
        chunks[0],
        "r",
        vec![ent("Alice", "person"), ent("GraphRAG", "software")],
        vec![rel("Alice", "GraphRAG", "uses")],
    );
    let b =
        f.db.graph_input_fp(STORE, &CommunityConfig::default())
            .unwrap();
    assert_ne!(a, b, "support 3 → 4 changes the graph fingerprint");
}
