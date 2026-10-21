//! Task 12A: generalization pass with class canonicalization and
//! G²-admitted lifted edges.

mod common;

use common::*;
use graphrag_core::generalize::{LiftConfig, base_form};
use graphrag_core::synonyms::normalize_entity_name;

fn cfg(ic_floor: f64) -> LiftConfig {
    LiftConfig {
        min_pairs: 3,
        g2_min: 10.83,
        ic_floor,
    }
}

/// Fake embedder: one-hot on a hash of the last word, so names sharing
/// their last word embed identically and others are orthogonal.
fn embed(s: &str) -> Vec<f32> {
    let last = s.split_whitespace().last().unwrap_or("").to_lowercase();
    let h = last
        .bytes()
        .fold(7u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
    let mut v = vec![0.0; 64];
    v[(h % 64) as usize] = 1.0;
    v
}

#[test]
fn base_form_singularizes_head_noun_with_exceptions() {
    assert_eq!(base_form("llms"), "llm");
    assert_eq!(base_form("LLMs"), "LLM");
    assert_eq!(base_form("knowledge graphs"), "knowledge graph");
    assert_eq!(base_form("queries"), "query");
    assert_eq!(base_form("classes"), "class");
    assert_eq!(base_form("indexes"), "index");
    assert_eq!(base_form("Été"), "Été");
    assert_eq!(base_form("Knowledge Graphs"), "Knowledge Graph");
    for keep in [
        "kubernetes",
        "nixos",
        "redis",
        "analysis",
        "status",
        "Postgres",
        "node.js",
        "aws",
    ] {
        assert_eq!(base_form(keep), keep, "{keep}");
    }
    assert_eq!(
        normalize_entity_name("Knowledge Graphs"),
        normalize_entity_name("knowledge graph")
    );
}

#[test]
fn plural_variants_merge_into_one_entity() {
    let f = fx();
    let (_, c) = f.doc(&sections(2, "s"), &[]);
    f.commit(c[0], "a", vec![ent("LLMs", "concept")], vec![]);
    f.commit(c[1], "b", vec![ent("LLM", "concept")], vec![]);
    assert_eq!(
        f.count("SELECT COUNT(*) FROM entities WHERE kind='entity'"),
        1
    );
}

#[test]
fn similar_kind_maps_to_existing_class() {
    let f = fx();
    let crate_class =
        f.db.resolve_class_similar(STORE, "crate", &embed, 0.92)
            .unwrap();
    let rust_crate =
        f.db.resolve_class_similar(STORE, "rust crate", &embed, 0.92)
            .unwrap();
    assert_eq!(rust_crate, crate_class);
    // The alias now answers exact resolution too (what commit_attempt uses).
    assert_eq!(
        f.db.resolve_class(STORE, "rust crate").unwrap(),
        crate_class
    );
}

#[test]
fn new_kind_creates_live_class() {
    let f = fx();
    let a =
        f.db.resolve_class_similar(STORE, "version control system", &embed, 0.92)
            .unwrap();
    let b =
        f.db.resolve_class_similar(STORE, "proxy service", &embed, 0.92)
            .unwrap();
    assert_ne!(a, b);
    let e = f.db.get_entity_by_id(b).unwrap();
    assert_eq!(e.entity_type.as_deref(), Some("collection"));
}

#[test]
fn known_classes_rank_by_members_and_include_document_classes() {
    let f = fx();
    let (_, c) = f.doc(&sections(3, "k"), &[]);
    f.commit(
        c[0],
        "a",
        vec![
            ent_kind("jj", "software", "version control system"),
            ent_kind("git", "software", "version control system"),
        ],
        vec![],
    );
    f.commit(
        c[1],
        "b",
        vec![ent_kind("nginx", "software", "proxy service")],
        vec![],
    );
    let (d2, c2) = f.doc("# other\nhg here", &[]);
    f.commit(
        c2[0],
        "c",
        vec![ent_kind("hg", "software", "rare class")],
        vec![],
    );
    let top = f.db.known_classes(STORE, None, 1).unwrap();
    assert_eq!(top, vec!["version control system".to_string()]);
    let with_doc = f.db.known_classes(STORE, Some(d2), 1).unwrap();
    assert!(with_doc.contains(&"rare class".to_string()));
    assert!(with_doc.contains(&"version control system".to_string()));
}

#[test]
fn relation_hierarchy_is_seeded() {
    let f = fx();
    assert_eq!(
        f.db.relation_broader("pushes_to").unwrap().as_deref(),
        Some("exchanges_data_with")
    );
    assert_eq!(
        f.db.relation_broader("chosen_over").unwrap().as_deref(),
        Some("compared_with")
    );
    f.db.set_relation_broader("mirrors_to", "exchanges_data_with", "llm")
        .unwrap();
    assert_eq!(
        f.db.relation_broader("mirrors_to").unwrap().as_deref(),
        Some("exchanges_data_with")
    );
    assert_eq!(f.db.relation_broader("uses").unwrap(), None);
}

/// jj→GitHub, git→Forgejo, hg→Bitbucket `pushes_to`, plus unrelated noise.
fn vcs_graph(f: &Fx) {
    let (_, c) = f.doc(&sections(16, "v"), &[]);
    let pairs = [("jj", "GitHub"), ("git", "Forgejo"), ("hg", "Bitbucket")];
    for (i, (a, b)) in pairs.iter().enumerate() {
        f.commit(
            c[i],
            &format!("p{i}"),
            vec![
                ent_kind(a, "software", "version control system"),
                ent_kind(b, "software", "code hosting service"),
            ],
            vec![rel(a, b, "pushes_to")],
        );
    }
    for i in 0..10 {
        let (a, b) = (format!("person{i}"), format!("city{i}"));
        f.commit(
            c[3 + i],
            &format!("n{i}"),
            vec![
                ent_kind(&a, "person", "person"),
                ent_kind(&b, "place", "city"),
            ],
            vec![rel(&a, &b, "located_in")],
        );
    }
}

#[test]
fn repeated_instance_relation_lifts_and_is_admitted() {
    let f = fx();
    vcs_graph(&f);
    let report = f.db.lift(STORE, &cfg(0.0)).unwrap();
    assert!(report.admitted >= 1, "{report:?}");
    let (pairs, admitted): (i64, bool) = f
        .raw()
        .query_row(
            "SELECT l.pairs, l.admitted FROM lifted_edges l
             JOIN entities h ON h.id = l.head_class JOIN entities t ON t.id = l.tail_class
             WHERE l.relation = 'exchanges_data_with' AND h.name = 'version control system'
               AND t.name = 'code hosting service'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(pairs, 3);
    assert!(admitted);
    assert_eq!(
        f.count("SELECT COUNT(*) FROM lifted_from lf JOIN lifted_edges l ON l.id = lf.lifted_id WHERE l.relation = 'exchanges_data_with'"),
        3,
        "derived_from links to the three instance relations"
    );
}

#[test]
fn generic_everything_edge_not_admitted() {
    let f = fx();
    let (_, c) = f.doc(&sections(20, "g"), &[]);
    for (i, chunk) in c.iter().enumerate() {
        let (a, b) = (format!("tool{i}"), format!("tool{}", (i + 1) % 20));
        f.commit(
            *chunk,
            &format!("g{i}"),
            vec![
                ent_kind(&a, "software", "software"),
                ent_kind(&b, "software", "software"),
            ],
            vec![rel(&a, &b, "uses")],
        );
    }
    f.db.lift(STORE, &cfg(0.0)).unwrap();
    let admitted: bool = f
        .raw()
        .query_row(
            "SELECT l.admitted FROM lifted_edges l JOIN entities h ON h.id = l.head_class
             WHERE l.relation = 'uses' AND h.name = 'software'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(!admitted, "observed ≈ expected is chance, not structure");
}

#[test]
fn support_is_distinct_pairs() {
    let f = fx();
    let (_, c) = f.doc(&sections(5, "d"), &[]);
    for (i, chunk) in c.iter().enumerate() {
        f.commit(
            *chunk,
            &format!("d{i}"),
            vec![
                ent_kind("jj", "software", "version control system"),
                ent_kind("GitHub", "software", "code hosting service"),
            ],
            vec![rel("jj", "GitHub", "pushes_to")],
        );
    }
    f.db.lift(STORE, &cfg(0.0)).unwrap();
    let pairs: i64 = f
        .raw()
        .query_row(
            "SELECT pairs FROM lifted_edges WHERE relation = 'pushes_to'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(pairs, 1);
}

#[test]
fn ic_floor_stops_lifting_at_root() {
    let f = fx();
    vcs_graph(&f);
    // Every class gets a common root ancestor.
    for class in [
        "version control system",
        "code hosting service",
        "person",
        "city",
    ] {
        let (_, c) = f.doc(&format!("# {class}\nroot"), &[]);
        let mut e = ent(class, "concept");
        e.is_class = true;
        e.broader = Some("thing".into());
        f.commit(c[0], class, vec![e], vec![]);
    }
    let thing = f.id("thing");
    assert!(f.db.information_content(thing).unwrap() < 0.01);
    f.db.lift(STORE, &cfg(0.5)).unwrap();
    assert_eq!(
        f.count(&format!(
            "SELECT COUNT(*) FROM lifted_edges WHERE head_class = {thing} OR tail_class = {thing}"
        )),
        0
    );
    assert!(f.count("SELECT COUNT(*) FROM lifted_edges") > 0);
}

#[test]
fn lift_is_incremental() {
    let f = fx();
    vcs_graph(&f);
    f.db.lift(STORE, &cfg(0.0)).unwrap();
    let before: Vec<(i64, String)> = {
        let conn = f.raw();
        let mut s = conn
            .prepare(
                "SELECT id, updated_at FROM lifted_edges WHERE relation = 'located_in' ORDER BY id",
            )
            .unwrap();
        s.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    std::thread::sleep(std::time::Duration::from_millis(20));
    let (_, c) = f.doc("# extra\nsvn", &[]);
    f.commit(
        c[0],
        "x",
        vec![
            ent_kind("svn", "software", "version control system"),
            ent_kind("SourceForge", "software", "code hosting service"),
        ],
        vec![rel("svn", "SourceForge", "pushes_to")],
    );
    let r = f.db.lift(STORE, &cfg(0.0)).unwrap();
    assert!(r.candidates > 0);
    let after: Vec<(i64, String)> = {
        let conn = f.raw();
        let mut s = conn
            .prepare(
                "SELECT id, updated_at FROM lifted_edges WHERE relation = 'located_in' ORDER BY id",
            )
            .unwrap();
        s.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    assert_eq!(before, after, "unaffected candidates are not rewritten");
    let pairs: i64 = f
        .raw()
        .query_row(
            "SELECT l.pairs FROM lifted_edges l JOIN entities h ON h.id = l.head_class
             WHERE l.relation = 'exchanges_data_with' AND h.name = 'version control system'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(pairs, 4);
}

#[test]
fn no_closure_rows_stored() {
    let f = fx();
    vcs_graph(&f);
    let relations = f.count("SELECT COUNT(*) FROM relations");
    let assertions = f.count("SELECT COUNT(*) FROM assertions");
    f.db.lift(STORE, &cfg(0.0)).unwrap();
    assert_eq!(f.count("SELECT COUNT(*) FROM relations"), relations);
    assert_eq!(f.count("SELECT COUNT(*) FROM assertions"), assertions);
}

#[test]
fn generalize_step_is_recorded() {
    let f = fx();
    vcs_graph(&f);
    let fp = f.db.generalize_input_fp(STORE, &cfg(0.0)).unwrap();
    f.db.lift(STORE, &cfg(0.0)).unwrap();
    let stored: String = f
        .raw()
        .query_row(
            "SELECT fp FROM step_status WHERE subject_kind='graph' AND step='generalize' AND state='current'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stored, fp);
}

#[test]
fn multibyte_names_do_not_panic() {
    for name in ["café", "知识图谱", "naïves", "Ωmegas", "emoji 🚀s", "Ärzte"] {
        let _ = base_form(name);
        let _ = normalize_entity_name(name);
    }
    assert_eq!(base_form("cafés"), "café");
}

#[test]
fn singularizing_is_case_independent() {
    for (a, b) in [
        ("Workflows", "workflow"),
        ("WORKFLOWS", "workflows"),
        ("Embeddings", "embedding"),
        ("Llms", "LLMs"),
    ] {
        assert_eq!(
            normalize_entity_name(a),
            normalize_entity_name(b),
            "{a} vs {b}"
        );
    }
    for keep in ["Postgres", "Kubernetes", "Redis", "Pandas", "Jenkins"] {
        assert_eq!(normalize_entity_name(keep), keep.to_lowercase());
    }
}
