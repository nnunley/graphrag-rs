//! Task 13: fingerprinted community runs replace ad hoc clustering.

mod common;

use common::*;
use graphrag_core::fingerprints::CommunityConfig;
use graphrag_core::leiden::CommunityGraph;

/// Two 4-cliques joined by one edge, each relation stated in its own section.
fn two_cliques(f: &Fx) {
    let groups = [["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]];
    let mut rels = Vec::new();
    for g in &groups {
        for i in 0..4 {
            for j in i + 1..4 {
                rels.push((g[i], g[j]));
            }
        }
    }
    rels.push(("a1", "b1"));
    let (_, chunks) = f.doc(&sections(rels.len(), "c"), &[]);
    for (k, (a, b)) in rels.iter().enumerate() {
        f.commit(
            chunks[k],
            &format!("r{k}"),
            vec![ent(a, "concept"), ent(b, "concept")],
            vec![rel(a, b, "associated_with")],
        );
    }
}

fn cfg() -> CommunityConfig {
    CommunityConfig {
        min_size_to_split: 3,
        ..CommunityConfig::default()
    }
}

#[test]
fn same_graph_reuses_run() {
    let f = fx();
    two_cliques(&f);
    let a = f.db.build_community_run(STORE, &cfg()).unwrap();
    assert!(!a.reused);
    assert!(a.communities >= 2);
    let b = f.db.build_community_run(STORE, &cfg()).unwrap();
    assert!(b.reused);
    assert_eq!(a.run_id, b.run_id);
}

#[test]
fn changed_weight_creates_new_run_and_replaces_old() {
    let f = fx();
    two_cliques(&f);
    let a = f.db.build_community_run(STORE, &cfg()).unwrap();
    let (_, chunks) = f.doc("# extra\nmore", &[]);
    f.commit(
        chunks[0],
        "x",
        vec![ent("a1", "concept"), ent("a2", "concept")],
        vec![rel("a1", "a2", "associated_with")],
    );
    let b = f.db.build_community_run(STORE, &cfg()).unwrap();
    assert!(!b.reused);
    assert_ne!(a.run_id, b.run_id);
    assert_eq!(
        f.count("SELECT COUNT(*) FROM community_runs WHERE current = 1"),
        1
    );
    assert_eq!(
        f.count("SELECT COUNT(*) FROM community_runs"),
        1,
        "no left-over runs"
    );
    assert_eq!(
        f.count(&format!(
            "SELECT COUNT(*) FROM communities WHERE run_id IS NOT {}",
            b.run_id
        )),
        0
    );
}

#[test]
fn run_records_config_and_seed() {
    let f = fx();
    two_cliques(&f);
    let c = CommunityConfig {
        asserted_weight: 0.5,
        seed: 7,
        ..cfg()
    };
    let r = f.db.build_community_run(STORE, &c).unwrap();
    let rec = f.db.current_run(STORE).unwrap().unwrap();
    assert_eq!(rec.run_id, r.run_id);
    assert_eq!(rec.seed, 7);
    let stored: CommunityConfig = serde_json::from_str(&rec.config_json).unwrap();
    assert_eq!(stored, c);
    assert_eq!(rec.graph_fp, f.db.graph_input_fp(STORE, &c).unwrap());
}

#[test]
fn communities_step_is_recorded() {
    let f = fx();
    two_cliques(&f);
    f.db.build_community_run(STORE, &cfg()).unwrap();
    let fp: String = f
        .raw()
        .query_row(
            "SELECT fp FROM step_status WHERE subject_kind='graph' AND step='communities' AND state='current'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(fp, f.db.graph_input_fp(STORE, &cfg()).unwrap());
}

#[test]
fn levels_capped_at_four() {
    let f = fx();
    two_cliques(&f);
    let c = CommunityConfig {
        min_size_to_split: 1,
        ..cfg()
    };
    let r = f.db.build_community_run(STORE, &c).unwrap();
    assert!(r.levels <= 4);
    assert!(f.count("SELECT COALESCE(MAX(level), 0) FROM communities") <= 3);
}

#[test]
fn interrupted_build_leaves_old_run_current() {
    let f = fx();
    two_cliques(&f);
    let a = f.db.build_community_run(STORE, &cfg()).unwrap();
    let before = f.count("SELECT COUNT(*) FROM communities");
    f.raw()
        .execute_batch(
            "CREATE TRIGGER fail_membership BEFORE INSERT ON entity_communities
             BEGIN SELECT RAISE(ABORT, 'injected'); END;",
        )
        .unwrap();
    let (_, chunks) = f.doc("# extra\nmore", &[]);
    f.commit(
        chunks[0],
        "x",
        vec![ent("b1", "concept"), ent("b2", "concept")],
        vec![rel("b1", "b2", "associated_with")],
    );
    assert!(f.db.build_community_run(STORE, &cfg()).is_err());
    let cur = f.db.current_run(STORE).unwrap().unwrap();
    assert_eq!(cur.run_id, a.run_id);
    assert_eq!(f.count("SELECT COUNT(*) FROM communities"), before);
}

#[test]
fn members_are_entities_and_fp_is_stable() {
    let f = fx();
    two_cliques(&f);
    f.db.build_community_run(STORE, &cfg()).unwrap();
    let fps = |f: &Fx| {
        let conn = f.raw();
        let mut s = conn
            .prepare("SELECT fp FROM communities ORDER BY fp")
            .unwrap();
        s.query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
    };
    let first = fps(&f);
    // Rebuild from scratch (force a new run) and compare community fingerprints.
    f.raw().execute("DELETE FROM community_runs", []).unwrap();
    f.db.build_community_run(STORE, &cfg()).unwrap();
    assert_eq!(first, fps(&f));
}

#[test]
fn leiden_is_deterministic() {
    let mut edges = Vec::new();
    for g in 0..6i64 {
        for i in 0..5i64 {
            for j in i + 1..5 {
                edges.push((g * 10 + i, g * 10 + j, 1.0));
            }
        }
        edges.push((g * 10, ((g + 1) % 6) * 10, 1.0));
    }
    let run = || {
        let g = CommunityGraph::from_edges(edges.clone());
        let r = g.leiden_hierarchical(None, 1e-4, 3, Some(3));
        let mut out: Vec<(i32, Vec<i64>)> = r
            .communities
            .iter()
            .map(|c| {
                let mut n = c.nodes.clone();
                n.sort_unstable();
                (c.level, n)
            })
            .collect();
        out.sort();
        out
    };
    let first = run();
    for _ in 0..10 {
        assert_eq!(run(), first);
    }
}
