//! Task 12B: ACT-R salience over a use log, with dormancy.

mod common;

use common::*;
use graphrag_core::fingerprints::CommunityConfig;
use graphrag_core::generalize::LiftConfig;
use graphrag_core::salience::{SalienceConfig, UseKind, activation};

const NOW: &str = "2026-10-20T12:00:00.000Z";

fn days_ago(d: i64) -> String {
    // Fixed calendar arithmetic is enough for tests: NOW minus d days.
    let base = 20_746i64; // 2026-10-20 as days since 1970-01-01
    let day = base - d;
    let (y, m, dd) = civil(day);
    format!("{y:04}-{m:02}-{dd:02}T12:00:00.000Z")
}

/// Howard Hinnant's days-to-civil.
fn civil(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[test]
fn date_helper_is_right() {
    assert_eq!(days_ago(0), NOW);
    assert_eq!(days_ago(365), "2025-10-20T12:00:00.000Z");
}

#[test]
fn activation_matches_worked_examples() {
    assert!((activation(&[(365.0, 1.0)], 0.5) + 2.95).abs() < 0.01);
    let ten: Vec<(f64, f64)> = (1..=10).map(|k| (3.0 * k as f64, 1.0)).collect();
    assert!((activation(&ten, 0.5) - 1.06).abs() < 0.02);
    assert_eq!(activation(&[], 0.5), f64::NEG_INFINITY);
    assert!(
        activation(&[(0.0, 1.0)], 0.5).is_finite(),
        "age floored at one hour"
    );
}

fn uses_of(f: &Fx, id: i64) -> i64 {
    f.count(&format!(
        "SELECT COUNT(*) FROM uses WHERE item_id = {id} AND kind = 'recall'"
    ))
}

#[test]
fn recall_counts_once_per_day() {
    let f = fx();
    let (_, c) = f.doc("note", &[]);
    f.commit(c[0], "r", vec![ent("Alice", "person")], vec![]);
    let a = f.id("Alice");
    for t in [
        "2026-10-20T01:00:00.000Z",
        "2026-10-20T09:00:00.000Z",
        "2026-10-20T23:59:00.000Z",
    ] {
        f.db.log_use("entity", a, UseKind::Recall, None, t).unwrap();
    }
    assert_eq!(uses_of(&f, a), 1);
    for t in ["2026-10-21T01:00:00.000Z", "2026-10-22T01:00:00.000Z"] {
        f.db.log_use("entity", a, UseKind::Recall, None, t).unwrap();
    }
    assert_eq!(uses_of(&f, a), 3);
}

/// The vcs graph (lifted edges admitted), with every use aged `age` days.
fn covered_graph(age: i64) -> Fx {
    let f = fx();
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
    f.commit(c[13], "u", vec![ent("UniqueGotcha", "concept")], vec![]);
    let r =
        f.db.lift(
            STORE,
            &LiftConfig {
                min_pairs: 3,
                g2_min: 10.83,
                ic_floor: 0.0,
            },
        )
        .unwrap();
    assert!(r.admitted >= 1);
    f.raw()
        .execute("UPDATE uses SET at = ?1", [days_ago(age)])
        .unwrap();
    f
}

fn status_of(f: &Fx, name: &str) -> String {
    f.raw()
        .query_row(
            "SELECT edge_status FROM entities WHERE name = ?1",
            [name],
            |r| r.get(0),
        )
        .unwrap()
}

#[test]
fn low_salience_covered_item_goes_dormant() {
    let f = covered_graph(800);
    let rep =
        f.db.refresh_salience(STORE, &SalienceConfig::default(), NOW)
            .unwrap();
    assert!(rep.went_dormant >= 1, "{rep:?}");
    assert_eq!(status_of(&f, "jj"), "dormant");
}

#[test]
fn low_salience_unique_item_stays_active() {
    let f = covered_graph(800);
    f.db.refresh_salience(STORE, &SalienceConfig::default(), NOW)
        .unwrap();
    assert_eq!(
        status_of(&f, "UniqueGotcha"),
        "active",
        "no class, so no gist above it"
    );
}

#[test]
fn single_mention_active_for_a_year() {
    let f = covered_graph(360);
    f.db.refresh_salience(STORE, &SalienceConfig::default(), NOW)
        .unwrap();
    assert_eq!(status_of(&f, "jj"), "active");
    let g = covered_graph(420);
    g.db.refresh_salience(STORE, &SalienceConfig::default(), NOW)
        .unwrap();
    assert_eq!(status_of(&g, "jj"), "dormant");
}

#[test]
fn repeated_search_keeps_item_active() {
    let f = covered_graph(800);
    let jj = f.id("jj");
    for w in 0..20 {
        f.db.log_use("entity", jj, UseKind::Recall, None, &days_ago(7 * w))
            .unwrap();
    }
    f.db.refresh_salience(STORE, &SalienceConfig::default(), NOW)
        .unwrap();
    assert_eq!(status_of(&f, "jj"), "active");
}

#[test]
fn below_tau_goes_dormant_not_deleted() {
    let f = covered_graph(800);
    let before = f.count("SELECT COUNT(*) FROM entities");
    let rels = f.count("SELECT COUNT(*) FROM relations");
    f.db.refresh_salience(STORE, &SalienceConfig::default(), NOW)
        .unwrap();
    assert_eq!(f.count("SELECT COUNT(*) FROM entities"), before);
    assert_eq!(f.count("SELECT COUNT(*) FROM relations"), rels);
    let s: f64 = f
        .raw()
        .query_row("SELECT salience FROM entities WHERE name = 'jj'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert!(s < -3.0);
}

#[test]
fn new_use_revives() {
    let f = covered_graph(800);
    f.db.refresh_salience(STORE, &SalienceConfig::default(), NOW)
        .unwrap();
    assert_eq!(status_of(&f, "jj"), "dormant");
    f.db.log_use("entity", f.id("jj"), UseKind::Recall, None, NOW)
        .unwrap();
    let rep =
        f.db.refresh_salience(STORE, &SalienceConfig::default(), NOW)
            .unwrap();
    assert!(rep.revived >= 1);
    assert_eq!(status_of(&f, "jj"), "active");
}

#[test]
fn assertion_counts_as_a_use() {
    let f = covered_graph(800);
    let jj = f.id("jj");
    let g = f.db.get_or_create_group(STORE, "tools").unwrap();
    f.db.assert_edge(
        jj,
        "part_of",
        g,
        &graphrag_core::assertions::Asserter::new("human:norman").unwrap(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        f.count(&format!(
            "SELECT COUNT(*) FROM uses WHERE item_id = {jj} AND kind = 'assert'"
        )),
        1
    );
}

#[test]
fn dormant_excluded_from_projection() {
    let f = covered_graph(800);
    f.db.refresh_salience(STORE, &SalienceConfig::default(), NOW)
        .unwrap();
    let p =
        f.db.project_graph(STORE, &CommunityConfig::default())
            .unwrap();
    let jj = f.id("jj");
    assert!(p.edges.iter().all(|e| e.0 != jj && e.1 != jj));
    assert!(!p.isolates.contains(&jj));
}

#[test]
fn dormant_instances_still_count_in_lifted_pairs() {
    let f = covered_graph(800);
    f.db.refresh_salience(STORE, &SalienceConfig::default(), NOW)
        .unwrap();
    f.db.lift(
        STORE,
        &LiftConfig {
            min_pairs: 3,
            g2_min: 10.83,
            ic_floor: 0.0,
        },
    )
    .unwrap();
    let pairs: i64 = f
        .raw()
        .query_row(
            "SELECT l.pairs FROM lifted_edges l JOIN entities h ON h.id = l.head_class
             WHERE l.relation = 'exchanges_data_with' AND h.name = 'version control system'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(pairs, 3);
}

#[test]
fn documents_and_chunks_never_dormant() {
    let f = covered_graph(800);
    f.db.refresh_salience(STORE, &SalienceConfig::default(), NOW)
        .unwrap();
    assert_eq!(
        f.count("SELECT COUNT(*) FROM entities WHERE kind != 'entity' AND edge_status != 'active'"),
        0
    );
    let docs = f.count("SELECT COUNT(*) FROM documents");
    assert!(docs > 0);
    assert_eq!(f.count("SELECT COUNT(*) FROM documents d WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)"), 0);
}
