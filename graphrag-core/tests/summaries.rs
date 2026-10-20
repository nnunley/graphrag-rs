//! Task 14: fingerprinted bottom-up community summaries.

mod common;

use common::*;
use graphrag_core::fingerprints::CommunityConfig;
use graphrag_core::summaries::{SUMMARY_PROMPT_VERSION, SummaryGateError, SummaryPlan};

/// A 4-clique in `papers` and a 4-clique in `conversations` sharing nothing,
/// plus an isolated pair (a 2-member community).
fn graph(f: &Fx) {
    let cliques = [
        (["a1", "a2", "a3", "a4"], "papers"),
        (["b1", "b2", "b3", "b4"], "conversations"),
    ];
    for (names, group) in cliques {
        let mut rels = Vec::new();
        for i in 0..4 {
            for j in i + 1..4 {
                rels.push((names[i], names[j]));
            }
        }
        let (_, chunks) = f.doc(&sections(rels.len(), group), &[group]);
        for (k, (a, b)) in rels.iter().enumerate() {
            f.commit(
                chunks[k],
                &format!("{group}{k}"),
                vec![ent(a, "concept"), ent(b, "concept")],
                vec![rel(a, b, "associated_with")],
            );
        }
    }
    let (_, c) = f.doc("# pair\nx y", &["conversations"]);
    f.commit(
        c[0],
        "pair",
        vec![ent("x", "concept"), ent("y", "concept")],
        vec![rel("x", "y", "uses")],
    );
}

fn cfg() -> CommunityConfig {
    CommunityConfig {
        min_size_to_split: 100,
        ..CommunityConfig::default()
    }
}

fn calls(plans: &[SummaryPlan]) -> Vec<(i64, String, String)> {
    plans
        .iter()
        .filter_map(|p| match p {
            SummaryPlan::Call {
                community_id,
                fp,
                prompt,
            } => Some((
                *community_id,
                fp.clone(),
                format!("{}\n{}", prompt.system, prompt.user),
            )),
            _ => None,
        })
        .collect()
}

#[test]
fn small_communities_skipped() {
    let f = fx();
    graph(&f);
    f.db.build_community_run(STORE, &cfg()).unwrap();
    let plans = f.db.plan_summaries(STORE, "m", 100).unwrap();
    let skipped: Vec<_> = plans
        .iter()
        .filter(|p| {
            matches!(
                p,
                SummaryPlan::Skip {
                    reason: "skipped_small",
                    ..
                }
            )
        })
        .collect();
    assert_eq!(skipped.len(), 1, "{plans:?}");
    assert_eq!(calls(&plans).len(), 2);
    // A skip counts as complete: planning again offers nothing for it.
    let again = f.db.plan_summaries(STORE, "m", 100).unwrap();
    assert_eq!(calls(&again).len(), 2);
    assert!(again.iter().all(|p| !matches!(
        p,
        SummaryPlan::Skip {
            reason: "skipped_small",
            ..
        }
    )));
}

#[test]
fn parent_waits_for_children() {
    let f = fx();
    graph(&f);
    // Force a two-level hierarchy: split everything with ≥ 2 members.
    let c = CommunityConfig {
        min_size_to_split: 2,
        max_levels: 2,
        ..CommunityConfig::default()
    };
    f.db.build_community_run(STORE, &c).unwrap();
    let parents: Vec<i64> = {
        let conn = f.raw();
        let mut s = conn
            .prepare("SELECT DISTINCT parent_id FROM communities WHERE parent_id IS NOT NULL")
            .unwrap();
        s.query_map([], |r| r.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    if parents.is_empty() {
        return; // Leiden found no split; nothing to order.
    }
    let plans = f.db.plan_summaries(STORE, "m", 100).unwrap();
    for (id, _, _) in calls(&plans) {
        assert!(
            !parents.contains(&id),
            "parent {id} planned before its children"
        );
    }
}

#[test]
fn applied_summary_is_stored_and_current() {
    let f = fx();
    graph(&f);
    f.db.build_community_run(STORE, &cfg()).unwrap();
    let plans = f.db.plan_summaries(STORE, "m", 100).unwrap();
    let (id, fp, prompt) = calls(&plans).remove(0);
    let member = if prompt.contains("a1") { "a1" } else { "b1" };
    let text =
        format!("This community links {member} with its neighbours through shared associations.");
    f.db.apply_summary(id, &fp, &text, "m").unwrap();
    assert_eq!(
        f.count(&format!("SELECT COUNT(*) FROM summaries WHERE community_fp = '{fp}' AND prompt_version = '{SUMMARY_PROMPT_VERSION}'")),
        1
    );
    let left = calls(&f.db.plan_summaries(STORE, "m", 100).unwrap());
    assert!(left.iter().all(|(c, _, _)| *c != id));
}

#[test]
fn equal_fp_reused_across_runs() {
    let f = fx();
    graph(&f);
    f.db.build_community_run(STORE, &cfg()).unwrap();
    for (id, fp, prompt) in calls(&f.db.plan_summaries(STORE, "m", 100).unwrap()) {
        let member = if prompt.contains("a1") { "a1" } else { "b1" };
        f.db.apply_summary(
            id,
            &fp,
            &format!("Summary naming {member} and the others in this group."),
            "m",
        )
        .unwrap();
    }
    // A new run over the same graph (forced) gets the same summaries back with no calls.
    f.db.clear_communities(STORE).unwrap();
    f.db.build_community_run(STORE, &cfg()).unwrap();
    let plans = f.db.plan_summaries(STORE, "m", 100).unwrap();
    assert!(calls(&plans).is_empty(), "{plans:?}");
    assert!(plans.iter().any(|p| matches!(
        p,
        SummaryPlan::Skip {
            reason: "reused",
            ..
        }
    )));
    assert_eq!(
        f.count("SELECT COUNT(*) FROM communities WHERE summary IS NOT NULL"),
        2
    );
}

#[test]
fn gate_rejects_short_and_ungrounded() {
    let f = fx();
    graph(&f);
    f.db.build_community_run(STORE, &cfg()).unwrap();
    let (id, fp, _) = calls(&f.db.plan_summaries(STORE, "m", 100).unwrap()).remove(0);
    assert!(matches!(
        f.db.apply_summary(id, &fp, "   ", "m"),
        Err(SummaryGateError::Empty)
    ));
    assert!(matches!(
        f.db.apply_summary(id, &fp, "a1 short", "m"),
        Err(SummaryGateError::TooShort)
    ));
    assert!(matches!(
        f.db.apply_summary(
            id,
            &fp,
            "A long summary that names none of the members at all here.",
            "m"
        ),
        Err(SummaryGateError::Ungrounded)
    ));
    assert_eq!(f.count("SELECT COUNT(*) FROM summaries"), 0);
}

#[test]
fn prompt_names_groups_when_mixed() {
    let f = fx();
    graph(&f);
    // Bridge the cliques so one community spans both groups.
    let (_, c) = f.doc(&sections(6, "bridge"), &["papers"]);
    let pairs = [
        ("a1", "b1"),
        ("a2", "b2"),
        ("a3", "b3"),
        ("a4", "b4"),
        ("a1", "b2"),
        ("a2", "b1"),
    ];
    for (k, (a, b)) in pairs.iter().enumerate() {
        f.commit(
            c[k],
            &format!("br{k}"),
            vec![ent(a, "concept"), ent(b, "concept")],
            vec![rel(a, b, "associated_with")],
        );
    }
    f.db.build_community_run(STORE, &cfg()).unwrap();
    let prompts = calls(&f.db.plan_summaries(STORE, "m", 100).unwrap());
    assert!(
        prompts
            .iter()
            .any(|(_, _, p)| p.contains("papers") && p.contains("conversations")),
        "{prompts:?}"
    );
}

#[test]
fn prompt_follows_contract() {
    let f = fx();
    graph(&f);
    f.db.build_community_run(STORE, &cfg()).unwrap();
    let (_, _, p) = calls(&f.db.plan_summaries(STORE, "m", 100).unwrap()).remove(0);
    for needle in ["2-4 sentences", "only in the material", "a1"] {
        if needle == "a1" {
            assert!(p.contains("a1") || p.contains("b1"));
        } else {
            assert!(p.contains(needle), "missing {needle}");
        }
    }
}

#[test]
fn dormant_members_are_left_out() {
    let f = fx();
    graph(&f);
    f.db.build_community_run(STORE, &cfg()).unwrap();
    f.raw()
        .execute(
            "UPDATE entities SET edge_status = 'dormant' WHERE name IN ('a1','a2')",
            [],
        )
        .unwrap();
    let plans = f.db.plan_summaries(STORE, "m", 100).unwrap();
    let small = plans
        .iter()
        .filter(|p| {
            matches!(
                p,
                SummaryPlan::Skip {
                    reason: "skipped_small",
                    ..
                }
            )
        })
        .count();
    assert_eq!(small, 2, "the a-clique drops to 2 active members");
}
