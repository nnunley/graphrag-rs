//! Task 2: per-step input fingerprints.

use graphrag_core::fingerprints::{
    ChunkerProfile, CommunityConfig, SummaryInput, chunk_plan_fp, document_fp, embed_fp,
    extract_fp, graph_fp, summary_fp,
};
use graphrag_core::paper::{ExtractionAttemptIdentityParts, ExtractionAttemptIdentityV1};

fn is_sha(s: &str) -> bool {
    s.len() == 71 && s.starts_with("sha256:") && s[7..].bytes().all(|b| b.is_ascii_hexdigit())
}

#[test]
fn each_step_has_a_distinct_domain() {
    let a = document_fp("x");
    let b = embed_fp("x", "", 0);
    let c = chunk_plan_fp("x", &ChunkerProfile::default_markdown());
    assert!(is_sha(&a) && is_sha(&b) && is_sha(&c));
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
}

#[test]
fn chunk_plan_fp_changes_with_profile() {
    let p = ChunkerProfile::default_markdown();
    let q = ChunkerProfile {
        overlap: 100,
        ..p.clone()
    };
    let k = ChunkerProfile {
        kind: "code",
        ..p.clone()
    };
    assert_ne!(chunk_plan_fp("x", &p), chunk_plan_fp("x", &q));
    assert_ne!(chunk_plan_fp("x", &p), chunk_plan_fp("x", &k));
    assert_eq!(p.size, 2000);
    assert_eq!(p.overlap, 200);
    assert_eq!(p.kind, "markdown-sections");
}

#[test]
fn graph_fp_ignores_edge_order() {
    let cfg = CommunityConfig::default();
    let e1 = vec![(1, 2, 1.0), (2, 3, 2.0)];
    let e2 = vec![(2, 3, 2.0), (1, 2, 1.0)];
    assert_eq!(
        graph_fp(&e1, &[5, 4], &cfg, 7),
        graph_fp(&e2, &[4, 5], &cfg, 7)
    );
}

#[test]
fn graph_fp_changes_with_any_weight_or_config_field() {
    let base = CommunityConfig::default();
    let edges = vec![(1, 2, 1.0)];
    let fp = |c: &CommunityConfig| graph_fp(&edges, &[], c, 7);
    let b = fp(&base);
    assert_ne!(b, graph_fp(&[(1, 2, 2.0)], &[], &base, 7), "edge weight");
    assert_ne!(b, graph_fp(&edges, &[9], &base, 7), "isolates");
    assert_ne!(b, graph_fp(&edges, &[], &base, 8), "seed");
    assert_ne!(
        b,
        fp(&CommunityConfig {
            asserted_weight: 1.0,
            ..base.clone()
        })
    );
    assert_ne!(
        b,
        fp(&CommunityConfig {
            mention_weight: 1.0,
            ..base.clone()
        })
    );
    assert_ne!(
        b,
        fp(&CommunityConfig {
            structure_weight: 1.0,
            ..base.clone()
        })
    );
    assert_ne!(
        b,
        fp(&CommunityConfig {
            class_weight: 1.0,
            ..base.clone()
        })
    );
    assert_ne!(
        b,
        fp(&CommunityConfig {
            max_levels: 3,
            ..base.clone()
        })
    );
    assert_ne!(
        b,
        fp(&CommunityConfig {
            min_size_to_split: 99,
            ..base.clone()
        })
    );
    assert_ne!(
        b,
        fp(&CommunityConfig {
            seed: 99,
            ..base.clone()
        })
    );
}

#[test]
fn community_config_defaults() {
    let c = CommunityConfig::default();
    assert_eq!(
        (
            c.asserted_weight,
            c.mention_weight,
            c.structure_weight,
            c.class_weight
        ),
        (0.0, 0.0, 0.0, 0.0)
    );
    assert_eq!(c.max_levels, 4);
}

#[test]
fn extract_fp_matches_m0_identity() {
    let parts = ExtractionAttemptIdentityParts {
        store: "memory",
        chunk_id: 3,
        chunk_content: "text",
        model: "qwen3.8-27b",
        extraction_profile: "paper-json",
        prompt_version: "paper-json-v1.1",
        response_schema_version: "graph-v1",
        glean_max_rounds: 0,
    };
    assert_eq!(
        extract_fp(&parts).unwrap(),
        ExtractionAttemptIdentityV1::derive(&parts)
            .unwrap()
            .fingerprint
    );
}

#[test]
fn embed_fp_changes_with_dim() {
    assert_ne!(embed_fp("t", "nomic", 768), embed_fp("t", "nomic", 512));
    assert_ne!(embed_fp("t", "nomic", 768), embed_fp("t", "other", 768));
}

#[test]
fn summary_fp_changes_with_inputs() {
    let input = SummaryInput {
        members: vec![("a".into(), "software".into(), "desc".into())],
        edges: vec![("a".into(), "b".into(), "uses".into(), 2.0, "d".into())],
        child_fps: vec![],
        groups: vec!["papers".into()],
    };
    let base = summary_fp(&input, "m", "summary-v1");
    let mut other = input.clone();
    other.edges[0].3 = 3.0;
    assert_ne!(base, summary_fp(&other, "m", "summary-v1"));
    assert_ne!(base, summary_fp(&input, "m2", "summary-v1"));
    assert_ne!(base, summary_fp(&input, "m", "summary-v2"));
}
