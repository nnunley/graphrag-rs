//! Cross-language contract tests for Project Capsule v1 references.
use graphrag_core::capsule::{
    CapsuleError, CapsuleKind, CapsuleRef, Commitment, Decision, EvidenceItem, Freshness,
    NextSegment, OpenThread, ProjectCapsuleV1, ProjectIdentity, VerifiedFact,
};

/// The exact cross-language fixture also tested by the Go worker.
const REF_FIXTURE: &str = r#"{"schema_version":1,"uri":"graphrag://capsules/v1/project/graphrag-rs-current","kind":"project","capsule_id":"graphrag-rs-current","project_id":"graphrag-rs","content_fingerprint":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","generated_at":"2026-08-07T00:00:00Z"}"#;

fn sample_capsule() -> ProjectCapsuleV1 {
    ProjectCapsuleV1 {
        schema_version: 1,
        capsule_id: "graphrag-rs-current".into(),
        project: ProjectIdentity {
            project_id: "graphrag-rs".into(),
            display_name: "graphrag-rs".into(),
            repository: Some("https://github.com/nnunley/graphrag-rs".into()),
            locations: vec!["/Users/ndn/development/graphrag-rs".into()],
        },
        purpose:
            "Indexing, discovery, and long-context synthesis layer for the agentic memory system."
                .into(),
        verified_state: vec![VerifiedFact {
            statement: "Community summaries populated 63/63.".into(),
            evidence: vec!["ev-community-population".into()],
        }],
        decisions: vec![Decision {
            decision: "Typed query pipeline compiles to leit UserQueryProgram.".into(),
            rationale: "Total language; executes AST directly.".into(),
            evidence: vec!["ev-query-report".into()],
        }],
        open_threads: vec![OpenThread {
            thread_id: "typed-query-consumer".into(),
            title: "Land graphrag typed-query consumer after leit PR #18.".into(),
            status: "blocked".into(),
            owner: "norman".into(),
            next_action: "Remediate leit PR #18 findings.".into(),
        }],
        commitments: vec![Commitment {
            commitment_id: "capsule-v1-spec".into(),
            title: "Specify Project Capsule v1.".into(),
            status: "in_progress".into(),
            owner: "agent".into(),
            next_action: "Land schema + deterministic reference.".into(),
            external_waiting: false,
        }],
        next_segment: NextSegment {
            title: "Capsule persistence + synthesis".into(),
            entry_point: "graphrag-core/src/capsule.rs".into(),
            first_action: "Wire capsule storage into Database.".into(),
        },
        evidence: vec![
            EvidenceItem {
                evidence_id: "ev-community-population".into(),
                uri: "graphrag://stores/conversations/communities".into(),
                observed_at: "2026-08-06T22:00:00Z".into(),
                fingerprint: None,
            },
            EvidenceItem {
                evidence_id: "ev-query-report".into(),
                uri: "file:///tmp/query-pipeline/REPORT.md".into(),
                observed_at: "2026-08-06T23:00:00Z".into(),
                fingerprint: Some(
                    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                        .into(),
                ),
            },
        ],
        freshness: Freshness {
            generated_at: "2026-08-07T00:00:00Z".into(),
            source_fingerprint:
                "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into(),
            stale_after: Some("2026-08-14T00:00:00Z".into()),
            input_fingerprints: vec![
                "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into(),
            ],
        },
    }
}

#[test]
fn ref_fixture_roundtrips_exactly() {
    let r: CapsuleRef = serde_json::from_str(REF_FIXTURE).unwrap();
    assert_eq!(r.schema_version, 1);
    assert_eq!(r.kind, CapsuleKind::Project);
    assert_eq!(r.capsule_id, "graphrag-rs-current");
    assert_eq!(r.project_id, "graphrag-rs");
    r.validate().unwrap();
    // Field order is fixed by struct declaration order: exact byte roundtrip.
    assert_eq!(serde_json::to_string(&r).unwrap(), REF_FIXTURE);
}

#[test]
fn reference_is_deterministic_and_valid() {
    let c = sample_capsule();
    let r1 = c.reference().unwrap();
    let r2 = c.reference().unwrap();
    assert_eq!(r1, r2);
    r1.validate().unwrap();
    assert_eq!(r1.uri, "graphrag://capsules/v1/project/graphrag-rs-current");
    assert!(r1.content_fingerprint.starts_with("sha256:"));
    assert_eq!(r1.content_fingerprint.len(), "sha256:".len() + 64);
    assert_eq!(r1.generated_at, c.freshness.generated_at);
}

#[test]
fn mutation_changes_fingerprint() {
    let c = sample_capsule();
    let base = c.reference().unwrap();
    let mut m = sample_capsule();
    m.purpose.push_str(" Updated.");
    assert_ne!(
        base.content_fingerprint,
        m.reference().unwrap().content_fingerprint
    );
}

#[test]
fn dangling_evidence_link_rejected() {
    let mut c = sample_capsule();
    c.verified_state[0].evidence = vec!["no-such-evidence".into()];
    assert!(matches!(
        c.validate(),
        Err(CapsuleError::DanglingEvidence { .. })
    ));
}

#[test]
fn empty_evidence_citation_rejected() {
    let mut c = sample_capsule();
    c.decisions[0].evidence.clear();
    assert!(matches!(
        c.validate(),
        Err(CapsuleError::MissingEvidenceCitation { .. })
    ));
}

#[test]
fn duplicate_evidence_ids_rejected() {
    let mut c = sample_capsule();
    let dup = c.evidence[0].clone();
    c.evidence.push(dup);
    assert!(matches!(
        c.validate(),
        Err(CapsuleError::DuplicateEvidenceId { .. })
    ));
}

#[test]
fn bad_fingerprint_rejected() {
    let mut c = sample_capsule();
    c.freshness.source_fingerprint = "sha256:XYZ".into();
    assert!(matches!(
        c.validate(),
        Err(CapsuleError::InvalidFingerprint { .. })
    ));
}

#[test]
fn bad_timestamp_rejected() {
    let mut c = sample_capsule();
    c.freshness.generated_at = "yesterday".into();
    assert!(matches!(
        c.validate(),
        Err(CapsuleError::InvalidTimestamp { .. })
    ));
}

#[test]
fn bad_capsule_id_rejected() {
    let mut c = sample_capsule();
    c.capsule_id = "has space".into();
    assert!(matches!(c.validate(), Err(CapsuleError::InvalidId { .. })));
    let mut c2 = sample_capsule();
    c2.capsule_id = String::new();
    assert!(matches!(c2.validate(), Err(CapsuleError::InvalidId { .. })));
}

#[test]
fn ref_uri_must_agree_with_kind_and_id() {
    let mut r: CapsuleRef = serde_json::from_str(REF_FIXTURE).unwrap();
    r.uri = "graphrag://capsules/v1/project/other-capsule".into();
    assert!(matches!(
        r.validate(),
        Err(CapsuleError::UriMismatch { .. })
    ));
}

#[test]
fn schema_version_must_be_one() {
    let mut c = sample_capsule();
    c.schema_version = 2;
    assert!(matches!(
        c.validate(),
        Err(CapsuleError::UnsupportedSchemaVersion { .. })
    ));
}

#[test]
fn kind_serializes_snake_case() {
    assert_eq!(
        serde_json::to_string(&CapsuleKind::Project).unwrap(),
        "\"project\""
    );
    assert_eq!(
        serde_json::to_string(&CapsuleKind::Segment).unwrap(),
        "\"segment\""
    );
    assert_eq!(
        serde_json::to_string(&CapsuleKind::Portfolio).unwrap(),
        "\"portfolio\""
    );
}
