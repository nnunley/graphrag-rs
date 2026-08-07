//! Integration tests for the SQLite CapsuleStore provider.
use graphrag_core::Database;
use graphrag_core::capsule::{
    CapsuleStore, Commitment, Decision, EvidenceItem, Freshness, NextSegment, OpenThread,
    ProjectCapsuleV1, ProjectIdentity, VerifiedFact,
};
use tempfile::TempDir;

fn sample(purpose: &str, generated_at: &str) -> ProjectCapsuleV1 {
    ProjectCapsuleV1 {
        schema_version: 1,
        capsule_id: "graphrag-rs-current".into(),
        project: ProjectIdentity {
            project_id: "graphrag-rs".into(),
            display_name: "graphrag-rs".into(),
            repository: Some("https://github.com/nnunley/graphrag-rs".into()),
            locations: vec!["/Users/ndn/development/graphrag-rs".into()],
        },
        purpose: purpose.into(),
        verified_state: vec![VerifiedFact {
            statement: "Capsule schema landed on main.".into(),
            evidence: vec!["ev-merge".into()],
        }],
        decisions: vec![Decision {
            decision: "Capsules persist in SQLite.".into(),
            rationale: "Single system of record.".into(),
            evidence: vec!["ev-merge".into()],
        }],
        open_threads: vec![OpenThread {
            thread_id: "vector-lane".into(),
            title: "Vector candidate source trait".into(),
            status: "queued".into(),
            owner: "agent".into(),
            next_action: "brute-force provider".into(),
        }],
        commitments: vec![Commitment {
            commitment_id: "persistence".into(),
            title: "Capsule persistence".into(),
            status: "in_progress".into(),
            owner: "agent".into(),
            next_action: "land store".into(),
            external_waiting: false,
        }],
        next_segment: NextSegment {
            title: "Synthesize first real capsule".into(),
            entry_point: "graphrag-core/src/capsule_store.rs".into(),
            first_action: "wire CLI".into(),
        },
        evidence: vec![EvidenceItem {
            evidence_id: "ev-merge".into(),
            uri: "https://github.com/nnunley/graphrag-rs/commit/905f368".into(),
            observed_at: "2026-08-07T16:03:09Z".into(),
            fingerprint: None,
        }],
        freshness: Freshness {
            generated_at: generated_at.into(),
            source_fingerprint:
                "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into(),
            stale_after: None,
            input_fingerprints: vec![],
        },
    }
}

fn open_db(dir: &TempDir) -> Database {
    Database::open(&dir.path().join("test.db")).unwrap()
}

#[test]
fn put_then_latest_roundtrips() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    let c = sample("Persist me.", "2026-08-07T18:00:00Z");
    let r = db.put_capsule(&c).unwrap();
    assert_eq!(r, c.reference().unwrap());
    let back = db.latest_capsule("graphrag-rs-current").unwrap().unwrap();
    assert_eq!(back, c);
}

#[test]
fn missing_capsule_is_none() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    assert!(db.latest_capsule("nope").unwrap().is_none());
    assert!(db.capsule_history("nope").unwrap().is_empty());
}

#[test]
fn put_is_idempotent_per_fingerprint() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    let c = sample("Same bytes.", "2026-08-07T18:00:00Z");
    let r1 = db.put_capsule(&c).unwrap();
    let r2 = db.put_capsule(&c).unwrap();
    assert_eq!(r1, r2);
    assert_eq!(db.capsule_history("graphrag-rs-current").unwrap().len(), 1);
}

#[test]
fn history_is_newest_first_and_latest_wins() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    let v1 = sample("First version.", "2026-08-07T18:00:00Z");
    let v2 = sample("Second version.", "2026-08-07T19:00:00Z");
    let r1 = db.put_capsule(&v1).unwrap();
    let r2 = db.put_capsule(&v2).unwrap();
    assert_ne!(r1.content_fingerprint, r2.content_fingerprint);
    let hist = db.capsule_history("graphrag-rs-current").unwrap();
    assert_eq!(hist, vec![r2.clone(), r1.clone()]);
    let latest = db.latest_capsule("graphrag-rs-current").unwrap().unwrap();
    assert_eq!(latest, v2);
}

#[test]
fn fetch_by_fingerprint_returns_that_version() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    let v1 = sample("First version.", "2026-08-07T18:00:00Z");
    let v2 = sample("Second version.", "2026-08-07T19:00:00Z");
    let r1 = db.put_capsule(&v1).unwrap();
    db.put_capsule(&v2).unwrap();
    let old = db
        .capsule_by_fingerprint("graphrag-rs-current", &r1.content_fingerprint)
        .unwrap()
        .unwrap();
    assert_eq!(old, v1);
    assert!(
        db.capsule_by_fingerprint(
            "graphrag-rs-current",
            "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn list_capsules_filters_by_project() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    let a = sample("Project A.", "2026-08-07T18:00:00Z");
    let mut b = sample("Project B.", "2026-08-07T18:30:00Z");
    b.capsule_id = "other-current".into();
    b.project.project_id = "other".into();
    db.put_capsule(&a).unwrap();
    db.put_capsule(&b).unwrap();
    let all = db.list_capsules(None).unwrap();
    assert_eq!(all.len(), 2);
    let only = db.list_capsules(Some("graphrag-rs")).unwrap();
    assert_eq!(only.len(), 1);
    assert_eq!(only[0].capsule_id, "graphrag-rs-current");
}

#[test]
fn invalid_capsule_is_rejected_before_storage() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    let mut c = sample("Bad evidence.", "2026-08-07T18:00:00Z");
    c.decisions[0].evidence = vec!["dangling".into()];
    assert!(db.put_capsule(&c).is_err());
    assert!(db.latest_capsule("graphrag-rs-current").unwrap().is_none());
}
