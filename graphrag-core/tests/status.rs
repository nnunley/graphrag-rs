//! Task 7: fingerprinted step status and the next-work query.

use graphrag_core::Database;
use graphrag_core::documents::{AddDocument, Origin};
use graphrag_core::fingerprints::{ChunkerProfile, CommunityConfig, ExtractionConfig};
use graphrag_core::status::{ExpectedFingerprints, StaticConfig, Step, StepState, SubjectKind};
use tempfile::TempDir;

const NOW: &str = "2026-10-20T12:00:00.000Z";

struct Fx {
    _dir: TempDir,
    db: Database,
    cfg: StaticConfig,
}

fn cfg() -> StaticConfig {
    StaticConfig {
        store: "memory".into(),
        embedder_id: "fake".into(),
        dim: 3,
        chunker: ChunkerProfile::default_markdown(),
        extraction: ExtractionConfig {
            model: "m".into(),
            profile: "paper-json".into(),
            prompt_version: "v1".into(),
            schema_version: "graph-v1".into(),
            glean_max_rounds: 0,
        },
        community: CommunityConfig::default(),
    }
}

fn fx() -> Fx {
    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("t.db")).unwrap();
    db.create_store("memory", 3).unwrap();
    Fx {
        _dir: dir,
        db,
        cfg: cfg(),
    }
}

impl Fx {
    fn add(&self, text: &str, groups: &[&str], origin: Origin) -> Vec<i64> {
        let p = ChunkerProfile::default_markdown();
        let out = self
            .db
            .add_document(&AddDocument {
                store: "memory",
                content: text,
                origin,
                source: None,
                doc_key: None,
                title: None,
                groups,
                surface: "cli",
                profile: &p,
            })
            .unwrap();
        let id = match out {
            graphrag_core::documents::AddOutcome::Created { document_id } => document_id,
            other => panic!("{other:?}"),
        };
        self.db.document_chunks(id).unwrap()
    }
    fn complete(&self, chunk: i64) {
        for step in [Step::Embed, Step::Extract] {
            if let Some(fp) = self
                .cfg
                .expected(&self.db, SubjectKind::Chunk, chunk, step)
                .unwrap()
            {
                self.db
                    .record_step(
                        SubjectKind::Chunk,
                        chunk,
                        step,
                        &fp,
                        StepState::Current,
                        None,
                    )
                    .unwrap();
            }
        }
    }
    fn finish_graph(&self) {
        let fp = self
            .cfg
            .expected(&self.db, SubjectKind::Graph, 0, Step::Communities)
            .unwrap()
            .unwrap();
        self.db
            .record_step(
                SubjectKind::Graph,
                0,
                Step::Communities,
                &fp,
                StepState::Current,
                None,
            )
            .unwrap();
    }
    fn work(&self) -> Vec<(SubjectKind, i64, Step)> {
        self.db
            .next_work(&self.cfg, NOW, 100)
            .unwrap()
            .into_iter()
            .map(|w| (w.kind, w.id, w.step))
            .collect()
    }
}

#[test]
fn missing_status_is_work() {
    let f = fx();
    let c = f.add("a note", &[], Origin::Source)[0];
    let w = f.work();
    assert!(w.contains(&(SubjectKind::Chunk, c, Step::Embed)));
    assert!(w.contains(&(SubjectKind::Chunk, c, Step::Extract)));
}

#[test]
fn matching_fp_is_not_work() {
    let f = fx();
    let c = f.add("a note", &[], Origin::Source)[0];
    f.complete(c);
    assert!(f.work().iter().all(|(k, _, _)| *k != SubjectKind::Chunk));
}

#[test]
fn changed_config_fp_makes_stale() {
    let mut f = fx();
    let c = f.add("a note", &[], Origin::Source)[0];
    f.complete(c);
    f.cfg.extraction.prompt_version = "v2".into();
    let w = f.work();
    assert!(w.contains(&(SubjectKind::Chunk, c, Step::Extract)));
    assert!(!w.contains(&(SubjectKind::Chunk, c, Step::Embed)));
    let r = f.db.status_report(&f.cfg).unwrap();
    assert_eq!(r.total.stale.get(&Step::Extract), Some(&1));
}

#[test]
fn escalated_chunk_comes_first() {
    let f = fx();
    let a = f.add("first note", &[], Origin::Source)[0];
    let b = f.add("second note", &[], Origin::Source)[0];
    assert_eq!(f.work()[0].1, a);
    assert_eq!(f.db.escalate_chunks(&[b]).unwrap(), 1);
    let w = f.db.next_work(&f.cfg, NOW, 100).unwrap();
    assert_eq!(w[0].id, b);
    assert_eq!(w[0].priority, 2);
}

#[test]
fn escalate_ignores_complete_chunks() {
    let f = fx();
    let a = f.add("done note", &[], Origin::Source)[0];
    f.complete(a);
    assert_eq!(f.db.escalate_chunks(&[a]).unwrap(), 0);
}

#[test]
fn new_documents_are_priority_one() {
    let f = fx();
    let a = f.add("fresh", &[], Origin::Source)[0];
    let w = f.db.next_work(&f.cfg, NOW, 100).unwrap();
    assert!(w.iter().any(|x| x.id == a && x.priority == 1));
}

#[test]
fn backoff_hides_item_until_next_try() {
    let f = fx();
    let c = f.add("a note", &[], Origin::Source)[0];
    let n =
        f.db.record_transient_failure(
            SubjectKind::Chunk,
            c,
            Step::Extract,
            "transport: refused",
            "2026-10-20T13:00:00.000Z",
        )
        .unwrap();
    assert_eq!(n, 1);
    assert!(!f.work().contains(&(SubjectKind::Chunk, c, Step::Extract)));
    let later =
        f.db.next_work(&f.cfg, "2026-10-20T14:00:00.000Z", 100)
            .unwrap();
    assert!(later.iter().any(|w| w.id == c && w.step == Step::Extract));
}

#[test]
fn failed_is_not_retried_until_fp_changes() {
    let mut f = fx();
    let c = f.add("a note", &[], Origin::Source)[0];
    let fp = f
        .cfg
        .expected(&f.db, SubjectKind::Chunk, c, Step::Extract)
        .unwrap()
        .unwrap();
    f.db.record_step(
        SubjectKind::Chunk,
        c,
        Step::Extract,
        &fp,
        StepState::Failed,
        Some("malformed x3"),
    )
    .unwrap();
    assert!(!f.work().contains(&(SubjectKind::Chunk, c, Step::Extract)));
    f.cfg.extraction.model = "m2".into();
    assert!(f.work().contains(&(SubjectKind::Chunk, c, Step::Extract)));
}

#[test]
fn communities_wait_for_chunk_work() {
    let f = fx();
    let c = f.add("a note", &[], Origin::Source)[0];
    assert!(!f.work().iter().any(|(k, _, _)| *k == SubjectKind::Graph));
    f.complete(c);
    assert!(
        f.work()
            .contains(&(SubjectKind::Graph, 0, Step::Communities))
    );
}

#[test]
fn current_communities_are_not_work() {
    let f = fx();
    let c = f.add("a note", &[], Origin::Source)[0];
    f.complete(c);
    let fp = f
        .cfg
        .expected(&f.db, SubjectKind::Graph, 0, Step::Communities)
        .unwrap()
        .unwrap();
    f.db.record_step(
        SubjectKind::Graph,
        0,
        Step::Communities,
        &fp,
        StepState::Current,
        None,
    )
    .unwrap();
    assert!(f.work().is_empty());
}

#[test]
fn failed_chunks_do_not_block_communities() {
    let f = fx();
    let c = f.add("a note", &[], Origin::Source)[0];
    f.complete(c);
    let d = f.add("another note", &[], Origin::Source)[0];
    for step in [Step::Embed, Step::Extract] {
        let fp = f
            .cfg
            .expected(&f.db, SubjectKind::Chunk, d, step)
            .unwrap()
            .unwrap();
        f.db.record_step(
            SubjectKind::Chunk,
            d,
            step,
            &fp,
            StepState::Failed,
            Some("bad"),
        )
        .unwrap();
    }
    assert!(
        f.work()
            .contains(&(SubjectKind::Graph, 0, Step::Communities))
    );
}

#[test]
fn report_counts_per_group() {
    let mut f = fx();
    let a = f.add("paper text", &["papers"], Origin::Source)[0];
    let b = f.add("note text", &["conversations"], Origin::Source)[0];
    f.complete(a);
    f.complete(b);
    f.cfg.extraction.prompt_version = "v2".into();
    // Only a's extraction is re-recorded under the new config.
    let fp = f
        .cfg
        .expected(&f.db, SubjectKind::Chunk, a, Step::Extract)
        .unwrap()
        .unwrap();
    f.db.record_step(
        SubjectKind::Chunk,
        a,
        Step::Extract,
        &fp,
        StepState::Current,
        None,
    )
    .unwrap();
    let r = f.db.status_report(&f.cfg).unwrap();
    assert_eq!(r.per_group["papers"].current.get(&Step::Extract), Some(&1));
    assert_eq!(
        r.per_group["conversations"].stale.get(&Step::Extract),
        Some(&1)
    );
    assert_eq!(r.per_group["papers"].stale.get(&Step::Extract), None);
    assert!(!r.is_complete());
}

#[test]
fn complete_means_no_missing_or_stale() {
    let f = fx();
    let a = f.add("good", &[], Origin::Source)[0];
    f.complete(a);
    let b = f.add("bad", &[], Origin::Source)[0];
    f.complete(b);
    let fp = f
        .cfg
        .expected(&f.db, SubjectKind::Chunk, b, Step::Extract)
        .unwrap()
        .unwrap();
    f.db.record_step(
        SubjectKind::Chunk,
        b,
        Step::Extract,
        &fp,
        StepState::Failed,
        Some("malformed"),
    )
    .unwrap();
    f.finish_graph();
    let r = f.db.status_report(&f.cfg).unwrap();
    assert_eq!(r.total.failed.get(&Step::Extract), Some(&1));
    assert_eq!(r.failed_samples.len(), 1);
    assert!(r.is_complete(), "failed is reported separately");
}

#[test]
fn compiled_documents_are_not_extracted() {
    let f = fx();
    let c = f.add("a wiki page", &["wiki"], Origin::Compiled)[0];
    let w = f.work();
    assert!(w.contains(&(SubjectKind::Chunk, c, Step::Embed)));
    assert!(!w.contains(&(SubjectKind::Chunk, c, Step::Extract)));
    f.complete(c);
    f.finish_graph();
    let r = f.db.status_report(&f.cfg).unwrap();
    assert!(r.is_complete());
    assert_eq!(r.total.missing.get(&Step::Extract), None);
}
