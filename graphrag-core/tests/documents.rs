//! Task 6: content-addressed documents with sections and doc_key versioning.

use graphrag_core::Database;
use graphrag_core::documents::{AddDocument, AddOutcome, Origin};
use graphrag_core::fingerprints::ChunkerProfile;
use rusqlite::Connection;
use tempfile::TempDir;

struct Fx {
    dir: TempDir,
    db: Database,
    profile: ChunkerProfile,
}

fn fx() -> Fx {
    let dir = TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("t.db")).unwrap();
    db.create_store("memory", 3).unwrap();
    Fx {
        dir,
        db,
        profile: ChunkerProfile::default_markdown(),
    }
}

impl Fx {
    fn add(&self, content: &str, groups: &[&str], doc_key: Option<&str>) -> AddOutcome {
        self.db
            .add_document(&AddDocument {
                store: "memory",
                content,
                origin: Origin::Source,
                source: Some("test"),
                doc_key,
                title: None,
                groups,
                surface: "cli",
                profile: &self.profile,
            })
            .unwrap()
    }
    fn raw(&self) -> Connection {
        Connection::open(self.dir.path().join("t.db")).unwrap()
    }
    fn count(&self, sql: &str) -> i64 {
        self.raw().query_row(sql, [], |r| r.get(0)).unwrap()
    }
}

fn doc_id(o: &AddOutcome) -> i64 {
    match o {
        AddOutcome::Created { document_id }
        | AddOutcome::Existing { document_id, .. }
        | AddOutcome::Replaced { document_id, .. } => *document_id,
    }
}

#[test]
fn identical_text_is_stored_once() {
    let f = fx();
    let a = f.add("the same note", &["conversations"], None);
    let b = f.add("the same note", &["conversations"], None);
    assert!(matches!(a, AddOutcome::Created { .. }));
    assert!(matches!(b, AddOutcome::Existing { new_groups: 0, .. }));
    assert_eq!(doc_id(&a), doc_id(&b));
    assert_eq!(f.count("SELECT COUNT(*) FROM documents"), 1);
    assert_eq!(f.count("SELECT COUNT(*) FROM document_adds"), 2);
    assert_eq!(
        f.count("SELECT COUNT(*) FROM chunks"),
        1,
        "no second set of chunks"
    );
}

#[test]
fn re_add_with_new_group_adds_one_edge() {
    let f = fx();
    let a = f.add("a paper", &["papers"], None);
    let b = f.add("a paper", &["papers", "conversations"], None);
    assert!(matches!(b, AddOutcome::Existing { new_groups: 1, .. }));
    let node = f.db.document_node(doc_id(&a)).unwrap();
    assert_eq!(
        f.db.groups_of(node).unwrap(),
        vec!["conversations".to_string(), "papers".to_string()]
    );
}

#[test]
fn doc_key_change_creates_replacement() {
    let f = fx();
    let a = f.add("version one", &["mica-docs"], Some("docs/a.md"));
    let b = f.add("version two", &["mica-docs"], Some("docs/a.md"));
    let (new, old) = match b {
        AddOutcome::Replaced {
            document_id,
            replaced,
        } => (document_id, replaced),
        other => panic!("expected Replaced, got {other:?}"),
    };
    assert_eq!(old, doc_id(&a));
    let active = f.db.active_documents("memory").unwrap();
    assert!(active.contains(&new) && !active.contains(&old));
    assert_eq!(
        f.count(
            "SELECT COUNT(*) FROM assertions WHERE relation = 'replaces' AND retracted_at IS NULL"
        ),
        1
    );
    assert_eq!(
        f.db.chunk_text(f.db.document_chunks(old).unwrap()[0])
            .unwrap(),
        "version one",
        "the old text is kept"
    );
}

#[test]
fn notes_without_key_never_replace() {
    let f = fx();
    let a = doc_id(&f.add("note one", &[], None));
    let b = doc_id(&f.add("note two", &[], None));
    let active = f.db.active_documents("memory").unwrap();
    assert!(active.contains(&a) && active.contains(&b));
}

#[test]
fn sections_become_nodes_with_has_part_edges() {
    let f = fx();
    let d = doc_id(&f.add("# A\n## B\ntext", &[], None));
    let node = f.db.document_node(d).unwrap();
    let secs = f.db.document_sections(d).unwrap();
    assert_eq!(secs.len(), 2);
    let (a, b) = (&secs[0], &secs[1]);
    assert_eq!(a.heading_path, "A");
    assert_eq!(b.heading_path, "A › B");
    let has_part = |head: i64, tail: i64| {
        f.db.active_assertions(head).unwrap().iter().any(|x| {
            x.relation == "has_part"
                && x.head_id == head
                && x.tail_id == tail
                && x.asserter == "agent:structure"
        })
    };
    assert!(has_part(node, a.node_id));
    assert!(has_part(a.node_id, b.node_id));
    assert!(!has_part(node, b.node_id));
    let chunk_sections: Vec<Option<i64>> =
        f.db.document_chunks(d)
            .unwrap()
            .iter()
            .map(|c| f.db.chunk_section(*c).unwrap())
            .collect();
    assert!(
        chunk_sections.contains(&Some(b.id)),
        "B holds the text chunk"
    );
}

#[test]
fn chunk_step_is_recorded_current() {
    let f = fx();
    let d = doc_id(&f.add("some text", &[], None));
    let fp: String = f
        .raw()
        .query_row(
            "SELECT fp FROM step_status WHERE subject_kind='document' AND subject_id=?1 AND step='chunk' AND state='current'",
            [d],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        fp,
        graphrag_core::fingerprints::chunk_plan_fp("some text", &f.profile)
    );
}

#[test]
fn rechunk_keeps_ids_of_unchanged_chunks() {
    let f = fx();
    let long: String = (0..600).map(|i| format!("word{i} ")).collect();
    let text = format!("# Short\nshort body\n# Long\n{long}\n");
    let d = doc_id(&f.add(&text, &[], None));
    let before = f.db.document_chunks(d).unwrap();
    let short_chunk = before[0];
    let other = ChunkerProfile {
        overlap: 50,
        ..f.profile.clone()
    };
    let report = f.db.rechunk_document(d, &other).unwrap();
    assert!(
        report.kept >= 1 && report.removed >= 1 && report.added >= 1,
        "{report:?}"
    );
    let after = f.db.document_chunks(d).unwrap();
    assert!(
        after.contains(&short_chunk),
        "untouched section keeps its chunk id"
    );
    assert!(!after.is_empty());
    assert_eq!(
        f.count("SELECT COUNT(*) FROM documents"),
        1,
        "text untouched"
    );
}

#[test]
fn empty_note_is_rejected() {
    let f = fx();
    let err =
        f.db.add_document(&AddDocument {
            store: "memory",
            content: "  \n\t ",
            origin: Origin::Source,
            source: None,
            doc_key: None,
            title: None,
            groups: &[],
            surface: "cli",
            profile: &f.profile,
        })
        .unwrap_err();
    assert!(err.to_string().contains("empty document"), "{err}");
}

#[test]
fn compiled_origin_is_stored() {
    let f = fx();
    let d =
        f.db.add_document(&AddDocument {
            store: "memory",
            content: "a wiki page",
            origin: Origin::Compiled,
            source: None,
            doc_key: Some("wiki/x.md"),
            title: Some("X"),
            groups: &["wiki"],
            surface: "cli",
            profile: &f.profile,
        })
        .unwrap();
    assert_eq!(f.db.document_origin(doc_id(&d)).unwrap(), Origin::Compiled);
}

#[test]
fn rechunk_works_for_converted_documents() {
    // add_chunk leaves a node-less document, the shape v1 conversion produces.
    let f = fx();
    let long: String = (0..900).map(|i| format!("w{i} ")).collect();
    let chunk =
        f.db.add_chunk("memory", &long, Some("brain"), None)
            .unwrap();
    let doc: i64 = f
        .raw()
        .query_row(
            "SELECT document_id FROM chunks WHERE id = ?1",
            [chunk],
            |r| r.get(0),
        )
        .unwrap();
    let report = f.db.rechunk_document(doc, &f.profile).unwrap();
    assert!(report.added >= 2, "{report:?}");
    assert!(
        f.db.document_node(doc).is_ok(),
        "a node is created on demand"
    );
    for c in f.db.document_chunks(doc).unwrap() {
        assert!(f.db.chunk_text(c).unwrap().chars().count() <= 2000);
    }
}

mod gc {
    use super::*;
    use graphrag_core::attempts::{
        AttemptOutcome, AttemptRecord, ExtractedEntity, ExtractedRelation,
    };
    use graphrag_core::fingerprints::extract_fp;
    use graphrag_core::paper::ExtractionAttemptIdentityParts;

    fn commit(f: &Fx, chunk: i64, names: &[&str]) {
        let text = f.db.chunk_text(chunk).unwrap();
        let fp = extract_fp(&ExtractionAttemptIdentityParts {
            store: "memory",
            chunk_id: chunk,
            chunk_content: &text,
            model: "m",
            extraction_profile: "p",
            prompt_version: "v",
            response_schema_version: "s",
            glean_max_rounds: 0,
        })
        .unwrap();
        let entities = names
            .iter()
            .map(|n| ExtractedEntity {
                name: n.to_string(),
                entity_type: "concept".into(),
                description: String::new(),
                kind: None,
                is_class: false,
                broader: None,
            })
            .collect();
        let relations = names
            .windows(2)
            .map(|w| ExtractedRelation {
                source: w[0].into(),
                target: w[1].into(),
                relation: "uses".into(),
                description: String::new(),
                strength: None,
            })
            .collect();
        f.db.commit_attempt(&AttemptRecord {
            chunk_id: chunk,
            attempt_fp: &fp,
            model: "m",
            profile: "p",
            prompt_version: "v",
            schema_version: "s",
            glean_max_rounds: 0,
            raw_response: &names.join(","),
            outcome: AttemptOutcome::Ok {
                entities,
                relations,
            },
            rejected_entities: 0,
        })
        .unwrap();
    }

    #[test]
    fn rechunk_gcs_entities_only_vanished_chunks_supported() {
        let f = fx();
        let long: String = (0..600).map(|i| format!("word{i} ")).collect();
        let d = doc_id(&f.add(&format!("# Long\n{long}\n"), &[], None));
        let chunks = f.db.document_chunks(d).unwrap();
        commit(&f, *chunks.last().unwrap(), &["Vanishing", "Ghost"]);
        let other = ChunkerProfile {
            overlap: 50,
            ..f.profile.clone()
        };
        f.db.rechunk_document(d, &other).unwrap();
        assert_eq!(
            f.count("SELECT COUNT(*) FROM entities WHERE kind = 'entity' AND name IN ('Vanishing', 'Ghost')"),
            0,
            "entities supported only by vanished chunks are removed"
        );
        assert_eq!(f.count("SELECT COUNT(*) FROM relations"), 0);
    }

    #[test]
    fn replaced_version_stops_counting_as_support() {
        let f = fx();
        let a = doc_id(&f.add("version one: Tokio uses Mio", &[], Some("a.md")));
        commit(&f, f.db.document_chunks(a).unwrap()[0], &["Tokio", "Mio"]);
        let b = doc_id(&f.add("version two: Tokio uses Mio still", &[], Some("a.md")));
        assert_eq!(
            f.count("SELECT COUNT(*) FROM relations"),
            0,
            "the retired version's extraction is gone"
        );
        commit(&f, f.db.document_chunks(b).unwrap()[0], &["Tokio", "Mio"]);
        let rel: i64 = f
            .raw()
            .query_row("SELECT id FROM relations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(f.db.relation_support(rel).unwrap(), 1);
        assert_eq!(
            f.db.chunk_text(f.db.document_chunks(a).unwrap()[0])
                .unwrap(),
            "version one: Tokio uses Mio",
            "old text kept"
        );
    }

    #[test]
    fn re_adding_older_text_reactivates_it() {
        let f = fx();
        let a = doc_id(&f.add("text A", &[], Some("f.md")));
        let b = doc_id(&f.add("text B", &[], Some("f.md")));
        assert_eq!(f.db.active_documents("memory").unwrap(), vec![b]);
        let again = doc_id(&f.add("text A", &[], Some("f.md")));
        assert_eq!(again, a);
        assert_eq!(
            f.db.active_documents("memory").unwrap(),
            vec![a],
            "A is current again, B retired"
        );
    }
}
