//! One-time migration: the old per-store layout becomes one graph.
//!
//! Every old store becomes a group; every note, lesson and doc page becomes
//! a content-addressed document `part_of` its group (asserter
//! `agent:migration`). Docs that exist as files are re-imported whole (their
//! headings become sections); the rest are reassembled from their chunks in
//! order. Paper copies are dropped and the paper is imported once into
//! `papers`. The old entity graph is deleted (no retention). Embeddings are
//! re-attached where a new chunk's text equals an old one's. `apply`
//! recomputes the plan inside its transaction and refuses if it differs
//! from the one shown, or if any source label is unresolved.

use crate::db::Database;
use crate::documents::{AddDocument, AddOutcome, Origin};
use crate::error::GraphRagError;
use crate::fingerprints::{ChunkerProfile, embed_fp};
use crate::status::{Step, StepState, SubjectKind};
use crate::token::fingerprint_text;
use rusqlite::params;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// The single store the migration creates.
pub const MEMORY_STORE: &str = "memory";

/// Where file-backed sources live, and the embedder whose vectors the old
/// chunks carry.
#[derive(Debug, Clone)]
pub struct MigrationSources {
    /// Root of the mica repo (`mica-docs` labels are `path.md#Heading`).
    pub mica_root: PathBuf,
    /// Directory holding project repos (`operation` labels are `project:doc#N`).
    pub dev_root: PathBuf,
    /// The GraphRAG paper, imported once into `papers`.
    pub paper_path: PathBuf,
    pub embedder_id: String,
    pub dim: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct GroupPlan {
    pub from_files: usize,
    pub from_chunk_groups: usize,
    pub single_chunk: usize,
    /// Documents already in `memory` (written by a v2 binary before the
    /// migration ran), kept with their groups.
    pub kept: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct DeletionCounts {
    pub entities: usize,
    pub relations: usize,
    pub communities: usize,
    pub extractions: usize,
    /// Paper copies dropped in favour of the one re-import.
    pub paper_chunks: usize,
    /// Every old chunk (all are replaced by the new documents' chunks).
    pub chunks: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct MigrationPlan {
    pub per_group: BTreeMap<String, GroupPlan>,
    pub deletions: DeletionCounts,
    /// `"store: label"` for every source the migration can't place.
    pub unresolved: Vec<String>,
}

impl MigrationPlan {
    /// Identity of this plan, so `--apply` can be bound to the one reviewed.
    pub fn fingerprint(&self) -> String {
        fingerprint_text(&serde_json::to_string(self).unwrap_or_default())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct MigrationReport {
    pub documents: usize,
    pub chunks: usize,
    pub sections: usize,
    pub embeddings_reattached: usize,
    pub deleted_documents: usize,
}

/// One document the migration will add.
#[derive(Debug, Clone)]
enum Unit {
    File {
        group: String,
        path: PathBuf,
        doc_key: String,
    },
    Assembled {
        group: String,
        label: String,
        parts: Vec<String>,
    },
    Single {
        group: String,
        text: String,
        source: Option<String>,
        created_at: String,
    },
    /// A document already in `memory`, re-asserted into its groups.
    Kept {
        groups: Vec<String>,
        text: String,
        source: Option<String>,
        doc_key: Option<String>,
        created_at: String,
    },
}

/// `(id, content, source, doc_key, created_at, node_id)` of a document already in `memory`.
type MemoryDoc = (
    i64,
    String,
    Option<String>,
    Option<String>,
    String,
    Option<i64>,
);

struct OldChunk {
    store: String,
    text: String,
    source: Option<String>,
    created_at: String,
}

fn old_chunks(db: &Database) -> Result<Vec<OldChunk>, GraphRagError> {
    let mut stmt = db.conn().prepare(
        "SELECT store, content, source, COALESCE(created_at, '') FROM chunk_view ORDER BY id",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(OldChunk {
            store: r.get(0)?,
            text: r.get(1)?,
            source: r.get(2)?,
            created_at: r.get(3)?,
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// `project:doc#N` → `(project, doc, N)`.
fn repo_label(source: &str) -> Option<(&str, &str, u64)> {
    let (label, n) = source.rsplit_once('#')?;
    let (project, doc) = label.split_once(':')?;
    let n = n.parse().ok()?;
    (!project.is_empty() && !doc.is_empty()).then_some((project, doc, n))
}

fn count(db: &Database, sql: &str) -> Result<usize, GraphRagError> {
    Ok(db.conn().query_row(sql, [], |r| r.get::<_, i64>(0))? as usize)
}

fn plan_units(
    db: &Database,
    src: &MigrationSources,
) -> Result<(Vec<Unit>, MigrationPlan), GraphRagError> {
    let mut plan = MigrationPlan::default();
    let mut units: Vec<Unit> = Vec::new();
    // label → index in `units`, so file/assembled units are created once.
    let mut by_label: HashMap<String, usize> = HashMap::new();
    let mut assembled: BTreeMap<String, Vec<(u64, String)>> = BTreeMap::new();
    let chunks = old_chunks(db)?;
    plan.deletions.chunks = chunks.len();

    for c in &chunks {
        let source = c.source.clone().unwrap_or_default();
        let single = |group: &str| Unit::Single {
            group: group.into(),
            text: c.text.clone(),
            source: c.source.clone(),
            created_at: c.created_at.clone(),
        };
        match c.store.as_str() {
            MEMORY_STORE => {} // already one graph: kept per document below
            "graphrag-paper" | "graphrag-paper-dense" => plan.deletions.paper_chunks += 1,
            "conversations" | "9front-kb" => units.push(single(&c.store)),
            "mica-docs" => {
                let (path, heading) = source.split_once('#').unwrap_or((source.as_str(), ""));
                let file = src.mica_root.join(path);
                let key = format!("mica-docs:{path}");
                if file.is_file() {
                    by_label.entry(key).or_insert_with(|| {
                        units.push(Unit::File {
                            group: "mica-docs".into(),
                            path: file,
                            doc_key: path.into(),
                        });
                        units.len() - 1
                    });
                } else if path.is_empty() {
                    plan.unresolved.push(format!("mica-docs: {source}"));
                } else {
                    let text = if heading.is_empty() || c.text.trim_start().starts_with('#') {
                        c.text.clone()
                    } else {
                        format!("# {heading}\n\n{}", c.text)
                    };
                    let parts = assembled.entry(key).or_default();
                    let n = parts.len() as u64;
                    parts.push((n, text));
                }
            }
            "operation" => {
                if source.starts_with("lesson:") || source.starts_with("operational:") {
                    units.push(single("operation"));
                } else if source.starts_with("graphrag-rs:microsoft-graphrag-paper#") {
                    plan.deletions.paper_chunks += 1;
                } else if let Some((project, doc, n)) = repo_label(&source) {
                    let key = format!("operation:{project}:{doc}");
                    let candidates = [
                        (
                            src.dev_root
                                .join(project)
                                .join("docs")
                                .join(format!("{doc}.md")),
                            format!("{project}/docs/{doc}.md"),
                        ),
                        (
                            src.dev_root.join(project).join(format!("{doc}.md")),
                            format!("{project}/{doc}.md"),
                        ),
                    ];
                    match candidates.into_iter().find(|(p, _)| p.is_file()) {
                        Some((path, doc_key)) => {
                            by_label.entry(key).or_insert_with(|| {
                                units.push(Unit::File {
                                    group: "operation".into(),
                                    path,
                                    doc_key,
                                });
                                units.len() - 1
                            });
                        }
                        None => assembled.entry(key).or_default().push((n, c.text.clone())),
                    }
                } else {
                    plan.unresolved.push(format!("operation: {source}"));
                }
            }
            other => plan.unresolved.push(format!("{other}: {source}")),
        }
    }
    // Documents a v2 binary already wrote, in id order (so doc_key
    // versions are re-added oldest first and stay retired in order).
    {
        let rows: Vec<MemoryDoc> = {
            let mut stmt = db.conn().prepare(
                "SELECT d.id, d.content, d.source, d.doc_key, COALESCE(d.created_at, ''), d.node_id FROM documents d
                 WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id AND c.store = ?1)
                 ORDER BY d.id",
            )?;
            stmt.query_map(params![MEMORY_STORE], |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                ))
            })?
            .collect::<Result<_, _>>()?
        };
        for (_, text, source, doc_key, created_at, node) in rows {
            let mut groups = match node {
                Some(n) => db.groups_of(n)?,
                None => Vec::new(),
            };
            if groups.is_empty() {
                groups.push("conversations".into());
            }
            units.push(Unit::Kept {
                groups,
                text,
                source,
                doc_key,
                created_at,
            });
        }
    }
    for (key, mut parts) in assembled {
        parts.sort_by_key(|p| p.0);
        let (group, label) = key.split_once(':').unwrap_or(("operation", key.as_str()));
        units.push(Unit::Assembled {
            group: group.into(),
            label: label.into(),
            parts: parts.into_iter().map(|p| p.1).collect(),
        });
    }
    if src.paper_path.is_file() {
        units.push(Unit::File {
            group: "papers".into(),
            path: src.paper_path.clone(),
            doc_key: "papers/microsoft-graphrag-paper.md".into(),
        });
    } else {
        plan.unresolved
            .push(format!("papers: {} missing", src.paper_path.display()));
    }
    for u in &units {
        let (group, field): (&str, fn(&mut GroupPlan) -> &mut usize) = match u {
            Unit::File { group, .. } => (group, |g| &mut g.from_files),
            Unit::Assembled { group, .. } => (group, |g| &mut g.from_chunk_groups),
            Unit::Single { group, .. } => (group, |g| &mut g.single_chunk),
            Unit::Kept { groups, .. } => (&groups[0], |g| &mut g.kept),
        };
        *field(plan.per_group.entry(group.to_string()).or_default()) += 1;
    }
    plan.deletions.entities = count(db, "SELECT COUNT(*) FROM entities WHERE kind = 'entity'")?;
    plan.deletions.relations = count(db, "SELECT COUNT(*) FROM relations")?;
    plan.deletions.communities = count(db, "SELECT COUNT(*) FROM communities")?;
    plan.deletions.extractions = count(db, "SELECT COUNT(*) FROM extractions")?;
    Ok((units, plan))
}

/// What the migration would do. Read-only.
pub fn plan_v2(db: &Database, sources: &MigrationSources) -> Result<MigrationPlan, GraphRagError> {
    Ok(plan_units(db, sources)?.1)
}

fn read(path: &Path) -> Result<String, GraphRagError> {
    std::fs::read_to_string(path)
        .map_err(|e| GraphRagError::Other(format!("read {}: {e}", path.display())))
}

/// Apply the migration in one transaction.
pub fn apply_v2_migration(
    db: &Database,
    plan: &MigrationPlan,
    sources: &MigrationSources,
) -> Result<MigrationReport, GraphRagError> {
    let conn = db.conn();
    let tx = db.write_tx()?;
    let (units, current) = plan_units(db, sources)?;
    if !current.unresolved.is_empty() {
        return Err(GraphRagError::InvalidInput(format!(
            "unresolved sources: {}",
            current.unresolved.join("; ")
        )));
    }
    if &current != plan {
        return Err(GraphRagError::InvalidInput(
            "the database changed since the plan was made; re-run --plan".into(),
        ));
    }

    // Old vectors by trimmed text, to re-attach where text is unchanged.
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT c.content, e.embedding FROM chunk_view c JOIN chunk_embeddings e ON e.chunk_id = c.id",
        )?;
        for row in stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
        })? {
            let (text, blob) = row?;
            let v: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            vectors.entry(fingerprint_text(text.trim())).or_insert(v);
        }
    }

    // Clean slate (Norman: no retention for the old graph).
    conn.execute_batch(
        "DELETE FROM chunk_entities;
         DELETE FROM entity_communities;
         DELETE FROM communities;
         DELETE FROM community_runs;
         DELETE FROM summaries;
         DELETE FROM extraction_entities;
         DELETE FROM extraction_relations;
         DELETE FROM extractions;
         DELETE FROM attempt_entities;
         DELETE FROM attempt_relations;
         DELETE FROM attempts;
         DELETE FROM lifted_from;
         DELETE FROM lifted_edges;
         DELETE FROM class_aliases;
         DELETE FROM uses;
         DELETE FROM assertions;
         DELETE FROM step_status;
         DELETE FROM entity_embeddings;
         DELETE FROM entity_merges;
         DELETE FROM relations;
         DELETE FROM chunk_embeddings;
         DELETE FROM chunks;
         DELETE FROM sections;
         DELETE FROM document_adds;
         DELETE FROM entities;",
    )?;
    if db.get_store(MEMORY_STORE).is_err() {
        db.create_store(MEMORY_STORE, sources.dim)?;
    }

    let profile = ChunkerProfile::default_markdown();
    let mut report = MigrationReport::default();
    for u in &units {
        let (groups, content, source, doc_key, created_at): (
            Vec<String>,
            String,
            Option<String>,
            Option<String>,
            Option<String>,
        ) = match u {
            Unit::File {
                group,
                path,
                doc_key,
            } => (
                vec![group.clone()],
                read(path)?,
                Some(doc_key.clone()),
                Some(doc_key.clone()),
                None,
            ),
            Unit::Assembled {
                group,
                label,
                parts,
            } => (
                vec![group.clone()],
                parts.join("\n\n"),
                Some(label.clone()),
                None,
                None,
            ),
            Unit::Single {
                group,
                text,
                source,
                created_at,
            } => (
                vec![group.clone()],
                text.clone(),
                source.clone(),
                None,
                Some(created_at.clone()),
            ),
            Unit::Kept {
                groups,
                text,
                source,
                doc_key,
                created_at,
            } => (
                groups.clone(),
                text.clone(),
                source.clone(),
                doc_key.clone(),
                Some(created_at.clone()),
            ),
        };
        let group_refs: Vec<&str> = groups.iter().map(String::as_str).collect();
        if content.trim().is_empty() {
            continue;
        }
        let outcome = db.add_document(&AddDocument {
            store: MEMORY_STORE,
            content: &content,
            origin: Origin::Source,
            source: source.as_deref(),
            doc_key: doc_key.as_deref(),
            title: None,
            groups: &group_refs,
            surface: "migration",
            profile: &profile,
        })?;
        // Old notes already exist as v1-converted documents; they count too.
        let document_id = match outcome {
            AddOutcome::Created { document_id }
            | AddOutcome::Replaced { document_id, .. }
            | AddOutcome::Existing { document_id, .. } => document_id,
        };
        report.documents += 1;
        // Keep a note's original time, so the log and salience stay honest.
        if let Some(at) = created_at.filter(|a| !a.is_empty()) {
            conn.execute(
                "UPDATE documents SET created_at = ?2 WHERE id = ?1",
                params![document_id, at],
            )?;
            conn.execute(
                "UPDATE chunks SET created_at = ?2 WHERE document_id = ?1",
                params![document_id, at],
            )?;
        }
    }

    // Re-attach vectors for unchanged text.
    {
        let rows: Vec<(i64, String, String)> = {
            let mut stmt =
                conn.prepare("SELECT id, content_fp, content FROM chunk_view WHERE store = ?1")?;
            stmt.query_map(params![MEMORY_STORE], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .collect::<Result<_, _>>()?
        };
        report.chunks = rows.len();
        for (id, fp, text) in rows {
            if let Some(v) = vectors.get(&fp) {
                db.set_chunk_embedding(id, v)?;
                let efp = embed_fp(&text, &sources.embedder_id, sources.dim);
                db.record_step(
                    SubjectKind::Chunk,
                    id,
                    Step::Embed,
                    &efp,
                    StepState::Current,
                    None,
                )?;
                report.embeddings_reattached += 1;
            }
        }
    }
    // The one counted cleanup: old documents no new unit claimed (paper
    // copies, per-section mica rows replaced by their files).
    report.deleted_documents = conn.execute(
        "DELETE FROM documents WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = documents.id)",
        [],
    )?;
    report.sections = count(db, "SELECT COUNT(*) FROM sections")?;
    conn.execute("DELETE FROM stores WHERE name != ?1", params![MEMORY_STORE])?;
    tx.commit()?;
    Ok(report)
}
