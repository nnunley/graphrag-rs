//! Content-addressed documents. The text is stored once, in an immutable
//! `documents` row; sections and chunks are spans into it. Adding text that
//! already exists only records the add and asserts any new group edges.
//! A `doc_key` (file path, page id) versions a document: new text under an
//! existing key is a new document that `replaces` the old one, whose text
//! is kept.

use crate::assertions::Asserter;
use crate::chunk_plan::plan_chunks;
use crate::db::Database;
use crate::error::GraphRagError;
use crate::fingerprints::{ChunkerProfile, chunk_plan_fp, document_fp};
use crate::nodes::NodeKind;
use crate::sections::SectionSpan;
use rusqlite::{OptionalExtension, params};
use std::collections::HashMap;

/// Separator in stored heading paths: `approval-workflow.md › Load`.
pub const HEADING_PATH_SEP: &str = " › ";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// States things first-hand (a note, a paper, a doc page); extracted.
    Source,
    /// A synthesis of other sources (a wiki page); searchable, not extracted.
    Compiled,
}

impl Origin {
    pub fn as_str(self) -> &'static str {
        match self {
            Origin::Source => "source",
            Origin::Compiled => "compiled",
        }
    }
}

pub struct AddDocument<'a> {
    pub store: &'a str,
    pub content: &'a str,
    pub origin: Origin,
    pub source: Option<&'a str>,
    pub doc_key: Option<&'a str>,
    pub title: Option<&'a str>,
    pub groups: &'a [&'a str],
    /// Which surface added it (`cli`, `mcp`, `migration`, …); the asserter
    /// of its group edges is `agent:<surface>`.
    pub surface: &'a str,
    pub profile: &'a ChunkerProfile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddOutcome {
    Created { document_id: i64 },
    Existing { document_id: i64, new_groups: usize },
    Replaced { document_id: i64, replaced: i64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RechunkReport {
    pub kept: usize,
    pub added: usize,
    pub removed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkContext {
    pub document_id: i64,
    pub heading_path: Vec<String>,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionRow {
    pub id: i64,
    pub node_id: i64,
    pub parent_id: Option<i64>,
    pub heading: String,
    pub heading_path: String,
    pub level: u8,
    pub ordinal: usize,
    pub start: usize,
    pub end: usize,
}

fn structure() -> Asserter {
    Asserter("agent:structure".into())
}

impl Database {
    pub fn add_document(&self, req: &AddDocument<'_>) -> Result<AddOutcome, GraphRagError> {
        if req.content.trim().is_empty() {
            return Err(GraphRagError::InvalidInput("empty document".into()));
        }
        self.get_store(req.store)?;
        let by = Asserter::new(&format!("agent:{}", req.surface))?;
        let fp = document_fp(req.content);
        let tx = self.conn().unchecked_transaction()?;

        let existing: Option<i64> = tx
            .query_row(
                "SELECT id FROM documents WHERE content_fp = ?1",
                params![fp],
                |r| r.get(0),
            )
            .optional()?;
        let outcome = if let Some(document_id) = existing {
            let node = self.ensure_document_node(document_id, req.store, &fp)?;
            let mut new_groups = 0;
            for g in req.groups {
                let gid = self.get_or_create_group(req.store, g)?;
                let before = self.active_assertions(node)?.len();
                self.assert_edge(node, "part_of", gid, &by, None, None)?;
                if self.active_assertions(node)?.len() > before {
                    new_groups += 1;
                }
            }
            if !self.document_has_chunks(document_id)? {
                self.write_structure(
                    document_id,
                    node,
                    req.store,
                    req.content,
                    req.source,
                    req.profile,
                )?;
            }
            AddOutcome::Existing {
                document_id,
                new_groups,
            }
        } else {
            let previous = match req.doc_key {
                Some(key) => self.latest_active_by_key(req.store, key)?,
                None => None,
            };
            tx.execute(
                "INSERT INTO documents (content, content_fp, doc_key, origin, source, title)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    req.content,
                    fp,
                    req.doc_key,
                    req.origin.as_str(),
                    req.source,
                    req.title
                ],
            )?;
            let document_id = tx.last_insert_rowid();
            let node = self.ensure_document_node(document_id, req.store, &fp)?;
            for g in req.groups {
                let gid = self.get_or_create_group(req.store, g)?;
                self.assert_edge(node, "part_of", gid, &by, None, None)?;
            }
            self.write_structure(
                document_id,
                node,
                req.store,
                req.content,
                req.source,
                req.profile,
            )?;
            match previous {
                Some(old) => {
                    let old_node = self.document_node(old)?;
                    self.assert_edge(node, "replaces", old_node, &by, None, None)?;
                    AddOutcome::Replaced {
                        document_id,
                        replaced: old,
                    }
                }
                None => AddOutcome::Created { document_id },
            }
        };
        let document_id = match outcome {
            AddOutcome::Created { document_id }
            | AddOutcome::Existing { document_id, .. }
            | AddOutcome::Replaced { document_id, .. } => document_id,
        };
        tx.execute(
            "INSERT INTO document_adds (document_id, source, surface) VALUES (?1, ?2, ?3)",
            params![document_id, req.source, req.surface],
        )?;
        tx.commit()?;
        Ok(outcome)
    }

    /// The document's graph node, created (`doc:<fp prefix>`) if missing.
    fn ensure_document_node(
        &self,
        document_id: i64,
        store: &str,
        fp: &str,
    ) -> Result<i64, GraphRagError> {
        let node: Option<i64> = self.conn().query_row(
            "SELECT node_id FROM documents WHERE id = ?1",
            params![document_id],
            |r| r.get(0),
        )?;
        if let Some(n) = node {
            return Ok(n);
        }
        let hex = fp.strip_prefix("sha256:").unwrap_or(fp);
        let n = self.create_node(store, NodeKind::Document, &format!("doc:{}", &hex[..16]))?;
        self.conn().execute(
            "UPDATE documents SET node_id = ?2 WHERE id = ?1",
            params![document_id, n],
        )?;
        Ok(n)
    }

    pub fn document_node(&self, document_id: i64) -> Result<i64, GraphRagError> {
        self.conn()
            .query_row(
                "SELECT node_id FROM documents WHERE id = ?1",
                params![document_id],
                |r| r.get::<_, Option<i64>>(0),
            )?
            .ok_or_else(|| GraphRagError::Other(format!("document {document_id} has no node")))
    }

    pub fn document_origin(&self, document_id: i64) -> Result<Origin, GraphRagError> {
        let o: String = self.conn().query_row(
            "SELECT origin FROM documents WHERE id = ?1",
            params![document_id],
            |r| r.get(0),
        )?;
        Ok(if o == "compiled" {
            Origin::Compiled
        } else {
            Origin::Source
        })
    }

    fn document_has_chunks(&self, document_id: i64) -> Result<bool, GraphRagError> {
        Ok(self.conn().query_row(
            "SELECT EXISTS(SELECT 1 FROM chunks WHERE document_id = ?1)",
            params![document_id],
            |r| r.get(0),
        )?)
    }

    fn latest_active_by_key(&self, store: &str, key: &str) -> Result<Option<i64>, GraphRagError> {
        Ok(self
            .conn()
            .query_row(
                "SELECT d.id FROM documents d JOIN entities n ON n.id = d.node_id
                 WHERE d.doc_key = ?2 AND n.store = ?1
                   AND NOT EXISTS (SELECT 1 FROM assertions a
                       WHERE a.tail_id = d.node_id AND a.relation = 'replaces' AND a.retracted_at IS NULL)
                 ORDER BY d.id DESC LIMIT 1",
                params![store, key],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// Document ids in `store` that no active `replaces` assertion retires.
    pub fn active_documents(&self, store: &str) -> Result<Vec<i64>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT d.id FROM documents d JOIN entities n ON n.id = d.node_id
             WHERE n.store = ?1
               AND NOT EXISTS (SELECT 1 FROM assertions a
                   WHERE a.tail_id = d.node_id AND a.relation = 'replaces' AND a.retracted_at IS NULL)
             ORDER BY d.id",
        )?;
        let rows = stmt.query_map(params![store], |r| r.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Chunk ids of a document, in span order.
    pub fn document_chunks(&self, document_id: i64) -> Result<Vec<i64>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT id FROM chunks WHERE document_id = ?1 ORDER BY span_start, span_end, id",
        )?;
        let rows = stmt.query_map(params![document_id], |r| r.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// What the extractor needs to know about a chunk besides its text.
    pub fn chunk_context(&self, chunk_id: i64) -> Result<ChunkContext, GraphRagError> {
        let (document_id, path, text): (i64, Option<String>, String) = self.conn().query_row(
            "SELECT c.document_id, s.heading_path, c.content FROM chunk_view c
             LEFT JOIN sections s ON s.id = c.section_id WHERE c.id = ?1",
            params![chunk_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        Ok(ChunkContext {
            document_id,
            heading_path: path
                .map(|p| p.split(HEADING_PATH_SEP).map(String::from).collect())
                .unwrap_or_default(),
            text,
        })
    }

    pub fn chunk_section(&self, chunk_id: i64) -> Result<Option<i64>, GraphRagError> {
        Ok(self.conn().query_row(
            "SELECT section_id FROM chunks WHERE id = ?1",
            params![chunk_id],
            |r| r.get(0),
        )?)
    }

    pub fn document_sections(&self, document_id: i64) -> Result<Vec<SectionRow>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT id, node_id, parent_id, heading, heading_path, level, ordinal, span_start, span_end
             FROM sections WHERE document_id = ?1 ORDER BY span_start, level, id",
        )?;
        let rows = stmt.query_map(params![document_id], |r| {
            Ok(SectionRow {
                id: r.get(0)?,
                node_id: r.get(1)?,
                parent_id: r.get(2)?,
                heading: r.get(3)?,
                heading_path: r.get(4)?,
                level: r.get::<_, i64>(5)? as u8,
                ordinal: r.get::<_, i64>(6)? as usize,
                start: r.get::<_, i64>(7)? as usize,
                end: r.get::<_, i64>(8)? as usize,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Sections (as nodes with `has_part` edges) and chunks for a document.
    fn write_structure(
        &self,
        document_id: i64,
        doc_node: i64,
        store: &str,
        content: &str,
        source: Option<&str>,
        profile: &ChunkerProfile,
    ) -> Result<(), GraphRagError> {
        let (sections, chunks) = plan_chunks(content, profile);
        let section_ids = self.sync_sections(document_id, doc_node, store, content, &sections)?;
        for c in &chunks {
            self.conn().execute(
                "INSERT INTO chunks (store, document_id, section_id, span_start, span_end, content_fp, source, priority)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1)",
                params![
                    store,
                    document_id,
                    c.section.map(|i| section_ids[i]),
                    c.start as i64,
                    c.end as i64,
                    c.content_fp,
                    source
                ],
            )?;
        }
        self.mark_chunked(document_id, content, profile)
    }

    fn mark_chunked(
        &self,
        document_id: i64,
        content: &str,
        profile: &ChunkerProfile,
    ) -> Result<(), GraphRagError> {
        self.conn().execute(
            "INSERT INTO step_status (subject_kind, subject_id, step, fp, state, updated_at)
             VALUES ('document', ?1, 'chunk', ?2, 'current', strftime('%Y-%m-%dT%H:%M:%fZ','now'))
             ON CONFLICT(subject_kind, subject_id, step) DO UPDATE SET
                 fp = excluded.fp, state = 'current', last_error = NULL, updated_at = excluded.updated_at",
            params![document_id, chunk_plan_fp(content, profile)],
        )?;
        Ok(())
    }

    /// Make the stored sections match `sections`, keeping the id of every
    /// section whose identity `(heading path, ordinal)` is unchanged.
    /// Returns the section row id for each planned section.
    fn sync_sections(
        &self,
        document_id: i64,
        doc_node: i64,
        store: &str,
        content: &str,
        sections: &[SectionSpan],
    ) -> Result<Vec<i64>, GraphRagError> {
        let existing: HashMap<(String, usize), SectionRow> = self
            .document_sections(document_id)?
            .into_iter()
            .map(|s| ((s.heading_path.clone(), s.ordinal), s))
            .collect();
        let mut ids: Vec<i64> = Vec::with_capacity(sections.len());
        let mut kept = Vec::new();
        for s in sections {
            let path = s.heading_path.join(HEADING_PATH_SEP);
            let parent_id = s.parent.map(|p| ids[p]);
            let fp = crate::token::fingerprint_text(&content[s.start..s.end]);
            if let Some(row) = existing.get(&(path.clone(), s.ordinal)) {
                self.conn().execute(
                    "UPDATE sections SET parent_id = ?2, span_start = ?3, span_end = ?4, content_fp = ?5
                     WHERE id = ?1",
                    params![row.id, parent_id, s.start as i64, s.end as i64, fp],
                )?;
                ids.push(row.id);
                kept.push(row.id);
                continue;
            }
            self.conn().execute(
                "INSERT INTO sections
                     (document_id, parent_id, heading, level, ordinal, heading_path, span_start, span_end, content_fp)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    document_id,
                    parent_id,
                    s.heading,
                    s.level as i64,
                    s.ordinal as i64,
                    path,
                    s.start as i64,
                    s.end as i64,
                    fp
                ],
            )?;
            let id = self.conn().last_insert_rowid();
            let node = self.create_node(store, NodeKind::Section, &format!("sec:{id}"))?;
            self.conn().execute(
                "UPDATE sections SET node_id = ?2 WHERE id = ?1",
                params![id, node],
            )?;
            let parent_node = match parent_id {
                Some(p) => self.conn().query_row(
                    "SELECT node_id FROM sections WHERE id = ?1",
                    params![p],
                    |r| r.get(0),
                )?,
                None => doc_node,
            };
            self.assert_edge(parent_node, "has_part", node, &structure(), None, None)?;
            ids.push(id);
        }
        for (_, row) in existing {
            if !kept.contains(&row.id) {
                self.conn().execute(
                    "UPDATE chunks SET section_id = NULL WHERE section_id = ?1",
                    params![row.id],
                )?;
                self.conn().execute(
                    "DELETE FROM assertions WHERE (head_id = ?1 OR tail_id = ?1) AND asserter = 'agent:structure'",
                    params![row.node_id],
                )?;
                self.conn()
                    .execute("DELETE FROM sections WHERE id = ?1", params![row.id])?;
                self.conn()
                    .execute("DELETE FROM entities WHERE id = ?1", params![row.node_id])?;
            }
        }
        Ok(ids)
    }

    /// Re-cut a document under a new chunker profile. Its text is untouched.
    /// Chunks whose text still appears in the new plan keep their id (and
    /// everything downstream); new ones are added; vanished ones are deleted
    /// along with their attempts, links and status.
    pub fn rechunk_document(
        &self,
        document_id: i64,
        profile: &ChunkerProfile,
    ) -> Result<RechunkReport, GraphRagError> {
        let tx = self.conn().unchecked_transaction()?;
        // v1-converted documents have no node yet: take the store from their
        // chunks and create the node on demand.
        let (content, fp, store, source): (String, String, String, Option<String>) = tx.query_row(
            "SELECT d.content, d.content_fp,
                    COALESCE(n.store, (SELECT c.store FROM chunks c WHERE c.document_id = d.id LIMIT 1)),
                    d.source
             FROM documents d LEFT JOIN entities n ON n.id = d.node_id WHERE d.id = ?1",
            params![document_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )?;
        let doc_node = self.ensure_document_node(document_id, &store, &fp)?;
        let (sections, planned) = plan_chunks(&content, profile);
        let section_ids = self.sync_sections(document_id, doc_node, &store, &content, &sections)?;

        let mut old: HashMap<String, Vec<i64>> = HashMap::new();
        {
            let mut stmt =
                tx.prepare("SELECT id, content_fp FROM chunks WHERE document_id = ?1 ORDER BY id")?;
            for row in stmt.query_map(params![document_id], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })? {
                let (id, fp) = row?;
                old.entry(fp).or_default().push(id);
            }
        }
        let mut report = RechunkReport {
            kept: 0,
            added: 0,
            removed: 0,
        };
        for c in &planned {
            let section_id = c.section.map(|i| section_ids[i]);
            let reuse = old
                .get_mut(&c.content_fp)
                .and_then(|ids| (!ids.is_empty()).then(|| ids.remove(0)));
            if let Some(id) = reuse {
                tx.execute(
                    "UPDATE chunks SET section_id = ?2, span_start = ?3, span_end = ?4 WHERE id = ?1",
                    params![id, section_id, c.start as i64, c.end as i64],
                )?;
                report.kept += 1;
            } else {
                tx.execute(
                    "INSERT INTO chunks (store, document_id, section_id, span_start, span_end, content_fp, source, priority)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1)",
                    params![store, document_id, section_id, c.start as i64, c.end as i64, c.content_fp, source],
                )?;
                report.added += 1;
            }
        }
        for id in old.into_values().flatten() {
            tx.execute(
                "DELETE FROM step_status WHERE subject_kind = 'chunk' AND subject_id = ?1",
                params![id],
            )?;
            tx.execute("DELETE FROM chunks WHERE id = ?1", params![id])?;
            report.removed += 1;
        }
        if planned.is_empty() {
            return Err(GraphRagError::Other(format!(
                "re-chunking document {document_id} produced no chunks"
            )));
        }
        self.mark_chunked(document_id, &content, profile)?;
        tx.commit()?;
        Ok(report)
    }
}
