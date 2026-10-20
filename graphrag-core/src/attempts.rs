//! Extraction attempts. An attempt is identified by the M0 attempt
//! fingerprint (chunk text + model + profile + prompt + schema), so a chunk
//! has at most one current attempt. Committing a new one supersedes the
//! old in the same transaction; superseded attempts and their instance rows
//! are deleted (no retention). Edge weight is support: the number of
//! distinct evidence units (sections, or the document when it has none)
//! with a current attempt that found the relation.

use crate::db::Database;
use crate::error::GraphRagError;
use crate::fingerprints::extract_fp;
use crate::paper::ExtractionAttemptIdentityParts;
use crate::status::{Step, StepState, SubjectKind};
use crate::synonyms::{
    RelationMapping, normalize_entity_name, normalize_entity_type, normalize_relation,
};
use crate::token::fingerprint_text;
use rusqlite::{OptionalExtension, params};
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub description: String,
    /// The class this entity is an instance of (`version control system`).
    pub kind: Option<String>,
    /// The name itself denotes a class.
    pub is_class: bool,
    /// The next more general class of `kind` (or of the entity, if a class).
    pub broader: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedRelation {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub description: String,
    pub strength: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptOutcome {
    Ok {
        entities: Vec<ExtractedEntity>,
        relations: Vec<ExtractedRelation>,
    },
    /// A valid extraction that found nothing. Terminal.
    Empty,
}

pub struct AttemptRecord<'a> {
    pub chunk_id: i64,
    pub attempt_fp: &'a str,
    pub model: &'a str,
    pub profile: &'a str,
    pub prompt_version: &'a str,
    pub schema_version: &'a str,
    pub glean_max_rounds: u32,
    pub raw_response: &'a str,
    pub outcome: AttemptOutcome,
    pub rejected_entities: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommitReport {
    pub replay: bool,
    pub superseded: Option<String>,
    pub entities: usize,
    pub relations: usize,
    pub unmapped_relations: usize,
    pub implicit_endpoints: usize,
    pub gc_entities: usize,
    pub gc_relations: usize,
}

/// One relation instance to write: `(head, tail, raw, description, strength)`.
type Instance = (i64, i64, String, String, Option<u8>);

impl Database {
    pub fn commit_attempt(&self, rec: &AttemptRecord<'_>) -> Result<CommitReport, GraphRagError> {
        let tx = self.conn().unchecked_transaction()?;
        let (store, text, document_id, section_id): (String, String, i64, Option<i64>) = tx
            .query_row(
                "SELECT store, content, document_id, section_id FROM chunk_view WHERE id = ?1",
                params![rec.chunk_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()?
            .ok_or_else(|| {
                GraphRagError::InvalidInput(format!("chunk {} not found", rec.chunk_id))
            })?;
        let expected = extract_fp(&ExtractionAttemptIdentityParts {
            store: &store,
            chunk_id: rec.chunk_id,
            chunk_content: &text,
            model: rec.model,
            extraction_profile: rec.profile,
            prompt_version: rec.prompt_version,
            response_schema_version: rec.schema_version,
            glean_max_rounds: rec.glean_max_rounds,
        })
        .map_err(|e| GraphRagError::InvalidInput(e.to_string()))?;
        if expected != rec.attempt_fp {
            return Err(GraphRagError::InvalidInput(format!(
                "attempt fingerprint does not match chunk {}'s current content",
                rec.chunk_id
            )));
        }
        let response_sha = fingerprint_text(rec.raw_response);
        let mut report = CommitReport::default();

        let same: Option<String> = tx
            .query_row(
                "SELECT response_sha256 FROM attempts WHERE attempt_fp = ?1",
                params![rec.attempt_fp],
                |r| r.get(0),
            )
            .optional()?;
        if same.as_deref() == Some(response_sha.as_str()) {
            report.replay = true;
            self.record_step(
                SubjectKind::Chunk,
                rec.chunk_id,
                Step::Extract,
                rec.attempt_fp,
                StepState::Current,
                None,
            )?;
            tx.commit()?;
            return Ok(report);
        }

        // Evidence unit and its node (the `uses` source).
        let (evidence_kind, evidence_id, evidence_node): (&str, i64, Option<i64>) = match section_id
        {
            Some(s) => (
                "section",
                s,
                tx.query_row(
                    "SELECT node_id FROM sections WHERE id = ?1",
                    params![s],
                    |r| r.get(0),
                )?,
            ),
            None => (
                "document",
                document_id,
                tx.query_row(
                    "SELECT node_id FROM documents WHERE id = ?1",
                    params![document_id],
                    |r| r.get(0),
                )?,
            ),
        };

        // Supersede: everything the chunk's previous attempts linked becomes a GC candidate.
        let mut cand_entities: BTreeSet<i64> = BTreeSet::new();
        let mut cand_relations: BTreeSet<i64> = BTreeSet::new();
        // Items the evidence unit already stated before this commit: no new mention.
        let mut seen_entities: BTreeSet<i64> = BTreeSet::new();
        let mut seen_relations: BTreeSet<i64> = BTreeSet::new();
        {
            let mut stmt = tx.prepare(
                "SELECT ae.entity_id FROM attempt_entities ae JOIN attempts a ON a.attempt_fp = ae.attempt_fp
                 WHERE a.chunk_id = ?1",
            )?;
            for id in stmt.query_map(params![rec.chunk_id], |r| r.get::<_, i64>(0))? {
                cand_entities.insert(id?);
            }
            let mut stmt = tx.prepare(
                "SELECT ar.relation_id FROM attempt_relations ar JOIN attempts a ON a.attempt_fp = ar.attempt_fp
                 WHERE a.chunk_id = ?1",
            )?;
            for id in stmt.query_map(params![rec.chunk_id], |r| r.get::<_, i64>(0))? {
                cand_relations.insert(id?);
            }
            if let Some(node) = evidence_node {
                let mut stmt = tx.prepare(
                    "SELECT item_kind, item_id FROM uses WHERE kind = 'mention' AND source_id = ?1",
                )?;
                for row in stmt.query_map(params![node], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
                })? {
                    let (k, id) = row?;
                    if k == "relation" {
                        seen_relations.insert(id);
                    } else {
                        seen_entities.insert(id);
                    }
                }
            }
        }
        let previous: Option<String> = tx
            .query_row(
                "SELECT attempt_fp FROM attempts WHERE chunk_id = ?1 ORDER BY created_at DESC LIMIT 1",
                params![rec.chunk_id],
                |r| r.get(0),
            )
            .optional()?;
        report.superseded = previous.filter(|p| p != rec.attempt_fp);
        tx.execute(
            "DELETE FROM attempts WHERE chunk_id = ?1",
            params![rec.chunk_id],
        )?;
        tx.execute(
            "DELETE FROM chunk_entities WHERE chunk_id = ?1",
            params![rec.chunk_id],
        )?;

        let (entities, relations) = match &rec.outcome {
            AttemptOutcome::Ok {
                entities,
                relations,
            } => (entities.as_slice(), relations.as_slice()),
            AttemptOutcome::Empty => (&[][..], &[][..]),
        };

        // Entities, then class edges from kind/broader, then relations.
        let mut by_key: HashMap<String, i64> = HashMap::new();
        let mut entity_rows: Vec<(i64, &'static str, String)> = Vec::new();
        let mut instances: Vec<Instance> = Vec::new();
        let mut class_uses: BTreeSet<i64> = BTreeSet::new();
        for e in entities {
            if e.name.trim().is_empty() {
                continue;
            }
            let id = self.get_or_create_entity(&store, &e.name, None, None)?;
            by_key.insert(normalize_entity_name(&e.name), id);
            let ty = if e.is_class {
                "collection"
            } else {
                normalize_entity_type(&e.entity_type)
            };
            entity_rows.push((id, ty, e.description.clone()));
            let kind = e.kind.as_deref().map(str::trim).filter(|k| !k.is_empty());
            let broader = e
                .broader
                .as_deref()
                .map(str::trim)
                .filter(|b| !b.is_empty());
            if e.is_class {
                self.conn().execute(
                    "UPDATE entities SET entity_type = 'collection' WHERE id = ?1",
                    params![id],
                )?;
                if let Some(b) = broader {
                    let bc = self.resolve_class(&store, b)?;
                    class_uses.insert(bc);
                    if bc != id {
                        instances.push((id, bc, "genls".into(), String::new(), None));
                    }
                }
            } else if let Some(k) = kind {
                let class = self.resolve_class(&store, k)?;
                class_uses.insert(class);
                if class != id {
                    instances.push((id, class, "isa".into(), String::new(), None));
                }
                if let Some(b) = broader {
                    let bc = self.resolve_class(&store, b)?;
                    class_uses.insert(bc);
                    if bc != class {
                        instances.push((class, bc, "genls".into(), String::new(), None));
                    }
                }
            }
        }
        for r in relations {
            let mut endpoint = |name: &str| -> Result<Option<i64>, GraphRagError> {
                if name.trim().is_empty() {
                    return Ok(None);
                }
                let key = normalize_entity_name(name);
                if let Some(id) = by_key.get(&key) {
                    return Ok(Some(*id));
                }
                let id = self.get_or_create_entity(&store, name, None, None)?;
                if self.conn().query_row(
                    "SELECT entity_type IS NULL FROM entities WHERE id = ?1",
                    params![id],
                    |r| r.get::<_, bool>(0),
                )? {
                    self.conn().execute(
                        "UPDATE entities SET entity_type = 'other' WHERE id = ?1",
                        params![id],
                    )?;
                }
                report.implicit_endpoints += 1;
                by_key.insert(key, id);
                entity_rows.push((id, "other", String::new()));
                Ok(Some(id))
            };
            let (Some(h), Some(t)) = (endpoint(&r.source)?, endpoint(&r.target)?) else {
                continue;
            };
            if h == t {
                continue;
            }
            instances.push((h, t, r.relation.clone(), r.description.clone(), r.strength));
        }

        tx.execute(
            "INSERT INTO attempts (attempt_fp, chunk_id, model, profile, prompt_version, schema_version,
                 outcome, response_sha256, raw_response, unmapped_relations, rejected_entities)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0, ?10)",
            params![
                rec.attempt_fp,
                rec.chunk_id,
                rec.model,
                rec.profile,
                rec.prompt_version,
                rec.schema_version,
                if matches!(rec.outcome, AttemptOutcome::Empty) { "empty" } else { "ok" },
                response_sha,
                rec.raw_response,
                rec.rejected_entities as i64
            ],
        )?;
        let mut touched: BTreeSet<i64> = BTreeSet::new();
        for (id, ty, desc) in &entity_rows {
            let n = tx.execute(
                "INSERT OR IGNORE INTO attempt_entities (attempt_fp, entity_id, entity_type, description)
                 VALUES (?1, ?2, ?3, ?4)",
                params![rec.attempt_fp, id, ty, desc],
            )?;
            tx.execute(
                "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?1, ?2)",
                params![rec.chunk_id, id],
            )?;
            if n > 0 {
                report.entities += 1;
                touched.insert(*id);
            }
        }
        let mut written_relations: BTreeSet<i64> = BTreeSet::new();
        for (h, t, raw, desc, strength) in &instances {
            let (relation, canonical) = match normalize_relation(raw) {
                RelationMapping::Canonical(c) => (c.to_string(), Some(c)),
                RelationMapping::Unmapped(u) => {
                    report.unmapped_relations += 1;
                    (u, None)
                }
            };
            let key = canonical.map_or(relation.as_str(), |c| c);
            let existing: Option<i64> = tx
                .query_row(
                    "SELECT id FROM relations WHERE store = ?1 AND head_id = ?2 AND tail_id = ?3
                       AND COALESCE(canonical_relation, relation) = ?4 ORDER BY id LIMIT 1",
                    params![store, h, t, key],
                    |r| r.get(0),
                )
                .optional()?;
            let rid = match existing {
                Some(id) => id,
                None => {
                    tx.execute(
                        "INSERT INTO relations (store, head_id, tail_id, relation, canonical_relation)
                         VALUES (?1, ?2, ?3, ?4, ?5)",
                        params![store, h, t, relation, canonical],
                    )?;
                    tx.last_insert_rowid()
                }
            };
            tx.execute(
                "INSERT OR IGNORE INTO attempt_relations
                     (attempt_fp, relation_id, raw_relation, description, strength, evidence_kind, evidence_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![rec.attempt_fp, rid, raw, desc, strength.map(i64::from), evidence_kind, evidence_id],
            )?;
            written_relations.insert(rid);
        }
        report.relations = written_relations.len();
        tx.execute(
            "UPDATE attempts SET unmapped_relations = ?2 WHERE attempt_fp = ?1",
            params![rec.attempt_fp, report.unmapped_relations as i64],
        )?;

        // A mention is a new evidence unit stating the item; a re-extraction
        // of the same unit logs nothing.
        if let Some(node) = evidence_node {
            let log = |kind: &str, id: i64| -> Result<(), GraphRagError> {
                tx.execute(
                    "INSERT INTO uses (item_kind, item_id, at, kind, source_id, weight)
                     VALUES (?1, ?2, strftime('%Y-%m-%dT%H:%M:%fZ','now'), 'mention', ?3, 1.0)",
                    params![kind, id, node],
                )?;
                Ok(())
            };
            for id in &touched {
                if !seen_entities.contains(id) && !class_uses.contains(id) {
                    log("entity", *id)?;
                }
            }
            for id in &class_uses {
                if !seen_entities.contains(id) {
                    log("class", *id)?;
                }
            }
            for id in &written_relations {
                if !seen_relations.contains(id) {
                    log("relation", *id)?;
                }
            }
        }

        // GC what lost its last support.
        for rid in &cand_relations {
            let supported: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM attempt_relations WHERE relation_id = ?1)",
                params![rid],
                |r| r.get(0),
            )?;
            if supported {
                continue;
            }
            if let Some((h, t)) = tx
                .query_row(
                    "SELECT head_id, tail_id FROM relations WHERE id = ?1",
                    params![rid],
                    |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)),
                )
                .optional()?
            {
                cand_entities.insert(h);
                cand_entities.insert(t);
            }
            tx.execute(
                "DELETE FROM uses WHERE item_kind = 'relation' AND item_id = ?1",
                params![rid],
            )?;
            report.gc_relations +=
                tx.execute("DELETE FROM relations WHERE id = ?1", params![rid])?;
        }
        for eid in &cand_entities {
            let keep: bool = tx
                .query_row(
                    "SELECT kind != 'entity'
                     OR EXISTS(SELECT 1 FROM attempt_entities WHERE entity_id = ?1)
                     OR EXISTS(SELECT 1 FROM relations WHERE head_id = ?1 OR tail_id = ?1)
                     OR EXISTS(SELECT 1 FROM assertions WHERE head_id = ?1 OR tail_id = ?1)
                     OR EXISTS(SELECT 1 FROM class_aliases WHERE class_id = ?1)
                 FROM entities WHERE id = ?1",
                    params![eid],
                    |r| r.get(0),
                )
                .optional()?
                .unwrap_or(true);
            if keep {
                continue;
            }
            tx.execute(
                "DELETE FROM uses WHERE item_kind IN ('entity','class') AND item_id = ?1",
                params![eid],
            )?;
            report.gc_entities += tx.execute("DELETE FROM entities WHERE id = ?1", params![eid])?;
        }

        for id in &touched {
            let ty = self.canonical_entity_type(*id)?;
            tx.execute(
                "UPDATE entities SET entity_type = ?2 WHERE id = ?1",
                params![id, ty],
            )?;
        }
        self.record_step(
            SubjectKind::Chunk,
            rec.chunk_id,
            Step::Extract,
            rec.attempt_fp,
            StepState::Current,
            None,
        )?;
        tx.commit()?;
        Ok(report)
    }

    /// Distinct evidence units (sections, or documents without sections)
    /// with a current attempt that links the relation.
    pub fn relation_support(&self, relation_id: i64) -> Result<usize, GraphRagError> {
        Ok(self.conn().query_row(
            "SELECT COUNT(DISTINCT evidence_kind || ':' || evidence_id) FROM attempt_relations
             WHERE relation_id = ?1",
            params![relation_id],
            |r| r.get::<_, i64>(0),
        )? as usize)
    }

    /// Majority type over current instances; ties go to the newest.
    pub fn canonical_entity_type(&self, entity_id: i64) -> Result<String, GraphRagError> {
        let ty: Option<String> = self
            .conn()
            .query_row(
                "SELECT ae.entity_type FROM attempt_entities ae JOIN attempts a ON a.attempt_fp = ae.attempt_fp
                 WHERE ae.entity_id = ?1 AND ae.entity_type IS NOT NULL
                 GROUP BY ae.entity_type
                 ORDER BY COUNT(*) DESC, MAX(a.created_at) DESC, MAX(a.rowid) DESC LIMIT 1",
                params![entity_id],
                |r| r.get(0),
            )
            .optional()?;
        match ty {
            Some(t) => Ok(t),
            None => Ok(self
                .conn()
                .query_row(
                    "SELECT entity_type FROM entities WHERE id = ?1",
                    params![entity_id],
                    |r| r.get::<_, Option<String>>(0),
                )?
                .unwrap_or_else(|| "other".into())),
        }
    }
}
