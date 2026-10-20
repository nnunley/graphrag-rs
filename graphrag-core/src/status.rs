//! Step status as data. Each subject records, per step, the fingerprint of
//! the inputs that step last saw. "What to do next" is a query: a step is
//! work when no row exists or its stored fingerprint differs from the one
//! computed now from the configuration. There is no queue to drift.

use crate::db::Database;
use crate::error::GraphRagError;
use crate::fingerprints::{
    ChunkerProfile, CommunityConfig, ExtractionConfig, chunk_plan_fp, embed_fp, extract_fp,
    graph_fp,
};
use crate::paper::ExtractionAttemptIdentityParts;
use rusqlite::{OptionalExtension, params};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SubjectKind {
    Document,
    Chunk,
    Graph,
    Community,
}

impl SubjectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SubjectKind::Document => "document",
            SubjectKind::Chunk => "chunk",
            SubjectKind::Graph => "graph",
            SubjectKind::Community => "community",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Step {
    Chunk,
    Embed,
    Extract,
    Generalize,
    Communities,
    Summary,
}

impl Step {
    pub fn as_str(self) -> &'static str {
        match self {
            Step::Chunk => "chunk",
            Step::Embed => "embed",
            Step::Extract => "extract",
            Step::Generalize => "generalize",
            Step::Communities => "communities",
            Step::Summary => "summary",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "chunk" => Step::Chunk,
            "embed" => Step::Embed,
            "extract" => Step::Extract,
            "generalize" => Step::Generalize,
            "communities" => Step::Communities,
            "summary" => Step::Summary,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepState {
    Current,
    Failed,
    Blocked,
}

impl StepState {
    fn as_str(self) -> &'static str {
        match self {
            StepState::Current => "current",
            StepState::Failed => "failed",
            StepState::Blocked => "blocked",
        }
    }
}

/// Stored state for a transient failure awaiting `next_try_at`.
const RETRY: &str = "retry";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkItem {
    pub kind: SubjectKind,
    pub id: i64,
    pub step: Step,
    pub expected_fp: String,
    pub priority: u8,
}

/// Computes the input fingerprint a step *should* have now. `None` means
/// the step doesn't apply to that subject (e.g. extracting a compiled page).
pub trait ExpectedFingerprints {
    fn store(&self) -> &str;
    fn expected(
        &self,
        db: &Database,
        kind: SubjectKind,
        id: i64,
        step: Step,
    ) -> Result<Option<String>, GraphRagError>;
}

/// Fingerprints from fixed configuration: the default implementation.
#[derive(Debug, Clone)]
pub struct StaticConfig {
    pub store: String,
    pub embedder_id: String,
    pub dim: usize,
    pub chunker: ChunkerProfile,
    pub extraction: ExtractionConfig,
    pub community: CommunityConfig,
}

impl ExpectedFingerprints for StaticConfig {
    fn store(&self) -> &str {
        &self.store
    }

    fn expected(
        &self,
        db: &Database,
        kind: SubjectKind,
        id: i64,
        step: Step,
    ) -> Result<Option<String>, GraphRagError> {
        Ok(match (kind, step) {
            (SubjectKind::Document, Step::Chunk) => {
                let content: String = db.conn().query_row(
                    "SELECT content FROM documents WHERE id = ?1",
                    params![id],
                    |r| r.get(0),
                )?;
                Some(chunk_plan_fp(&content, &self.chunker))
            }
            (SubjectKind::Chunk, Step::Embed) => {
                Some(embed_fp(&db.chunk_text(id)?, &self.embedder_id, self.dim))
            }
            (SubjectKind::Chunk, Step::Extract) => {
                let (store, text, origin): (String, String, String) = db.conn().query_row(
                    "SELECT c.store, c.content, d.origin FROM chunk_view c
                     JOIN documents d ON d.id = c.document_id WHERE c.id = ?1",
                    params![id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )?;
                if origin == "compiled" {
                    None
                } else {
                    let e = &self.extraction;
                    Some(
                        extract_fp(&ExtractionAttemptIdentityParts {
                            store: &store,
                            chunk_id: id,
                            chunk_content: &text,
                            model: &e.model,
                            extraction_profile: &e.profile,
                            prompt_version: &e.prompt_version,
                            response_schema_version: &e.schema_version,
                            glean_max_rounds: e.glean_max_rounds,
                        })
                        .map_err(|err| GraphRagError::Other(err.to_string()))?,
                    )
                }
            }
            (SubjectKind::Graph, Step::Communities) => {
                Some(db.graph_input_fp(&self.store, &self.community)?)
            }
            _ => None,
        })
    }
}

/// Per-step counts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct StepCounts {
    pub current: BTreeMap<Step, usize>,
    pub stale: BTreeMap<Step, usize>,
    pub missing: BTreeMap<Step, usize>,
    pub failed: BTreeMap<Step, usize>,
}

impl StepCounts {
    fn add(&mut self, step: Step, class: Class) {
        let m = match class {
            Class::Current => &mut self.current,
            Class::Stale => &mut self.stale,
            Class::Missing => &mut self.missing,
            Class::Failed => &mut self.failed,
        };
        *m.entry(step).or_default() += 1;
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct StatusReport {
    pub per_group: BTreeMap<String, StepCounts>,
    pub total: StepCounts,
    pub backlog_by_priority: [usize; 3],
    pub failed_samples: Vec<(i64, Step, String)>,
    pub unmapped_relations: usize,
    pub rejected_entities: usize,
    pub type_conflicts: usize,
    pub pending_summaries: usize,
    pub classes: usize,
    pub new_classes_24h: usize,
    pub lifted_admitted: usize,
    pub dormant_entities: usize,
    pub dormant_relations: usize,
}

impl StatusReport {
    /// Zero missing and zero stale steps. Failed items don't count against it.
    pub fn is_complete(&self) -> bool {
        self.total.missing.values().all(|n| *n == 0) && self.total.stale.values().all(|n| *n == 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Current,
    Stale,
    Missing,
    Failed,
}

#[derive(Debug, Clone)]
struct Row {
    fp: Option<String>,
    state: String,
    next_try_at: Option<String>,
    last_error: Option<String>,
}

fn classify(row: Option<&Row>, expected: &str) -> Class {
    match row {
        None => Class::Missing,
        Some(r) => match r.fp.as_deref() {
            None => Class::Missing,
            Some(fp) if fp != expected => Class::Stale,
            Some(_) if r.state == "current" => Class::Current,
            Some(_) if r.state == "failed" => Class::Failed,
            Some(_) => Class::Missing,
        },
    }
}

/// Whether a step with this row is ready to run now.
fn ready(row: Option<&Row>, expected: &str, now: &str) -> bool {
    if let Some(r) = row
        && r.next_try_at.as_deref().is_some_and(|t| t > now)
    {
        return false;
    }
    matches!(classify(row, expected), Class::Missing | Class::Stale)
}

/// The chunk steps, in the order the worker runs them.
const CHUNK_STEPS: [Step; 2] = [Step::Embed, Step::Extract];

impl Database {
    pub fn record_step(
        &self,
        kind: SubjectKind,
        id: i64,
        step: Step,
        fp: &str,
        state: StepState,
        error: Option<&str>,
    ) -> Result<(), GraphRagError> {
        self.conn().execute(
            "INSERT INTO step_status (subject_kind, subject_id, step, fp, state, attempts, last_error, next_try_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, CASE WHEN ?5 = 'failed' THEN 1 ELSE 0 END, ?6, NULL,
                     strftime('%Y-%m-%dT%H:%M:%fZ','now'))
             ON CONFLICT(subject_kind, subject_id, step) DO UPDATE SET
                 fp = excluded.fp, state = excluded.state, last_error = excluded.last_error,
                 attempts = CASE WHEN excluded.state = 'failed' THEN step_status.attempts + 1 ELSE 0 END,
                 next_try_at = NULL, updated_at = excluded.updated_at",
            params![kind.as_str(), id, step.as_str(), fp, state.as_str(), error],
        )?;
        Ok(())
    }

    /// A retryable failure: no attempt is recorded, only the error and the
    /// earliest time to try again. Returns the consecutive attempt count.
    pub fn record_transient_failure(
        &self,
        kind: SubjectKind,
        id: i64,
        step: Step,
        error: &str,
        next_try_at: &str,
    ) -> Result<u32, GraphRagError> {
        self.conn().execute(
            "INSERT INTO step_status (subject_kind, subject_id, step, fp, state, attempts, last_error, next_try_at, updated_at)
             VALUES (?1, ?2, ?3, NULL, ?6, 1, ?4, ?5, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
             ON CONFLICT(subject_kind, subject_id, step) DO UPDATE SET
                 state = ?6, attempts = step_status.attempts + 1, last_error = ?4,
                 next_try_at = ?5, updated_at = excluded.updated_at",
            params![kind.as_str(), id, step.as_str(), error, next_try_at, RETRY],
        )?;
        Ok(self.conn().query_row(
            "SELECT attempts FROM step_status WHERE subject_kind = ?1 AND subject_id = ?2 AND step = ?3",
            params![kind.as_str(), id, step.as_str()],
            |r| r.get(0),
        )?)
    }

    /// Move recall hits with an incomplete chunk step to the front of the
    /// work (priority 2). One UPDATE; never waits on extraction.
    pub fn escalate_chunks(&self, chunk_ids: &[i64]) -> Result<usize, GraphRagError> {
        if chunk_ids.is_empty() {
            return Ok(0);
        }
        let ids = chunk_ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let n = self.conn().execute(
            &format!(
                "UPDATE chunks SET priority = 2
                 WHERE id IN ({ids}) AND priority < 2
                   AND (NOT EXISTS (SELECT 1 FROM step_status s WHERE s.subject_kind = 'chunk'
                            AND s.subject_id = chunks.id AND s.step = 'embed' AND s.state = 'current')
                        OR ((SELECT origin FROM documents d WHERE d.id = chunks.document_id) = 'source'
                            AND NOT EXISTS (SELECT 1 FROM step_status s WHERE s.subject_kind = 'chunk'
                                AND s.subject_id = chunks.id AND s.step = 'extract' AND s.state = 'current')))"
            ),
            [],
        )?;
        Ok(n)
    }

    fn status_rows(&self, kind: SubjectKind) -> Result<HashMap<(i64, Step), Row>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT subject_id, step, fp, state, next_try_at, last_error FROM step_status WHERE subject_kind = ?1",
        )?;
        let rows = stmt.query_map(params![kind.as_str()], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                Row {
                    fp: r.get(2)?,
                    state: r.get(3)?,
                    next_try_at: r.get(4)?,
                    last_error: r.get(5)?,
                },
            ))
        })?;
        let mut out = HashMap::new();
        for row in rows {
            let (id, step, row) = row?;
            if let Some(step) = Step::parse(&step) {
                out.insert((id, step), row);
            }
        }
        Ok(out)
    }

    /// `(chunk id, priority, document id)` for chunks of active documents,
    /// highest priority first, then oldest.
    fn active_chunks(&self, store: &str) -> Result<Vec<(i64, u8, i64)>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT c.id, c.priority, c.document_id FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE c.store = ?1
               AND (d.node_id IS NULL OR NOT EXISTS (SELECT 1 FROM assertions a
                    WHERE a.tail_id = d.node_id AND a.relation = 'replaces' AND a.retracted_at IS NULL))
             ORDER BY c.priority DESC, c.id ASC",
        )?;
        let rows = stmt.query_map(params![store], |r| {
            Ok((r.get(0)?, r.get::<_, i64>(1)?.clamp(0, 2) as u8, r.get(2)?))
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    fn store_documents(&self, store: &str) -> Result<Vec<i64>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT DISTINCT d.id FROM documents d LEFT JOIN entities n ON n.id = d.node_id
             WHERE n.store = ?1 OR EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id AND c.store = ?1)
             ORDER BY d.id",
        )?;
        let rows = stmt.query_map(params![store], |r| r.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn next_work(
        &self,
        cfg: &dyn ExpectedFingerprints,
        now: &str,
        limit: usize,
    ) -> Result<Vec<WorkItem>, GraphRagError> {
        let mut out = Vec::new();
        let store = cfg.store().to_string();

        let doc_rows = self.status_rows(SubjectKind::Document)?;
        for d in self.store_documents(&store)? {
            if out.len() >= limit {
                return Ok(out);
            }
            if let Some(fp) = cfg.expected(self, SubjectKind::Document, d, Step::Chunk)?
                && ready(doc_rows.get(&(d, Step::Chunk)), &fp, now)
            {
                out.push(WorkItem {
                    kind: SubjectKind::Document,
                    id: d,
                    step: Step::Chunk,
                    expected_fp: fp,
                    priority: 1,
                });
            }
        }

        let rows = self.status_rows(SubjectKind::Chunk)?;
        for (c, priority, _) in self.active_chunks(&store)? {
            for step in CHUNK_STEPS {
                if out.len() >= limit {
                    return Ok(out);
                }
                if let Some(fp) = cfg.expected(self, SubjectKind::Chunk, c, step)?
                    && ready(rows.get(&(c, step)), &fp, now)
                {
                    out.push(WorkItem {
                        kind: SubjectKind::Chunk,
                        id: c,
                        step,
                        expected_fp: fp,
                        priority,
                    });
                }
            }
        }
        if !out.is_empty() {
            return Ok(out);
        }

        // Whole-graph steps run only when no chunk-level work is ready.
        let graph_rows = self.status_rows(SubjectKind::Graph)?;
        for step in [Step::Generalize, Step::Communities] {
            if let Some(fp) = cfg.expected(self, SubjectKind::Graph, 0, step)?
                && ready(graph_rows.get(&(0, step)), &fp, now)
            {
                out.push(WorkItem {
                    kind: SubjectKind::Graph,
                    id: 0,
                    step,
                    expected_fp: fp,
                    priority: 0,
                });
                // Communities depend on generalization; one at a time.
                return Ok(out);
            }
        }

        let community_rows = self.status_rows(SubjectKind::Community)?;
        for id in self.current_run_community_ids(&store)? {
            if out.len() >= limit {
                break;
            }
            if let Some(fp) = cfg.expected(self, SubjectKind::Community, id, Step::Summary)?
                && ready(community_rows.get(&(id, Step::Summary)), &fp, now)
            {
                out.push(WorkItem {
                    kind: SubjectKind::Community,
                    id,
                    step: Step::Summary,
                    expected_fp: fp,
                    priority: 0,
                });
            }
        }
        Ok(out)
    }

    fn current_run_community_ids(&self, store: &str) -> Result<Vec<i64>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT c.id FROM communities c JOIN community_runs r ON r.run_id = c.run_id
             WHERE r.store = ?1 AND r.current = 1 ORDER BY c.level DESC, c.id",
        )?;
        let rows = stmt.query_map(params![store], |r| r.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Input fingerprint for clustering: the current graph as an edge list.
    /// Task 12's weighted projection replaces the unit weights used here.
    pub fn graph_input_fp(
        &self,
        store: &str,
        cfg: &CommunityConfig,
    ) -> Result<String, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT head_id, tail_id FROM relations WHERE store = ?1 AND edge_status = 'active'",
        )?;
        let edges: Vec<(i64, i64, f64)> = stmt
            .query_map(params![store], |r| Ok((r.get(0)?, r.get(1)?, 1.0)))?
            .collect::<Result<_, _>>()?;
        let mut stmt = self.conn().prepare(
            "SELECT id FROM entities e WHERE store = ?1 AND kind = 'entity' AND edge_status = 'active'
               AND NOT EXISTS (SELECT 1 FROM relations r WHERE r.head_id = e.id OR r.tail_id = e.id)",
        )?;
        let isolates: Vec<i64> = stmt
            .query_map(params![store], |r| r.get(0))?
            .collect::<Result<_, _>>()?;
        Ok(graph_fp(&edges, &isolates, cfg, cfg.seed))
    }

    pub fn status_report(
        &self,
        cfg: &dyn ExpectedFingerprints,
    ) -> Result<StatusReport, GraphRagError> {
        let store = cfg.store().to_string();
        let mut report = StatusReport::default();
        let mut doc_groups: HashMap<i64, Vec<String>> = HashMap::new();
        let mut groups_for = |db: &Database, doc: i64| -> Result<Vec<String>, GraphRagError> {
            if let Some(g) = doc_groups.get(&doc) {
                return Ok(g.clone());
            }
            let node: Option<i64> = db
                .conn()
                .query_row(
                    "SELECT node_id FROM documents WHERE id = ?1",
                    params![doc],
                    |r| r.get(0),
                )
                .optional()?
                .flatten();
            let mut g = match node {
                Some(n) => db.groups_of(n)?,
                None => Vec::new(),
            };
            if g.is_empty() {
                g.push("(ungrouped)".into());
            }
            doc_groups.insert(doc, g.clone());
            Ok(g)
        };

        let doc_rows = self.status_rows(SubjectKind::Document)?;
        for d in self.store_documents(&store)? {
            if let Some(fp) = cfg.expected(self, SubjectKind::Document, d, Step::Chunk)? {
                let class = classify(doc_rows.get(&(d, Step::Chunk)), &fp);
                report.total.add(Step::Chunk, class);
                for g in groups_for(self, d)? {
                    report
                        .per_group
                        .entry(g)
                        .or_default()
                        .add(Step::Chunk, class);
                }
            }
        }

        let rows = self.status_rows(SubjectKind::Chunk)?;
        for (c, priority, doc) in self.active_chunks(&store)? {
            let groups = groups_for(self, doc)?;
            let mut incomplete = false;
            for step in CHUNK_STEPS {
                let Some(fp) = cfg.expected(self, SubjectKind::Chunk, c, step)? else {
                    continue;
                };
                let row = rows.get(&(c, step));
                let class = classify(row, &fp);
                if matches!(class, Class::Missing | Class::Stale) {
                    incomplete = true;
                }
                if class == Class::Failed && report.failed_samples.len() < 10 {
                    let err = row.and_then(|r| r.last_error.clone()).unwrap_or_default();
                    report.failed_samples.push((c, step, err));
                }
                report.total.add(step, class);
                for g in &groups {
                    report
                        .per_group
                        .entry(g.clone())
                        .or_default()
                        .add(step, class);
                }
            }
            if incomplete {
                report.backlog_by_priority[priority as usize] += 1;
            }
        }

        let graph_rows = self.status_rows(SubjectKind::Graph)?;
        for step in [Step::Generalize, Step::Communities] {
            if let Some(fp) = cfg.expected(self, SubjectKind::Graph, 0, step)? {
                report
                    .total
                    .add(step, classify(graph_rows.get(&(0, step)), &fp));
            }
        }

        let count = |sql: &str| -> Result<usize, GraphRagError> {
            Ok(self
                .conn()
                .query_row(sql, params![store], |r| r.get::<_, i64>(0))?
                .max(0) as usize)
        };
        report.unmapped_relations = count(
            "SELECT COALESCE(SUM(a.unmapped_relations), 0) FROM attempts a JOIN chunks c ON c.id = a.chunk_id WHERE c.store = ?1",
        )?;
        report.rejected_entities = count(
            "SELECT COALESCE(SUM(a.rejected_entities), 0) FROM attempts a JOIN chunks c ON c.id = a.chunk_id WHERE c.store = ?1",
        )?;
        report.type_conflicts = count(
            "SELECT COUNT(*) FROM (SELECT ae.entity_id FROM attempt_entities ae
                 JOIN entities e ON e.id = ae.entity_id WHERE e.store = ?1
                 GROUP BY ae.entity_id HAVING COUNT(DISTINCT ae.entity_type) > 1)",
        )?;
        report.classes = count(
            "SELECT COUNT(*) FROM entities WHERE store = ?1 AND kind = 'entity' AND entity_type = 'collection'",
        )?;
        report.new_classes_24h = count(
            "SELECT COUNT(*) FROM (SELECT u.item_id FROM uses u JOIN entities e ON e.id = u.item_id
                 WHERE u.item_kind = 'class' AND e.store = ?1 GROUP BY u.item_id
                 HAVING MIN(u.at) > strftime('%Y-%m-%dT%H:%M:%fZ','now','-1 day'))",
        )?;
        report.lifted_admitted = count(
            "SELECT COUNT(*) FROM lifted_edges l JOIN entities e ON e.id = l.head_class WHERE e.store = ?1 AND l.admitted = 1",
        )?;
        report.dormant_entities =
            count("SELECT COUNT(*) FROM entities WHERE store = ?1 AND edge_status = 'dormant'")?;
        report.dormant_relations =
            count("SELECT COUNT(*) FROM relations WHERE store = ?1 AND edge_status = 'dormant'")?;
        report.pending_summaries = {
            let community_rows = self.status_rows(SubjectKind::Community)?;
            let mut n = 0;
            for id in self.current_run_community_ids(&store)? {
                if let Some(fp) = cfg.expected(self, SubjectKind::Community, id, Step::Summary)? {
                    let class = classify(community_rows.get(&(id, Step::Summary)), &fp);
                    report.total.add(Step::Summary, class);
                    if matches!(class, Class::Missing | Class::Stale) {
                        n += 1;
                    }
                }
            }
            n
        };
        Ok(report)
    }
}
