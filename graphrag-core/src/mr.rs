//! Map-reduce work units: plan / execute / apply.
//!
//! GraphRAG's expensive stages (chunk -> triples, community -> summary,
//! context -> partial answer) are independent, idempotent units. This module
//! owns *what needs doing*, *how to persist it*, and *what is already done* —
//! deliberately NOT concurrency, retry, or model choice. Those belong to
//! whatever drives the units: an agent runtime today, a worker fleet later.
//!
//! ```text
//! plan_extract()  -> [WorkUnit]       what still needs a model call
//!    (executor)   -> [ExtractResult]  caller's concern: parallelism, retry
//! apply_extract() -> usize            parse, persist, checkpoint
//! ```
//!
//! The prompting/parsing strategy is INJECTED. Keeping it out of core is what
//! lets this crate stay free of any HTTP/LLM dependency, and lets the contract
//! be tested end-to-end without a model.
//!
//! The `extractions` table is the checkpoint: it records that a chunk was
//! attempted, with which model, and how many triples resulted — so a chunk
//! that legitimately yields zero triples is terminal rather than replanned
//! forever.

use crate::db::{Database, EntityInput};
use crate::error::GraphRagError;
use rusqlite::params;
use serde::{Deserialize, Serialize};

/// Prompt for one unit, as produced by an injected strategy.
pub struct ExtractPrompt {
    pub system: String,
    pub user: String,
    pub format: Option<serde_json::Value>,
}

/// How to prompt a model for extraction and how to read its reply.
///
/// Implemented outside this crate (see graphrag-llm) so core carries no
/// model or transport dependency.
pub trait ExtractionStrategy {
    fn prompt(&self, chunk: &str, chunk_index: usize, total: usize) -> ExtractPrompt;
    /// Parse a model reply into triples. Malformed input yields an empty vec
    /// rather than an error: a bad reply is a zero-yield outcome, not a fault.
    fn parse(&self, response: &str) -> Vec<EntityInput>;
}

/// One unit of model work. Serialized to JSONL for an external executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkUnit {
    pub unit_id: String,
    pub kind: String,
    pub chunk_id: i64,
    /// Set for community-scoped stages (summarize); None for chunk stages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub community_id: Option<i64>,
    pub model: String,
    pub system: String,
    pub user: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
}

/// A completed unit, as returned by the executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractResult {
    pub chunk_id: i64,
    pub model: String,
    pub response: String,
}

/// Pending extraction units for `store`, ascending by chunk id.
///
/// A chunk is pending when it has no `extractions` row. Ordering is stable, so
/// repeated planning is deterministic and a partial run simply resumes.
pub fn plan_extract(
    db: &Database,
    store: &str,
    model: &str,
    strategy: &dyn ExtractionStrategy,
    limit: Option<usize>,
) -> Result<Vec<WorkUnit>, GraphRagError> {
    db.get_store(store)?; // unknown store -> typed error
    let chunks = db.pending_extraction_chunks(store, limit)?;
    let total = chunks.len();
    Ok(chunks
        .into_iter()
        .enumerate()
        .map(|(idx, (chunk_id, content))| {
            let p = strategy.prompt(&content, idx, total);
            WorkUnit {
                unit_id: format!("extract:chunk:{chunk_id}"),
                kind: "extract".into(),
                chunk_id,
                community_id: None,
                model: model.to_string(),
                system: p.system,
                user: p.user,
                format: p.format,
            }
        })
        .collect())
}

/// Parse and persist executor results; checkpoint every chunk touched.
///
/// Returns the number of triples persisted. Results may arrive partially and
/// out of order — each unit is independent, so a crashed executor run leaves
/// the remainder pending. Malformed replies persist nothing but still
/// checkpoint: the attempt happened, and replanning it would loop forever.
pub fn apply_extract(
    db: &Database,
    store: &str,
    strategy: &dyn ExtractionStrategy,
    results: &[ExtractResult],
) -> Result<usize, GraphRagError> {
    db.get_store(store)?;
    let mut persisted = 0usize;
    for r in results {
        let triples = strategy.parse(&r.response);
        for t in &triples {
            db.persist_triple(store, r.chunk_id, t)?;
        }
        persisted += triples.len();
        db.checkpoint_extraction(r.chunk_id, &r.model, triples.len())?;
    }
    Ok(persisted)
}

impl Database {
    /// Chunks in `store` with no extraction checkpoint, ascending by id.
    pub fn pending_extraction_chunks(
        &self,
        store: &str,
        limit: Option<usize>,
    ) -> Result<Vec<(i64, String)>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT c.id, c.content FROM chunks c
             WHERE c.store = ?1
               AND c.id NOT IN (SELECT chunk_id FROM extractions)
             ORDER BY c.id
             LIMIT ?2",
        )?;
        let cap = limit.map(|n| n as i64).unwrap_or(-1); // -1 = unlimited in SQLite
        let rows = stmt.query_map(params![store, cap], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Persist one triple and its chunk provenance. Entities are get-or-create,
    /// so re-applying the same result does not duplicate them.
    pub fn persist_triple(
        &self,
        store: &str,
        chunk_id: i64,
        t: &EntityInput,
    ) -> Result<(), GraphRagError> {
        let head_id = self.get_or_create_entity(store, &t.head, t.head_type.as_deref(), None)?;
        let tail_id = self.get_or_create_entity(store, &t.tail, t.tail_type.as_deref(), None)?;
        self.add_relation(store, head_id, tail_id, &t.relation, None)?;
        self.link_chunk_entity(chunk_id, head_id)?;
        self.link_chunk_entity(chunk_id, tail_id)?;
        Ok(())
    }

    /// Record that `chunk_id` was attempted. Idempotent by chunk.
    pub fn checkpoint_extraction(
        &self,
        chunk_id: i64,
        model: &str,
        triple_count: usize,
    ) -> Result<(), GraphRagError> {
        self.conn().execute(
            "INSERT INTO extractions (chunk_id, model, triple_count, extracted_at)
             VALUES (?1, ?2, ?3, datetime('now'))
             ON CONFLICT(chunk_id) DO UPDATE SET
               model = excluded.model,
               triple_count = excluded.triple_count,
               extracted_at = excluded.extracted_at",
            params![chunk_id, model, triple_count as i64],
        )?;
        Ok(())
    }
}

// --- stage 2: community -> summary ----------------------------------------
//
// Unlike extraction, summarization has no parse step: the model's reply IS the
// summary. So no strategy is injected — core builds the prompt from community
// context and any model can execute it.

/// A completed summary unit, as returned by the executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryResult {
    pub community_id: i64,
    pub model: String,
    pub response: String,
}

/// Pending community-summary units for `store`, ascending by community id.
///
/// A community is pending when its `summary` is NULL **and** every child it
/// has is already summarized. That makes summarization proceed bottom-up over
/// the Leiden hierarchy, which the GraphRAG design requires: a higher-level
/// community is described by substituting its SUB-COMMUNITY summaries for the
/// element summaries, so the children must exist first.
///
/// Leaf communities prompt from their entities. Parents prompt from their
/// child summaries — a root may span thousands of entities that would never
/// fit a context window, whereas a handful of child summaries always will.
///
/// Writing any summary — including an empty one — is terminal, so a community
/// the model cannot describe is not replanned forever.
pub fn plan_summarize(
    db: &Database,
    store: &str,
    model: &str,
    limit: Option<usize>,
) -> Result<Vec<WorkUnit>, GraphRagError> {
    db.get_store(store)?;
    let pending = db.pending_summary_communities(store, limit)?;
    let mut units = Vec::with_capacity(pending.len());
    for (community_id, level) in pending {
        let children = db.child_community_summaries(community_id)?;
        let user = if children.is_empty() {
            // Leaf: describe the elements directly.
            let names: Vec<String> = db
                .get_community_entities(community_id)?
                .iter()
                .map(|e| match &e.entity_type {
                    Some(t) => format!("{} ({})", e.name, t),
                    None => e.name.clone(),
                })
                .collect();
            format!(
                "Community {community_id} (level {level}) contains these entities:\n{}\n\n\
                 Write a concise summary of what this community is about.",
                names.join("\n")
            )
        } else {
            // Higher level: roll up the sub-community summaries instead.
            format!(
                "Community {community_id} (level {level}) groups {} sub-communities, \
                 summarized as:\n{}\n\n\
                 Write a single higher-level summary of the theme these share.",
                children.len(),
                children
                    .iter()
                    .map(|(id, sum)| format!("- [{id}] {sum}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        units.push(WorkUnit {
            unit_id: format!("summarize:community:{community_id}"),
            kind: "summarize".into(),
            chunk_id: 0,
            community_id: Some(community_id),
            model: model.to_string(),
            system: SUMMARY_SYSTEM.into(),
            user,
            format: None,
        });
    }
    Ok(units)
}

const SUMMARY_SYSTEM: &str = "You summarize clusters of related entities from a knowledge graph. \
Reply with a short prose summary only — no preamble, no bullet lists.";

/// Persist community summaries. Partial and out-of-order results are fine.
pub fn apply_summarize(
    db: &Database,
    store: &str,
    results: &[SummaryResult],
) -> Result<usize, GraphRagError> {
    db.get_store(store)?;
    for r in results {
        // Persist even a blank reply: NULL means "not attempted", and an
        // un-summarizable community must not requeue forever.
        db.update_community_summary(r.community_id, r.response.trim())?;
    }
    Ok(results.len())
}

impl Database {
    /// Summarized children of `community_id`, ascending by id.
    pub fn child_community_summaries(
        &self,
        community_id: i64,
    ) -> Result<Vec<(i64, String)>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT id, summary FROM communities
             WHERE parent_id = ?1 AND summary IS NOT NULL
             ORDER BY id",
        )?;
        let rows = stmt.query_map(params![community_id], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Communities in `store` that are ready to summarize: no summary yet, and
    /// no unsummarized children. Ascending by id; bottom-up by construction.
    pub fn pending_summary_communities(
        &self,
        store: &str,
        limit: Option<usize>,
    ) -> Result<Vec<(i64, i32)>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT id, level FROM communities c
             WHERE c.store = ?1 AND c.summary IS NULL
               AND NOT EXISTS (
                   SELECT 1 FROM communities k
                   WHERE k.parent_id = c.id AND k.summary IS NULL
               )
             ORDER BY id
             LIMIT ?2",
        )?;
        let cap = limit.map(|n| n as i64).unwrap_or(-1);
        let rows = stmt.query_map(params![store, cap], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, i32>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }
}

// --- pipeline status --------------------------------------------------------
//
// `plan` answers "what units does stage X have?". An orchestrator also needs
// "which phases exist, which are ready, and what changed underneath me?" —
// especially because extraction keeps producing entities, which silently
// invalidates a previous community detection.

/// One phase of the indexing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseStatus {
    pub phase: String,
    /// What a unit of this phase is: chunk | entity | community.
    pub unit: String,
    pub total: usize,
    pub done: usize,
    /// Units that could be worked right now.
    pub pending: usize,
    /// Units that exist but are waiting on a dependency (unsummarized children).
    pub blocked: usize,
    pub ready: bool,
    pub blocked_by: Option<String>,
    /// Concrete next action, naming the command to run.
    pub guidance: String,
    /// For `summarize`: (level, communities, summarized) ascending by level.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub levels: Vec<(i32, usize, usize)>,
}

/// Whole-pipeline view for one store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatus {
    pub store: String,
    pub phases: Vec<PhaseStatus>,
    /// The single most useful next action across all phases.
    pub next: String,
}

impl PipelineStatus {
    pub fn phase(&self, name: &str) -> Option<&PhaseStatus> {
        self.phases.iter().find(|p| p.phase == name)
    }
}

/// Report every phase: counts, readiness, and what to run next.
pub fn pipeline_status(db: &Database, store: &str) -> Result<PipelineStatus, GraphRagError> {
    db.get_store(store)?;

    let chunks = db.list_chunks(store)?.len();
    let extracted = chunks - db.pending_extraction_chunks(store, None)?.len();
    let entities = db.list_entities(store)?.len();
    let unclustered = db.unclustered_entity_count(store)?;
    let comms = db.list_communities(store)?;
    let summarized = comms.iter().filter(|c| c.summary.is_some()).count();
    let ready_now = db.pending_summary_communities(store, None)?.len();
    let unsummarized = comms.len() - summarized;

    let extract = PhaseStatus {
        phase: "extract".into(),
        unit: "chunk".into(),
        total: chunks,
        done: extracted,
        pending: chunks - extracted,
        blocked: 0,
        ready: chunks > extracted,
        blocked_by: None,
        guidance: if chunks == 0 {
            "no chunks yet: ingest documents with `graphrag note`".into()
        } else if chunks > extracted {
            format!(
                "{} chunk(s) need triples: `graphrag plan extract`",
                chunks - extracted
            )
        } else {
            "all chunks extracted".into()
        },
        levels: Vec::new(),
    };

    let unembedded = db.entities_without_embeddings(store, None)?.len();
    let embed = PhaseStatus {
        phase: "embed".into(),
        unit: "entity".into(),
        total: entities,
        done: entities - unembedded,
        pending: unembedded,
        blocked: 0,
        ready: unembedded > 0,
        blocked_by: (entities == 0).then(|| "extract".to_string()),
        guidance: if entities == 0 {
            "no entities yet: run extraction first".into()
        } else if unembedded > 0 {
            format!(
                "{unembedded} entit{} need vectors: `graphrag embed --store {store}`",
                if unembedded == 1 { "y" } else { "ies" }
            )
        } else {
            format!("all {entities} entities embedded")
        },
        levels: Vec::new(),
    };

    // Entities appearing after a detection run are the staleness signal: they
    // exist but belong to no community.
    let communities = PhaseStatus {
        phase: "communities".into(),
        unit: "entity".into(),
        total: entities,
        done: entities - unclustered,
        pending: unclustered,
        blocked: 0,
        ready: entities > 0 && unclustered > 0,
        blocked_by: (entities == 0).then(|| "extract".to_string()),
        guidance: if entities == 0 {
            "no entities yet: run extraction first".into()
        } else if unclustered > 0 {
            format!(
                "{unclustered} entit{} appeared since the last detection: `graphrag enrich --store {store}`",
                if unclustered == 1 { "y" } else { "ies" }
            )
        } else {
            format!("all {entities} entities are clustered")
        },
        levels: Vec::new(),
    };

    let mut levels: Vec<(i32, usize, usize)> = Vec::new();
    for c in &comms {
        match levels.iter_mut().find(|(l, _, _)| *l == c.level) {
            Some(e) => {
                e.1 += 1;
                if c.summary.is_some() {
                    e.2 += 1;
                }
            }
            None => levels.push((c.level, 1, usize::from(c.summary.is_some()))),
        }
    }
    levels.sort_by_key(|(l, _, _)| *l);

    let summarize = PhaseStatus {
        phase: "summarize".into(),
        unit: "community".into(),
        total: comms.len(),
        done: summarized,
        pending: ready_now,
        blocked: unsummarized - ready_now,
        ready: ready_now > 0,
        blocked_by: (comms.is_empty()).then(|| "communities".to_string()),
        guidance: if comms.is_empty() {
            "no communities yet: run `graphrag enrich` first".into()
        } else if ready_now > 0 {
            format!(
                "{ready_now} community summary/ies ready (bottom-up): `graphrag plan summarize`"
            )
        } else if unsummarized > 0 {
            format!("{unsummarized} parent(s) waiting on child summaries")
        } else {
            format!(
                "all {} communities summarized across {} level(s)",
                comms.len(),
                levels.len()
            )
        },
        levels,
    };

    let next = [&extract, &embed, &communities, &summarize]
        .iter()
        .find(|p| p.ready)
        .map(|p| p.guidance.clone())
        .unwrap_or_else(|| "pipeline complete: nothing pending".to_string());

    Ok(PipelineStatus {
        store: store.to_string(),
        phases: vec![extract, embed, communities, summarize],
        next,
    })
}

impl Database {
    /// Entities in `store` belonging to no community — i.e. that appeared
    /// after the last detection run.
    pub fn unclustered_entity_count(&self, store: &str) -> Result<usize, GraphRagError> {
        let n: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM entities e
             WHERE e.store = ?1
               AND e.id NOT IN (SELECT entity_id FROM entity_communities)",
            params![store],
            |r| r.get(0),
        )?;
        Ok(n as usize)
    }
}

// --- upward aggregation: build a hierarchy that converges to one root ------
//
// `leiden_hierarchical` subdivides DOWNWARD: it splits large communities into
// finer ones, leaving as many parentless roots as the base partition found.
// That cannot answer a global question — there is no vantage point that sees
// the whole graph.
//
// This builds UPWARD instead, the direction the GraphRAG design needs. Cluster
// the entity graph, then treat each community as a node and cluster THOSE,
// repeating until a single root remains. Every community therefore has a
// parent, every entity belongs to a leaf, and following parents from anywhere
// reaches the root that names what lies beneath it.

/// Tuning for [`build_community_hierarchy`].
#[derive(Debug, Clone)]
pub struct HierarchyConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    /// Stop aggregating past this many levels, even without a single root.
    pub max_levels: usize,
}

impl Default for HierarchyConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            max_levels: 10,
        }
    }
}

/// Summary of a hierarchy build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyReport {
    pub levels: usize,
    /// Communities per level, coarsest (root) first.
    pub per_level: Vec<usize>,
    pub roots: usize,
    pub entities_clustered: usize,
}

/// Cluster `store` into a rooted community hierarchy, replacing any existing
/// communities. Level 0 is the root; deeper levels are finer, matching the
/// GraphRAG convention where C0 is the coarsest summary.
pub fn build_community_hierarchy(
    db: &Database,
    store: &str,
    cfg: HierarchyConfig,
) -> Result<HierarchyReport, GraphRagError> {
    db.get_store(store)?;
    db.clear_communities(store)?;

    let entities = db.list_entities(store)?;
    if entities.is_empty() {
        return Ok(HierarchyReport {
            levels: 0,
            per_level: vec![],
            roots: 0,
            entities_clustered: 0,
        });
    }
    let relations = db.list_relations(store)?;

    // --- level N (finest): partition the entity graph itself.
    // Isolated entities are included by roster, so nothing is orphaned even
    // when the extractor produced no relation for it.
    let graph = crate::leiden::CommunityGraph::from_edges_and_nodes(
        relations.iter().map(|r| (r.head_id, r.tail_id, 1.0)),
        entities.iter().map(|e| e.id),
    );
    let base = graph.leiden(Some(cfg.max_iterations), cfg.tolerance);

    // groups[level] = for each community at that level, the entity ids beneath it
    let mut groups: Vec<Vec<Vec<i64>>> =
        vec![base.communities.iter().map(|c| c.collect_nodes()).collect()];

    // Any entity the partition missed becomes its own group: nothing is orphaned.
    let mut seen: std::collections::HashSet<i64> = groups[0].iter().flatten().copied().collect();
    for e in &entities {
        if seen.insert(e.id) {
            groups[0].push(vec![e.id]);
        }
    }

    // --- aggregate upward until one community remains.
    while groups.last().map(|g| g.len()).unwrap_or(0) > 1 && groups.len() < cfg.max_levels {
        let cur = groups.last().unwrap();
        // Which group does each entity belong to at this level?
        let mut owner: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for (gi, g) in cur.iter().enumerate() {
            for &e in g {
                owner.insert(e, gi);
            }
        }
        // Edges between groups become weighted edges of the community graph.
        let mut weights: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();
        for r in &relations {
            let (Some(&a), Some(&b)) = (owner.get(&r.head_id), owner.get(&r.tail_id)) else {
                continue;
            };
            if a != b {
                *weights.entry((a.min(b), a.max(b))).or_insert(0.0) += 1.0;
            }
        }
        if weights.is_empty() {
            break; // disconnected: no meaningful coarser partition exists
        }
        let cg = crate::leiden::CommunityGraph::from_edges_and_nodes(
            weights.iter().map(|(&(a, b), &w)| (a as i64, b as i64, w)),
            (0..cur.len()).map(|i| i as i64),
        );
        let merged = cg.leiden(Some(cfg.max_iterations), cfg.tolerance);
        let mut next: Vec<Vec<i64>> = Vec::new();
        let mut grouped: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for c in &merged.communities {
            let mut members = Vec::new();
            for gi in c.collect_nodes() {
                grouped.insert(gi as usize);
                members.extend_from_slice(&cur[gi as usize]);
            }
            if !members.is_empty() {
                next.push(members);
            }
        }
        // Groups the coarser pass did not place carry forward unchanged.
        for (gi, g) in cur.iter().enumerate() {
            if !grouped.contains(&gi) {
                next.push(g.clone());
            }
        }
        if next.len() >= cur.len() {
            break; // no consolidation achieved; stop rather than loop forever
        }
        groups.push(next);
    }

    // If aggregation stalled with several groups, cap them with a synthetic
    // root so the invariant "everything reaches one root" always holds.
    if groups.last().map(|g| g.len()).unwrap_or(0) > 1 {
        let all: Vec<i64> = entities.iter().map(|e| e.id).collect();
        groups.push(vec![all]);
    }

    // --- persist coarsest-first: level 0 is the root.
    groups.reverse();
    let mut parent_ids: Vec<i64> = Vec::new();
    let mut per_level = Vec::new();
    for (level, level_groups) in groups.iter().enumerate() {
        let mut ids = Vec::with_capacity(level_groups.len());
        for members in level_groups {
            // Parent = the community one level up that contains these members.
            let parent = if level == 0 {
                None
            } else {
                members.first().and_then(|first| {
                    groups[level - 1]
                        .iter()
                        .position(|g| g.contains(first))
                        .and_then(|pos| parent_ids.get(pos).copied())
                })
            };
            let cid = db.create_community(store, level as i32, base.modularity, parent)?;
            for &e in members {
                db.link_entity_community(e, cid)?;
            }
            ids.push(cid);
        }
        per_level.push(ids.len());
        parent_ids = ids;
    }

    Ok(HierarchyReport {
        levels: groups.len(),
        per_level,
        roots: 1,
        entities_clustered: entities.len(),
    })
}

// --- phase: entity embedding ------------------------------------------------
//
// Entities carry their own vectors on the RAG side of GraphRAG. This is what
// lets lexical variants that remain distinct nodes co-surface: measured on the
// default embedder, "Leiden" / "Leiden algorithm" / "hierarchical Leiden" sit
// at cosine 0.74-1.00 to each other and 0.24-0.31 to unrelated terms. It is
// also the input the merge-candidate tooling needs to propose consolidation.
//
// Unlike extraction this is local compute, so the executor is usually the same
// process — but it keeps the plan/apply shape so a caller may batch it, run it
// on another machine, or checkpoint a long backfill.

/// An entity awaiting a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedUnit {
    pub entity_id: i64,
    /// The text to embed: the entity name, qualified by type when known.
    pub text: String,
}

/// A computed entity vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResult {
    pub entity_id: i64,
    pub vector: Vec<f32>,
}

/// Entities in `store` with no stored vector, ascending by id.
pub fn plan_embed(
    db: &Database,
    store: &str,
    limit: Option<usize>,
) -> Result<Vec<EmbedUnit>, GraphRagError> {
    db.get_store(store)?;
    Ok(db
        .entities_without_embeddings(store, limit)?
        .into_iter()
        .map(|(entity_id, name, ty)| EmbedUnit {
            text: match ty {
                Some(t) if !t.is_empty() => format!("{name} ({t})"),
                _ => name,
            },
            entity_id,
        })
        .collect())
}

/// Persist computed entity vectors. Returns how many were stored.
pub fn apply_embed(
    db: &Database,
    store: &str,
    results: &[EmbedResult],
) -> Result<usize, GraphRagError> {
    db.get_store(store)?;
    for r in results {
        db.set_entity_embedding(r.entity_id, &r.vector)?;
    }
    Ok(results.len())
}

impl Database {
    /// Entities lacking a vector: `(id, name, entity_type)`, ascending by id.
    pub fn entities_without_embeddings(
        &self,
        store: &str,
        limit: Option<usize>,
    ) -> Result<Vec<(i64, String, Option<String>)>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT id, name, entity_type FROM entities
             WHERE store = ?1 AND id NOT IN (SELECT entity_id FROM entity_embeddings)
             ORDER BY id
             LIMIT ?2",
        )?;
        let cap = limit.map(|n| n as i64).unwrap_or(-1);
        let rows = stmt.query_map(params![store, cap], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }
}
