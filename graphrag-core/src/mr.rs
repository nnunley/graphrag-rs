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
/// A community is pending when its `summary` is NULL. Writing any summary —
/// including an empty one — is terminal, so a community the model cannot
/// describe is not replanned forever.
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
        let entities = db.get_community_entities(community_id)?;
        let names: Vec<String> = entities
            .iter()
            .map(|e| match &e.entity_type {
                Some(t) => format!("{} ({})", e.name, t),
                None => e.name.clone(),
            })
            .collect();
        units.push(WorkUnit {
            unit_id: format!("summarize:community:{community_id}"),
            kind: "summarize".into(),
            chunk_id: 0,
            community_id: Some(community_id),
            model: model.to_string(),
            system: SUMMARY_SYSTEM.into(),
            user: format!(
                "Community {community_id} (level {level}) contains these entities:\n{}\n\n\
                 Write a concise summary of what this community is about.",
                names.join("\n")
            ),
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
    /// Communities in `store` with no summary yet, ascending by id.
    pub fn pending_summary_communities(
        &self,
        store: &str,
        limit: Option<usize>,
    ) -> Result<Vec<(i64, i32)>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT id, level FROM communities
             WHERE store = ?1 AND summary IS NULL
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
