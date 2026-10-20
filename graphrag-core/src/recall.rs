//! Recall helpers shared by the CLI and MCP surfaces: which chunks are
//! eligible (active documents, optionally scoped to groups) and the graph
//! expansion lane that reaches across groups through shared entities.

use crate::db::Database;
use crate::error::GraphRagError;
use crate::salience::UseKind;
use rusqlite::params;
use std::collections::{BTreeMap, HashMap, HashSet};

/// `(groups, heading path)` of a chunk.
pub type ChunkLabel = (Vec<String>, Vec<String>);

/// The expansion lane's result.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Expansion {
    /// `(chunk id, score)`, best first.
    pub chunks: Vec<(i64, f64)>,
    /// Active entities reached from the seed chunks.
    pub entities: Vec<i64>,
}

impl Database {
    /// Chunks of documents `part_of` any of `groups` (through subgroups).
    pub fn chunk_ids_in_groups(
        &self,
        store: &str,
        groups: &[&str],
    ) -> Result<HashSet<i64>, GraphRagError> {
        let mut out = HashSet::new();
        for g in groups {
            for node in self.members_of_group(store, g)? {
                let mut stmt = self.conn().prepare(
                    "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.node_id = ?1",
                )?;
                for id in stmt.query_map(params![node], |r| r.get::<_, i64>(0))? {
                    out.insert(id?);
                }
            }
        }
        Ok(out)
    }

    /// Chunks whose document is not retired by an active `replaces`.
    pub fn active_chunk_ids(&self, store: &str) -> Result<HashSet<i64>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE c.store = ?1
               AND (d.node_id IS NULL OR NOT EXISTS (SELECT 1 FROM assertions a
                    WHERE a.tail_id = d.node_id AND a.relation = 'replaces' AND a.retracted_at IS NULL))",
        )?;
        Ok(stmt
            .query_map(params![store], |r| r.get(0))?
            .collect::<Result<_, _>>()?)
    }

    /// Chunks that share active entities with the seed chunks, scored by
    /// Σ 1/ln(2 + df) over shared entities (df = chunks mentioning the
    /// entity), so a rare shared entity outweighs a hub. Logs a recall use
    /// for each entity reached.
    pub fn expand_recall(
        &self,
        store: &str,
        seeds: &[i64],
        allowed: Option<&HashSet<i64>>,
        limit: usize,
        now: &str,
    ) -> Result<Expansion, GraphRagError> {
        let seed_set: HashSet<i64> = seeds.iter().copied().collect();
        let mut entities: Vec<i64> = Vec::new();
        for s in seeds {
            let mut stmt = self.conn().prepare(
                "SELECT e.id FROM chunk_entities ce JOIN entities e ON e.id = ce.entity_id
                 WHERE ce.chunk_id = ?1 AND e.store = ?2 AND e.kind = 'entity' AND e.edge_status = 'active'",
            )?;
            for id in stmt.query_map(params![s, store], |r| r.get::<_, i64>(0))? {
                let id = id?;
                if !entities.contains(&id) {
                    entities.push(id);
                }
            }
        }
        let active = self.active_chunk_ids(store)?;
        let mut score: HashMap<i64, f64> = HashMap::new();
        for e in &entities {
            let mut stmt = self
                .conn()
                .prepare("SELECT chunk_id FROM chunk_entities WHERE entity_id = ?1")?;
            let chunks: Vec<i64> = stmt
                .query_map(params![e], |r| r.get(0))?
                .collect::<Result<_, _>>()?;
            let weight = 1.0 / (2.0 + chunks.len() as f64).ln();
            for c in chunks {
                if seed_set.contains(&c)
                    || !active.contains(&c)
                    || allowed.is_some_and(|a| !a.contains(&c))
                {
                    continue;
                }
                *score.entry(c).or_default() += weight;
            }
        }
        for e in &entities {
            self.log_use("entity", *e, UseKind::Recall, None, now)?;
        }
        let mut ranked: Vec<(i64, f64)> = score.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(limit);
        Ok(Expansion {
            chunks: ranked,
            entities,
        })
    }

    /// Groups and heading path for each chunk, for recall output.
    pub fn chunk_labels(
        &self,
        chunk_ids: &[i64],
    ) -> Result<BTreeMap<i64, ChunkLabel>, GraphRagError> {
        let mut out = BTreeMap::new();
        for id in chunk_ids {
            let ctx = self.chunk_context(*id)?;
            let node: Option<i64> = self.conn().query_row(
                "SELECT node_id FROM documents WHERE id = ?1",
                params![ctx.document_id],
                |r| r.get(0),
            )?;
            let groups = match node {
                Some(n) => self.groups_of(n)?,
                None => Vec::new(),
            };
            out.insert(*id, (groups, ctx.heading_path));
        }
        Ok(out)
    }
}
