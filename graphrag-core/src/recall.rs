//! Recall helpers shared by the CLI and MCP surfaces: which chunks are
//! eligible (active documents, optionally scoped to groups) and the graph
//! expansion lane that reaches across groups through shared entities.

use crate::db::Database;
use crate::error::GraphRagError;
use crate::lexical::LexicalIndex;
use crate::salience::UseKind;
use crate::vector_source::{BruteForceVectorSource, VectorCandidateSource};
use rusqlite::params;
use std::collections::{BTreeMap, HashMap, HashSet};

use leit_fusion::fuse_default;

/// One fused recall result.
#[derive(Debug, Clone, PartialEq)]
pub struct RecallHit {
    pub chunk_id: i64,
    pub rrf: f64,
    /// Cosine similarity, when the vector lane found it.
    pub vec: Option<f32>,
    /// BM25 score, when the lexical lane found it.
    pub lex: Option<f32>,
    /// Expansion score, when the graph lane found it.
    pub graph: Option<f64>,
    pub created_at: String,
    pub source: Option<String>,
    pub groups: Vec<String>,
    pub heading_path: Vec<String>,
    pub content: String,
}

fn ranked(ids: impl Iterator<Item = i64>) -> Vec<leit_fusion::RankedResult> {
    ids.enumerate()
        .map(|(i, id)| leit_fusion::RankedResult::new(id.to_string(), i + 1))
        .collect()
}

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

    /// Hybrid recall: exact cosine and BM25 over eligible chunks (active
    /// documents, optionally scoped to `groups`), fused with RRF, then a
    /// third lane that follows shared entities from the best half of those
    /// hits. `lexical` is a prebuilt index over the whole store (a cache);
    /// without one, an index over the eligible chunks is built here.
    #[allow(clippy::too_many_arguments)]
    pub fn hybrid_recall(
        &self,
        store: &str,
        query: &str,
        embedding: &[f32],
        groups: &[&str],
        top: usize,
        lexical: Option<&LexicalIndex>,
        now: &str,
    ) -> Result<Vec<RecallHit>, GraphRagError> {
        let mut allowed = self.active_chunk_ids(store)?;
        let scope = if groups.is_empty() {
            None
        } else {
            let s = self.chunk_ids_in_groups(store, groups)?;
            allowed.retain(|c| s.contains(c));
            Some(s)
        };
        let candidate_k = top.max(5) * 4;

        let source = BruteForceVectorSource::for_store(self, store)?;
        let vector_hits: Vec<_> = source
            .top_candidates(embedding, source.len().max(1))?
            .into_iter()
            .filter(|h| allowed.contains(&h.chunk_id))
            .take(candidate_k)
            .collect();

        let lexical_hits: Vec<(i64, f32)> = match lexical {
            Some(idx) => idx
                .search(query, candidate_k * 8)?
                .into_iter()
                .filter(|(id, _)| allowed.contains(id))
                .take(candidate_k)
                .collect(),
            None => {
                let chunks: Vec<_> = self
                    .list_chunks(store)?
                    .into_iter()
                    .filter(|c| allowed.contains(&c.id))
                    .collect();
                match LexicalIndex::build_from_chunks(&chunks)? {
                    Some(idx) => idx.search(query, candidate_k)?,
                    None => Vec::new(),
                }
            }
        };
        if vector_hits.is_empty() && lexical_hits.is_empty() {
            return Ok(Vec::new());
        }
        let first = fuse_default(&[
            ranked(vector_hits.iter().map(|h| h.chunk_id)),
            ranked(lexical_hits.iter().map(|(id, _)| *id)),
        ]);
        let seeds: Vec<i64> = first
            .iter()
            .take(top.div_ceil(2))
            .filter_map(|f| f.id.parse().ok())
            .collect();
        let expansion = self.expand_recall(store, &seeds, scope.as_ref(), top.div_ceil(2), now)?;
        let fused = fuse_default(&[
            ranked(vector_hits.iter().map(|h| h.chunk_id)),
            ranked(lexical_hits.iter().map(|(id, _)| *id)),
            ranked(expansion.chunks.iter().map(|(id, _)| *id)),
        ]);
        let final_ids: Vec<i64> = fused
            .iter()
            .take(top)
            .filter_map(|f| f.id.parse().ok())
            .collect();
        let by_id: HashMap<i64, _> = self
            .get_chunks_by_ids(&final_ids)?
            .into_iter()
            .map(|c| (c.id, c))
            .collect();
        let labels = self.chunk_labels(&final_ids)?;
        let vec_score: HashMap<i64, f32> =
            vector_hits.iter().map(|h| (h.chunk_id, h.score)).collect();
        let lex_score: HashMap<i64, f32> = lexical_hits.iter().copied().collect();
        let graph_score: HashMap<i64, f64> = expansion.chunks.iter().copied().collect();
        let mut hits = Vec::new();
        for f in fused.iter().take(top) {
            let Ok(id) = f.id.parse::<i64>() else {
                continue;
            };
            let Some(chunk) = by_id.get(&id) else {
                continue;
            };
            let (groups, heading_path) = labels.get(&id).cloned().unwrap_or_default();
            hits.push(RecallHit {
                chunk_id: id,
                rrf: f.score,
                vec: vec_score.get(&id).copied(),
                lex: lex_score.get(&id).copied(),
                graph: graph_score.get(&id).copied(),
                created_at: chunk.created_at.clone(),
                source: chunk.source.clone(),
                groups,
                heading_path,
                content: chunk.content.clone(),
            });
        }
        Ok(hits)
    }
}
