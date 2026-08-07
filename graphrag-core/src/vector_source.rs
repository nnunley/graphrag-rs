//! Canonical chunk-vector candidate source.
//!
//! Vectors are canonical in SQLite (`chunk_embeddings`); any index built over
//! them is a disposable cache. This module provides the fusion-shaped
//! candidate-source contract plus an exact brute-force provider, which is the
//! correct default at small scale (thousands of vectors): perfect recall, no
//! sidecar index file, no approximate-search parameters.
//!
//! The trait is deliberately shaped as a *ranked candidate source* (best
//! first, higher score better) so it composes with `leit_fusion`-style rank
//! fusion today and can migrate behind leit's planned `leit_vector`
//! candidate-source seam without redesign.

use crate::db::Database;
use crate::error::GraphRagError;
use rusqlite::params;

/// One ranked vector hit. `score` is cosine similarity in `[-1, 1]`,
/// higher is better. Zero-magnitude vectors score `0.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorCandidate {
    pub chunk_id: i64,
    pub score: f32,
}

/// A ranked vector candidate source: best first, deterministic order
/// (ties break by ascending chunk id).
pub trait VectorCandidateSource {
    type Error;

    /// Top `k` candidates for `query`, best first.
    fn top_candidates(&self, query: &[f32], k: usize) -> Result<Vec<VectorCandidate>, Self::Error>;
}

/// Exact scan over all embedded chunks of one store, loaded once at
/// construction. Rebuild the source to observe later writes.
pub struct BruteForceVectorSource {
    dim: usize,
    vectors: Vec<(i64, Vec<f32>)>,
}

impl BruteForceVectorSource {
    /// Load every chunk embedding for `store`. Chunks without embeddings are
    /// skipped. Fails if a stored blob does not match the store dimension.
    pub fn for_store(db: &Database, store: &str) -> Result<Self, GraphRagError> {
        let dim = db.get_store(store)?.dim;
        let vectors = db.all_chunk_embeddings(store)?;
        for (id, v) in &vectors {
            if v.len() != dim {
                return Err(GraphRagError::Capsule(format!(
                    "chunk {id} embedding has dimension {}, store {store:?} expects {dim}",
                    v.len()
                )));
            }
        }
        Ok(Self { dim, vectors })
    }

    /// Number of embedded chunks visible to this source.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// True when no embedded chunks are visible.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

impl VectorCandidateSource for BruteForceVectorSource {
    type Error = GraphRagError;

    fn top_candidates(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<VectorCandidate>, GraphRagError> {
        if query.len() != self.dim {
            return Err(GraphRagError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut hits: Vec<VectorCandidate> = self
            .vectors
            .iter()
            .map(|(id, v)| VectorCandidate {
                chunk_id: *id,
                score: cosine_similarity(query, v),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.chunk_id.cmp(&b.chunk_id))
        });
        hits.truncate(k);
        Ok(hits)
    }
}

impl Database {
    /// Store the canonical embedding for a chunk (little-endian f32 blob).
    pub fn set_chunk_embedding(
        &self,
        chunk_id: i64,
        embedding: &[f32],
    ) -> Result<(), GraphRagError> {
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.conn().execute(
            "INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding) VALUES (?1, ?2)",
            params![chunk_id, blob],
        )?;
        Ok(())
    }

    /// The canonical embedding for a chunk, if stored.
    pub fn get_chunk_embedding(&self, chunk_id: i64) -> Result<Option<Vec<f32>>, GraphRagError> {
        use rusqlite::OptionalExtension;
        let blob: Option<Vec<u8>> = self
            .conn()
            .query_row(
                "SELECT embedding FROM chunk_embeddings WHERE chunk_id = ?1",
                params![chunk_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(blob.map(|b| bytes_to_f32(&b)))
    }

    /// All `(chunk_id, embedding)` pairs for a store, ascending by chunk id.
    pub fn all_chunk_embeddings(&self, store: &str) -> Result<Vec<(i64, Vec<f32>)>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT ce.chunk_id, ce.embedding
             FROM chunk_embeddings ce JOIN chunks c ON c.id = ce.chunk_id
             WHERE c.store = ?1 ORDER BY ce.chunk_id",
        )?;
        let rows = stmt.query_map(params![store], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, blob) = row?;
            out.push((id, bytes_to_f32(&blob)));
        }
        Ok(out)
    }
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
