//! Integration tests for canonical chunk-vector storage and the
//! brute-force vector candidate source.
use graphrag_core::vector_source::{BruteForceVectorSource, VectorCandidateSource};
use graphrag_core::{Database, GraphRagError};
use tempfile::TempDir;

fn db_with_store(dir: &TempDir, dim: usize) -> Database {
    let db = Database::open(&dir.path().join("t.db")).unwrap();
    db.create_store("s", dim).unwrap();
    db
}

fn add_chunk(db: &Database, content: &str, embedding: &[f32]) -> i64 {
    let id = db.add_chunk("s", content, None, None).unwrap();
    db.set_chunk_embedding(id, embedding).unwrap();
    id
}

#[test]
fn chunk_embeddings_roundtrip_in_sqlite() {
    let dir = TempDir::new().unwrap();
    let db = db_with_store(&dir, 3);
    let id = add_chunk(&db, "hello", &[0.1, 0.2, 0.3]);
    let back = db.get_chunk_embedding(id).unwrap().unwrap();
    assert_eq!(back, vec![0.1, 0.2, 0.3]);
    assert!(db.get_chunk_embedding(id + 999).unwrap().is_none());
}

#[test]
fn brute_force_ranks_by_cosine_similarity_best_first() {
    let dir = TempDir::new().unwrap();
    let db = db_with_store(&dir, 2);
    let exact = add_chunk(&db, "exact", &[1.0, 0.0]);
    let near = add_chunk(&db, "near", &[0.9, 0.1]);
    let orthogonal = add_chunk(&db, "orthogonal", &[0.0, 1.0]);
    let source = BruteForceVectorSource::for_store(&db, "s").unwrap();
    let hits = source.top_candidates(&[1.0, 0.0], 3).unwrap();
    let keys: Vec<i64> = hits.iter().map(|h| h.chunk_id).collect();
    assert_eq!(keys, vec![exact, near, orthogonal]);
    assert!(hits[0].score > hits[1].score && hits[1].score > hits[2].score);
    assert!((hits[0].score - 1.0).abs() < 1e-6);
}

#[test]
fn k_truncates_and_zero_k_is_empty() {
    let dir = TempDir::new().unwrap();
    let db = db_with_store(&dir, 2);
    add_chunk(&db, "a", &[1.0, 0.0]);
    add_chunk(&db, "b", &[0.0, 1.0]);
    let source = BruteForceVectorSource::for_store(&db, "s").unwrap();
    assert_eq!(source.top_candidates(&[1.0, 0.0], 1).unwrap().len(), 1);
    assert!(source.top_candidates(&[1.0, 0.0], 0).unwrap().is_empty());
}

#[test]
fn ties_break_by_chunk_id_deterministically() {
    let dir = TempDir::new().unwrap();
    let db = db_with_store(&dir, 2);
    let a = add_chunk(&db, "dup1", &[0.5, 0.5]);
    let b = add_chunk(&db, "dup2", &[0.5, 0.5]);
    let source = BruteForceVectorSource::for_store(&db, "s").unwrap();
    let hits = source.top_candidates(&[0.5, 0.5], 2).unwrap();
    assert_eq!(
        vec![a, b],
        hits.iter().map(|h| h.chunk_id).collect::<Vec<_>>()
    );
    assert!(a < b);
}

#[test]
fn dimension_mismatch_is_a_typed_error() {
    let dir = TempDir::new().unwrap();
    let db = db_with_store(&dir, 3);
    add_chunk(&db, "x", &[0.1, 0.2, 0.3]);
    let source = BruteForceVectorSource::for_store(&db, "s").unwrap();
    let err = source.top_candidates(&[1.0, 0.0], 1).unwrap_err();
    assert!(matches!(
        err,
        GraphRagError::DimensionMismatch {
            expected: 3,
            got: 2
        }
    ));
}

#[test]
fn empty_store_and_chunks_without_embeddings_are_skipped() {
    let dir = TempDir::new().unwrap();
    let db = db_with_store(&dir, 2);
    let source = BruteForceVectorSource::for_store(&db, "s").unwrap();
    assert!(source.top_candidates(&[1.0, 0.0], 5).unwrap().is_empty());
    // chunk without embedding never appears
    db.add_chunk("s", "no-embedding", None, None).unwrap();
    let with = add_chunk(&db, "embedded", &[1.0, 0.0]);
    let source = BruteForceVectorSource::for_store(&db, "s").unwrap();
    let hits = source.top_candidates(&[1.0, 0.0], 5).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].chunk_id, with);
}

#[test]
fn zero_magnitude_vectors_never_rank_first() {
    let dir = TempDir::new().unwrap();
    let db = db_with_store(&dir, 2);
    let zero = add_chunk(&db, "zero", &[0.0, 0.0]);
    let real = add_chunk(&db, "real", &[1.0, 0.0]);
    let source = BruteForceVectorSource::for_store(&db, "s").unwrap();
    let hits = source.top_candidates(&[1.0, 0.0], 2).unwrap();
    assert_eq!(hits[0].chunk_id, real);
    assert_eq!(hits[1].chunk_id, zero);
    assert_eq!(hits[1].score, 0.0);
}
