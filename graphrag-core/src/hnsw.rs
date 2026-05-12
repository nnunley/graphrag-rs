use crate::error::GraphRagError;
use std::path::Path;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// Wrapper around usearch Index for HNSW vector search
pub struct HnswIndex {
    index: Index,
    dim: usize,
}

/// Search result with key and distance
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub key: u64,
    pub distance: f32,
}

impl HnswIndex {
    /// Create a new HNSW index with specified dimensions
    pub fn new(dim: usize) -> Result<Self, GraphRagError> {
        let options = IndexOptions {
            dimensions: dim,
            metric: MetricKind::Cos, // Cosine similarity for embeddings
            quantization: ScalarKind::F32,
            connectivity: 16,     // M parameter - connections per node
            expansion_add: 128,   // ef_construction
            expansion_search: 64, // ef_search
            ..IndexOptions::default()
        };

        let index = Index::new(&options)
            .map_err(|e| GraphRagError::Hnsw(format!("Failed to create index: {}", e)))?;

        Ok(Self { index, dim })
    }

    /// Load an existing index from disk
    pub fn load(path: &Path, dim: usize) -> Result<Self, GraphRagError> {
        let hnsw = Self::new(dim)?;

        if path.exists() {
            hnsw.index
                .load(path.to_str().unwrap_or_default())
                .map_err(|e| GraphRagError::Hnsw(format!("Failed to load index: {}", e)))?;
        }

        Ok(hnsw)
    }

    /// Save the index to disk
    pub fn save(&self, path: &Path) -> Result<(), GraphRagError> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        self.index
            .save(path.to_str().unwrap_or_default())
            .map_err(|e| GraphRagError::Hnsw(format!("Failed to save index: {}", e)))?;

        Ok(())
    }

    /// Add a vector with the given key (chunk_id)
    pub fn add(&self, key: u64, vector: &[f32]) -> Result<(), GraphRagError> {
        if vector.len() != self.dim {
            return Err(GraphRagError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        // Ensure capacity before adding
        let current_size = self.index.size();
        if self.index.capacity() <= current_size {
            self.index
                .reserve(current_size + 1000) // Reserve space for 1000 more vectors
                .map_err(|e| GraphRagError::Hnsw(format!("Failed to reserve capacity: {}", e)))?;
        }

        self.index
            .add(key, vector)
            .map_err(|e| GraphRagError::Hnsw(format!("Failed to add vector: {}", e)))?;

        Ok(())
    }

    /// Search for the k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, GraphRagError> {
        if query.len() != self.dim {
            return Err(GraphRagError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let results = self
            .index
            .search(query, k)
            .map_err(|e| GraphRagError::Hnsw(format!("Failed to search: {}", e)))?;

        Ok(results
            .keys
            .iter()
            .zip(results.distances.iter())
            .map(|(&key, &distance)| SearchResult { key, distance })
            .collect())
    }

    /// Reserve capacity for n vectors
    #[allow(dead_code)]
    pub fn reserve(&self, capacity: usize) -> Result<(), GraphRagError> {
        self.index
            .reserve(capacity)
            .map_err(|e| GraphRagError::Hnsw(format!("Failed to reserve capacity: {}", e)))?;
        Ok(())
    }

    /// Get the number of vectors in the index
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// Check if the index is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the dimension of vectors in this index
    #[allow(dead_code)]
    pub fn dim(&self) -> usize {
        self.dim
    }
}
