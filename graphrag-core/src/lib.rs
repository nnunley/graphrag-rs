//! GraphRAG Core Library
//!
//! A knowledge graph system combining HNSW vector search with entity-relation
//! extraction and community detection using the Leiden algorithm.
//!
//! ## Features
//!
//! - **Vector Search**: Fast approximate nearest neighbor search using HNSW (usearch)
//! - **Knowledge Graph**: Entity-relation storage with typed entities
//! - **Community Detection**: Leiden algorithm for discovering entity clusters
//! - **Graph Expansion**: Follow relations to discover related content
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │              graphrag-core                   │
//! ├─────────────┬─────────────┬─────────────────┤
//! │   Database  │    HNSW     │    Leiden       │
//! │   (SQLite)  │  (usearch)  │  (communities)  │
//! └─────────────┴─────────────┴─────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use graphrag_core::{Database, HnswIndex};
//! use std::path::Path;
//!
//! // Open database and index
//! let db = Database::open(Path::new("data/graphrag.db"))?;
//! let index = HnswIndex::load(Path::new("data/store.usearch"), 768)?;
//!
//! // Create a store
//! let store = db.create_store("documents", 768)?;
//!
//! // Add content with embedding
//! let chunk_id = db.add_chunk("documents", "Hello world", Some("source"), None)?;
//! index.add(chunk_id as u64, &embedding)?;
//!
//! // Search
//! let results = index.search(&query_embedding, 10)?;
//! ```

pub mod db;
pub mod entity_types;
pub mod error;
pub mod hnsw;
pub mod leiden;
pub mod synonyms;

#[cfg(feature = "embeddings")]
pub mod embedder;

#[cfg(feature = "chunking")]
pub mod chunker;

#[cfg(feature = "code")]
pub mod code_chunker;

pub use db::{Chunk, CommunityRecord, Database, Entity, EntityInput, Relation, Store};
pub use entity_types::{
    CANONICAL_ENTITY_TYPES, STANDARD_TYPE_SYNONYMS, canonical_entity_types,
    load_standard_type_synonyms,
};
pub use error::GraphRagError;
pub use hnsw::{HnswIndex, SearchResult};
pub use leiden::{
    Community, CommunityGraph, CommunityHierarchy, FlatCommunity, HierarchicalResult,
};
pub use synonyms::{STANDARD_SYNONYMS, canonical_relations, load_standard_synonyms};

#[cfg(feature = "embeddings")]
pub use embedder::{Embedder, EmbedderConfig, EmbedderModel};

#[cfg(feature = "chunking")]
pub use chunker::{ChunkerConfig, chunk_markdown, chunk_plain, chunk_text};

#[cfg(feature = "code")]
pub use code_chunker::{
    CodeChunkerConfig, CodeLanguage, chunk_code, chunk_code_auto, supported_extensions,
    supported_languages,
};
