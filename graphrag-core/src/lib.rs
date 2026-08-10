//! GraphRAG Core Library
//!
//! A knowledge graph system combining exact vector search with entity-relation
//! extraction and community detection using the Leiden algorithm.
//!
//! ## Features
//!
//! - **Vector Search**: Exact brute-force cosine search over canonical SQLite embeddings
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
//! │   Database  │   Vectors   │    Leiden       │
//! │   (SQLite)  │ (bruteforce)│  (communities)  │
//! └─────────────┴─────────────┴─────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use graphrag_core::{BruteForceVectorSource, Database, VectorCandidateSource};
//! use std::path::Path;
//!
//! // Open database
//! let db = Database::open(Path::new("data/graphrag.db"))?;
//!
//! // Create a store
//! let store = db.create_store("documents", 768)?;
//!
//! // Add content with embedding
//! let chunk_id = db.add_chunk("documents", "Hello world", Some("source"), None)?;
//! db.set_chunk_embedding(chunk_id, &embedding)?;
//!
//! // Search (cosine similarity, higher is better)
//! let source = BruteForceVectorSource::for_store(&db, "documents")?;
//! let results = source.top_candidates(&query_embedding, 10)?;
//! ```

pub mod capsule;
pub mod capsule_store;
pub mod db;
pub mod entity_types;
pub mod error;
pub mod export;
pub mod leiden;
pub mod lexical;
pub mod mr;
pub mod synonyms;
pub mod vector_source;

#[cfg(feature = "embeddings")]
pub mod embedder;

#[cfg(feature = "chunking")]
pub mod chunker;

#[cfg(feature = "code")]
pub mod code_chunker;

pub use capsule::{
    CapsuleError, CapsuleKind, CapsuleRef, CapsuleStore, Commitment, Decision, EvidenceItem,
    Freshness, NextSegment, OpenThread, ProjectCapsuleV1, ProjectIdentity, VerifiedFact,
    capsule_uri,
};
pub use db::{Chunk, CommunityRecord, Database, Entity, EntityInput, Relation, Store};
pub use entity_types::{
    CANONICAL_ENTITY_TYPES, STANDARD_TYPE_SYNONYMS, canonical_entity_types,
    load_standard_type_synonyms,
};
pub use error::GraphRagError;
pub use lexical::LexicalIndex;
pub use mr::{
    ExtractPrompt, ExtractResult, ExtractionStrategy, HierarchyConfig, HierarchyReport,
    PhaseStatus, PipelineStatus, WorkUnit, apply_extract, plan_extract,
};
pub use vector_source::{BruteForceVectorSource, VectorCandidate, VectorCandidateSource};

pub use leiden::{
    Community, CommunityGraph, CommunityHierarchy, FlatCommunity, HierarchicalResult,
};
/// Re-export so consumers can fuse lanes without taking leit_fusion as a
/// direct dep (the lexical module is the only intended user of fusion here).
pub use leit_fusion;
pub use synonyms::{STANDARD_SYNONYMS, canonical_relations, load_standard_synonyms};

#[cfg(feature = "embeddings")]
pub use embedder::{
    Embedder, EmbedderConfig, EmbedderModel, RemoteEmbedderConfig, default_embedder_cache_dir,
};

#[cfg(feature = "chunking")]
pub use chunker::{
    ChunkSpan, ChunkerConfig, chunk_markdown, chunk_plain, chunk_spans, chunk_text, nth_chunk,
    review_spans,
};

#[cfg(feature = "code")]
pub use code_chunker::{
    CodeChunkerConfig, CodeLanguage, chunk_code, chunk_code_auto, supported_extensions,
    supported_languages,
};
