use nu_protocol::{LabeledError, Span};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphRagError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("HNSW index error: {0}")]
    Hnsw(String),

    #[error("Store '{0}' not found")]
    StoreNotFound(String),

    #[error("Store '{0}' already exists")]
    StoreExists(String),

    #[error("Entity '{0}' not found")]
    EntityNotFound(String),

    #[error("Invalid embedding dimension: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("{0}")]
    Other(String),
}

impl GraphRagError {
    pub fn into_labeled_error(self, span: Span) -> LabeledError {
        LabeledError::new(self.to_string()).with_label("graphrag error", span)
    }
}

impl From<GraphRagError> for LabeledError {
    fn from(err: GraphRagError) -> Self {
        LabeledError::new(err.to_string())
    }
}
