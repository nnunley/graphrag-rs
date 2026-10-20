//! The embedding interface the worker needs: an identity (part of the
//! embed-step fingerprint), a dimension, and `embed`. Ungated, so code and
//! tests that use a fake don't pull in a model runtime.

use crate::error::GraphRagError;

pub trait TextEmbedder {
    /// Stable identity, e.g. `fastembed:nomic-embed-text-v1.5`.
    fn id(&self) -> String;
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Result<Vec<f32>, GraphRagError>;
}
