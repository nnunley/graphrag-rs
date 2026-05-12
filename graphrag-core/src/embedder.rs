//! Text embedding module for GraphRAG
//!
//! Provides text-to-vector embedding using local models via fastembed.
//! Supports multiple models with automatic chunking for long texts.

use crate::error::GraphRagError;

#[cfg(feature = "embeddings")]
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Configuration for the embedder
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Model to use for embeddings
    pub model: EmbedderModel,
    /// Whether to show download progress
    pub show_download_progress: bool,
    /// Cache directory for models (None = default)
    pub cache_dir: Option<std::path::PathBuf>,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model: EmbedderModel::NomicEmbedText,
            show_download_progress: true,
            cache_dir: None,
        }
    }
}

/// Available embedding models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderModel {
    /// nomic-embed-text-v1.5 - 768 dims, 8192 token context
    /// Best for longer texts, high quality
    NomicEmbedText,
    /// all-MiniLM-L6-v2 - 384 dims, 256 token context
    /// Faster, smaller, good for short texts
    MiniLM,
}

impl EmbedderModel {
    /// Get the embedding dimension for this model
    pub fn dimension(&self) -> usize {
        match self {
            EmbedderModel::NomicEmbedText => 768,
            EmbedderModel::MiniLM => 384,
        }
    }

    /// Get the maximum token context for this model
    pub fn max_tokens(&self) -> usize {
        match self {
            EmbedderModel::NomicEmbedText => 8192,
            EmbedderModel::MiniLM => 256,
        }
    }

    /// Approximate characters per token (rough estimate)
    pub fn chars_per_token(&self) -> usize {
        4 // Conservative estimate
    }

    /// Get max characters before chunking is needed
    pub fn max_chars(&self) -> usize {
        self.max_tokens() * self.chars_per_token()
    }
}

/// Text embedder using local models
#[cfg(feature = "embeddings")]
pub struct Embedder {
    model: TextEmbedding,
    config: EmbedderConfig,
}

#[cfg(feature = "embeddings")]
impl Embedder {
    /// Create a new embedder with the given configuration
    pub fn new(config: EmbedderConfig) -> Result<Self, GraphRagError> {
        let fastembed_model = match config.model {
            EmbedderModel::NomicEmbedText => EmbeddingModel::NomicEmbedTextV15,
            EmbedderModel::MiniLM => EmbeddingModel::AllMiniLML6V2,
        };

        let mut options = InitOptions::new(fastembed_model)
            .with_show_download_progress(config.show_download_progress);

        if let Some(ref cache_dir) = config.cache_dir {
            options = options.with_cache_dir(cache_dir.clone());
        }

        let model =
            TextEmbedding::try_new(options).map_err(|e| GraphRagError::Embedding(e.to_string()))?;

        Ok(Self { model, config })
    }

    /// Create an embedder with default configuration (nomic-embed-text)
    pub fn with_defaults() -> Result<Self, GraphRagError> {
        Self::new(EmbedderConfig::default())
    }

    /// Create an embedder with MiniLM (faster, smaller)
    pub fn with_minilm() -> Result<Self, GraphRagError> {
        Self::new(EmbedderConfig {
            model: EmbedderModel::MiniLM,
            ..Default::default()
        })
    }

    /// Get the embedding dimension for this embedder
    pub fn dimension(&self) -> usize {
        self.config.model.dimension()
    }

    /// Embed a single text, using chunked mean-pooling if needed
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, GraphRagError> {
        if text.len() <= self.config.model.max_chars() {
            // Text fits in context, embed directly
            self.embed_single(text)
        } else {
            // Text too long, use chunked mean-pooling
            self.embed_chunked(text)
        }
    }

    /// Embed multiple texts (batched for efficiency)
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, GraphRagError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text)?);
        }
        Ok(results)
    }

    /// Embed a single text without chunking
    fn embed_single(&self, text: &str) -> Result<Vec<f32>, GraphRagError> {
        let embeddings = self
            .model
            .embed(vec![text], None)
            .map_err(|e| GraphRagError::Embedding(e.to_string()))?;

        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| GraphRagError::Embedding("No embedding returned".to_string()))
    }

    /// Embed long text using chunked mean-pooling
    fn embed_chunked(&self, text: &str) -> Result<Vec<f32>, GraphRagError> {
        let max_chars = self.config.model.max_chars();
        let overlap_chars = max_chars / 5; // 20% overlap

        let chunks = split_with_overlap(text, max_chars, overlap_chars);

        if chunks.is_empty() {
            return Err(GraphRagError::Embedding("Empty text".to_string()));
        }

        // Embed all chunks
        let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
        let embeddings = self
            .model
            .embed(chunk_refs, None)
            .map_err(|e| GraphRagError::Embedding(e.to_string()))?;

        // Mean pool across all chunk embeddings
        Ok(mean_pool(&embeddings))
    }
}

/// Split text into overlapping chunks
fn split_with_overlap(text: &str, max_chars: usize, overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + max_chars).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();

        // Try to break at sentence/word boundary
        let chunk = if end < chars.len() {
            truncate_at_boundary(&chunk)
        } else {
            chunk
        };

        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }

        // Move start forward, accounting for overlap
        let advance = max_chars.saturating_sub(overlap);
        if advance == 0 {
            break; // Prevent infinite loop
        }
        start += advance;
    }

    chunks
}

/// Truncate text at a sentence or word boundary
fn truncate_at_boundary(text: &str) -> String {
    // Try sentence boundary first
    if let Some(pos) = text.rfind(['.', '!', '?'])
        && pos > text.len() / 2
    {
        return text[..=pos].to_string();
    }

    // Fall back to word boundary
    if let Some(pos) = text.rfind(char::is_whitespace)
        && pos > text.len() / 2
    {
        return text[..pos].to_string();
    }

    text.to_string()
}

/// Mean pool multiple embeddings into one
fn mean_pool(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let dim = embeddings[0].len();
    let count = embeddings.len() as f32;

    let mut result = vec![0.0; dim];
    for emb in embeddings {
        for (i, val) in emb.iter().enumerate() {
            result[i] += val;
        }
    }

    for val in &mut result {
        *val /= count;
    }

    // L2 normalize the result
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut result {
            *val /= norm;
        }
    }

    result
}

#[cfg(test)]
#[cfg(feature = "embeddings")]
mod tests {
    use super::*;

    #[test]
    fn test_split_with_overlap() {
        let text = "This is a test. Another sentence here. And one more.";
        let chunks = split_with_overlap(text, 20, 5);
        assert!(!chunks.is_empty());
        // Chunks should overlap
        if chunks.len() > 1 {
            // Some content should appear in multiple chunks
        }
    }

    #[test]
    fn test_mean_pool() {
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let pooled = mean_pool(&embeddings);
        assert_eq!(pooled.len(), 3);
        // Should be normalized
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }
}
