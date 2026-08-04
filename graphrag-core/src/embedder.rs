//! Text embedding module for GraphRAG
//!
//! Provides text-to-vector embedding using local models via fastembed
//! or remote APIs (OpenAI, etc.).
//! Supports multiple models with automatic chunking for long texts.

use crate::error::GraphRagError;
use std::path::PathBuf;

#[cfg(feature = "embeddings")]
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

const EMBED_CACHE_DIR_ENV: &str = "GRAPHRAG_EMBED_CACHE_DIR";

/// Configuration for the embedder
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Model to use for embeddings
    pub model: EmbedderModel,
    /// Whether to show download progress (local models only)
    pub show_download_progress: bool,
    /// Cache directory for models (None = default, local models only)
    pub cache_dir: Option<std::path::PathBuf>,
    /// Remote API configuration (used for remote models)
    pub remote: Option<RemoteEmbedderConfig>,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model: EmbedderModel::NomicEmbedText,
            show_download_progress: true,
            cache_dir: None,
            remote: None,
        }
    }
}

/// Resolve the default cache directory for local embedding models.
///
/// This keeps model downloads out of the current working directory when the
/// embedder is started from long-lived services like MCP servers.
pub fn default_embedder_cache_dir() -> PathBuf {
    resolve_embedder_cache_dir(
        std::env::var(EMBED_CACHE_DIR_ENV).ok(),
        dirs::cache_dir(),
        dirs::home_dir(),
    )
}

fn resolve_embedder_cache_dir(
    env_override: Option<String>,
    cache_dir: Option<PathBuf>,
    home_dir: Option<PathBuf>,
) -> PathBuf {
    if let Some(path) = env_override
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
    {
        return path;
    }

    cache_dir
        .or_else(|| home_dir.map(|home| home.join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("graphrag")
        .join("fastembed")
}

/// Configuration for remote embedding APIs
#[derive(Debug, Clone)]
pub struct RemoteEmbedderConfig {
    /// API key for the remote service
    pub api_key: String,
    /// Base URL override (None = use default)
    pub base_url: Option<String>,
    /// Model identifier (e.g., "text-embedding-ada-002")
    pub model: Option<String>,
}

impl RemoteEmbedderConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: None,
            model: None,
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = Some(url.to_string());
        self
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
    /// OpenAI text-embedding-ada-002 - 1536 dims
    /// Requires API key
    OpenAIAda002,
    /// OpenAI text-embedding-3-small - 1536 dims
    /// Newer, better quality model
    OpenAI3Small,
}

impl EmbedderModel {
    /// Get the embedding dimension for this model
    pub fn dimension(&self) -> usize {
        match self {
            EmbedderModel::NomicEmbedText => 768,
            EmbedderModel::MiniLM => 384,
            EmbedderModel::OpenAIAda002 => 1536,
            EmbedderModel::OpenAI3Small => 1536,
        }
    }

    /// Get the maximum token context for this model
    pub fn max_tokens(&self) -> usize {
        match self {
            EmbedderModel::NomicEmbedText => 8192,
            EmbedderModel::MiniLM => 256,
            EmbedderModel::OpenAIAda002 => 8192,
            EmbedderModel::OpenAI3Small => 8192,
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

    /// Check if this is a remote model (requires API)
    pub fn is_remote(&self) -> bool {
        matches!(
            self,
            EmbedderModel::OpenAIAda002 | EmbedderModel::OpenAI3Small
        )
    }

    /// Get the default model name for remote APIs
    pub fn default_remote_model(&self) -> &'static str {
        match self {
            EmbedderModel::OpenAIAda002 => "text-embedding-ada-002",
            EmbedderModel::OpenAI3Small => "text-embedding-3-small",
            _ => "",
        }
    }
}

/// Text embedder using local models
#[cfg(feature = "embeddings")]
pub struct Embedder {
    #[cfg(feature = "embeddings")]
    local: Option<TextEmbedding>,
    #[cfg(feature = "embeddings_remote")]
    remote: Option<RemoteEmbedder>,
    config: EmbedderConfig,
}

#[cfg(feature = "embeddings_remote")]
enum RemoteEmbedder {
    OpenAI(OpenAIEmbedder),
}

#[cfg(feature = "embeddings_remote")]
pub struct OpenAIEmbedder {
    client: reqwest::blocking::Client,
    api_key: String,
    base_url: String,
    model: String,
}

#[cfg(feature = "embeddings_remote")]
impl OpenAIEmbedder {
    pub fn new(config: &RemoteEmbedderConfig, model: EmbedderModel) -> Result<Self, GraphRagError> {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let model = config
            .model
            .clone()
            .unwrap_or_else(|| model.default_remote_model().to_string());

        let client = reqwest::blocking::Client::new();

        Ok(Self {
            client,
            api_key: config.api_key.clone(),
            base_url,
            model,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, GraphRagError> {
        let url = format!("{}/embeddings", self.base_url);

        #[derive(serde::Serialize)]
        struct EmbedRequest<'a> {
            input: &'a str,
            model: &'a str,
        }

        #[derive(serde::Deserialize)]
        struct EmbedResponse {
            data: Vec<EmbedData>,
        }

        #[derive(serde::Deserialize)]
        struct EmbedData {
            embedding: Vec<f32>,
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&EmbedRequest {
                input: text,
                model: &self.model,
            })
            .send()
            .map_err(|e| GraphRagError::Embedding(format!("API request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(GraphRagError::Embedding(format!(
                "API error ({}): {}",
                status, body
            )));
        }

        let result: EmbedResponse = response
            .json()
            .map_err(|e| GraphRagError::Embedding(format!("Failed to parse response: {}", e)))?;

        result
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| GraphRagError::Embedding("No embedding in response".to_string()))
    }
}

#[cfg(feature = "embeddings")]
impl Embedder {
    /// Create a new embedder with the given configuration
    pub fn new(config: EmbedderConfig) -> Result<Self, GraphRagError> {
        #[cfg(feature = "embeddings")]
        let local = if config.model.is_remote() {
            None
        } else {
            let fastembed_model = match config.model {
                EmbedderModel::NomicEmbedText => EmbeddingModel::NomicEmbedTextV15,
                EmbedderModel::MiniLM => EmbeddingModel::AllMiniLML6V2,
                _ => return Err(GraphRagError::Embedding("Unknown local model".to_string())),
            };

            let mut options = InitOptions::new(fastembed_model)
                .with_show_download_progress(config.show_download_progress);

            if let Some(ref cache_dir) = config.cache_dir {
                options = options.with_cache_dir(cache_dir.clone());
            }

            let model = TextEmbedding::try_new(options)
                .map_err(|e| GraphRagError::Embedding(e.to_string()))?;
            Some(model)
        };

        #[cfg(feature = "embeddings_remote")]
        let remote = if config.model.is_remote() {
            let remote_config = config.remote.as_ref().ok_or_else(|| {
                GraphRagError::Embedding(
                    "Remote embedder requires API key configuration".to_string(),
                )
            })?;
            Some(RemoteEmbedder::OpenAI(OpenAIEmbedder::new(
                remote_config,
                config.model,
            )?))
        } else {
            None
        };

        #[cfg(not(feature = "embeddings"))]
        let local: Option<TextEmbedding> = None;

        Ok(Self {
            local,
            #[cfg(feature = "embeddings_remote")]
            remote,
            config,
        })
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
        #[cfg(feature = "embeddings")]
        if let Some(ref model) = self.local {
            let embeddings = model
                .embed(vec![text], None)
                .map_err(|e| GraphRagError::Embedding(e.to_string()))?;

            return embeddings
                .into_iter()
                .next()
                .ok_or_else(|| GraphRagError::Embedding("No embedding returned".to_string()));
        }

        #[cfg(feature = "embeddings_remote")]
        if let Some(ref remote) = self.remote {
            match remote {
                RemoteEmbedder::OpenAI(oai) => return oai.embed(text),
            }
        }

        Err(GraphRagError::Embedding(
            "No embedder configured".to_string(),
        ))
    }

    /// Embed long text using chunked mean-pooling
    fn embed_chunked(&self, text: &str) -> Result<Vec<f32>, GraphRagError> {
        let max_chars = self.config.model.max_chars();
        let overlap_chars = max_chars / 5; // 20% overlap

        let chunks = split_with_overlap(text, max_chars, overlap_chars);

        if chunks.is_empty() {
            return Err(GraphRagError::Embedding("Empty text".to_string()));
        }

        #[cfg(feature = "embeddings")]
        if let Some(ref model) = self.local {
            let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
            let embeddings = model
                .embed(chunk_refs, None)
                .map_err(|e| GraphRagError::Embedding(e.to_string()))?;
            return Ok(mean_pool(&embeddings));
        }

        #[cfg(feature = "embeddings_remote")]
        if let Some(ref remote) = self.remote {
            match remote {
                RemoteEmbedder::OpenAI(oai) => {
                    let mut embeddings = Vec::new();
                    for chunk in chunks {
                        let emb = oai.embed(&chunk)?;
                        embeddings.push(emb);
                    }
                    return Ok(mean_pool(&embeddings));
                }
            }
        }

        Err(GraphRagError::Embedding(
            "No embedder configured".to_string(),
        ))
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

    #[test]
    fn test_default_embedder_cache_dir_uses_env_override() {
        let path = resolve_embedder_cache_dir(
            Some("custom-cache".to_string()),
            Some(PathBuf::from("platform-cache")),
            Some(PathBuf::from("home-dir")),
        );
        assert_eq!(path, PathBuf::from("custom-cache"));
    }

    #[test]
    fn test_default_embedder_cache_dir_uses_os_cache_dir() {
        let path = resolve_embedder_cache_dir(
            None,
            Some(PathBuf::from("platform-cache")),
            Some(PathBuf::from("home-dir")),
        );
        assert_eq!(path, PathBuf::from("platform-cache/graphrag/fastembed"));
    }

    #[test]
    fn test_default_embedder_cache_dir_falls_back_to_home_cache() {
        let path = resolve_embedder_cache_dir(None, None, Some(PathBuf::from("home-dir")));
        assert_eq!(path, PathBuf::from("home-dir/.cache/graphrag/fastembed"));
    }
}
