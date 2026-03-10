//! Text chunking module using text-splitter for semantic chunking
//!
//! Provides intelligent text splitting that respects semantic boundaries
//! like paragraphs, sentences, and markdown structure.

use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};

/// Configuration for text chunking
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Target chunk size in characters
    pub chunk_size: usize,
    /// Whether to use markdown-aware splitting
    pub markdown: bool,
    /// Optional overlap range (min..max chars)
    pub overlap: Option<usize>,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 2000,
            markdown: true,
            overlap: Some(200),
        }
    }
}

impl ChunkerConfig {
    /// Create a new config with specified chunk size
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            ..Default::default()
        }
    }

    /// Set markdown mode
    pub fn with_markdown(mut self, markdown: bool) -> Self {
        self.markdown = markdown;
        self
    }

    /// Set overlap size
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = Some(overlap);
        self
    }

    /// Disable overlap
    pub fn without_overlap(mut self) -> Self {
        self.overlap = None;
        self
    }
}

/// Chunk text into semantic segments
///
/// Uses text-splitter's semantic chunking which respects:
/// - Paragraph boundaries
/// - Sentence boundaries
/// - Word boundaries
/// - For markdown: headers, code blocks, lists, etc.
pub fn chunk_text(text: &str, config: &ChunkerConfig) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    // Use a range if overlap is specified to allow some flexibility
    let chunk_config = if let Some(overlap) = config.overlap {
        let min_size = config.chunk_size.saturating_sub(overlap);
        ChunkConfig::new(min_size..config.chunk_size)
    } else {
        ChunkConfig::new(config.chunk_size)
    };

    if config.markdown {
        let splitter = MarkdownSplitter::new(chunk_config);
        splitter.chunks(text).map(String::from).collect()
    } else {
        let splitter = TextSplitter::new(chunk_config);
        splitter.chunks(text).map(String::from).collect()
    }
}

/// Chunk markdown text with default settings (2000 chars, 200 overlap)
pub fn chunk_markdown(text: &str) -> Vec<String> {
    chunk_text(text, &ChunkerConfig::default())
}

/// Chunk plain text with default settings
pub fn chunk_plain(text: &str) -> Vec<String> {
    chunk_text(
        text,
        &ChunkerConfig::default().with_markdown(false),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text() {
        let chunks = chunk_text("", &ChunkerConfig::default());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_small_text() {
        let text = "Hello, world!";
        let chunks = chunk_text(text, &ChunkerConfig::new(1000));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_markdown_chunking() {
        let text = r#"# Header 1

This is a paragraph with some text.

## Header 2

Another paragraph here.

### Header 3

More content.
"#;
        let chunks = chunk_markdown(text);
        assert!(!chunks.is_empty());
        // Should respect header boundaries
        for chunk in &chunks {
            println!("Chunk: {} chars", chunk.len());
        }
    }
}
