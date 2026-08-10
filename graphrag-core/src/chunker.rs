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

/// A chunk located in its source text.
///
/// Two consumers are served by one type. Tooling wants `start`/`end` — byte
/// offsets that slice the original exactly, so a chunk can be handed to `sed`,
/// an editor, or a patch without re-deriving where it came from. Humans want
/// `text` and line numbers, so a review pass can be read and cited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpan {
    /// Position in the chunk sequence, 0-based.
    pub index: usize,
    /// Byte offset of the first byte, inclusive.
    pub start: usize,
    /// Byte offset one past the last byte, exclusive.
    pub end: usize,
    /// 1-based line of `start`.
    pub line_start: usize,
    /// 1-based line of the last byte.
    pub line_end: usize,
    /// The chunk itself; always equal to `source[start..end]`.
    pub text: String,
}

impl ChunkSpan {
    /// Byte length of the chunk.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// True when the chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// A `sed` address range selecting this chunk's lines, e.g. `12,18`.
    ///
    /// Line-granular by nature: `sed` cannot address bytes. Use `start`/`end`
    /// directly for byte-exact edits.
    pub fn sed_range(&self) -> String {
        format!("{},{}", self.line_start, self.line_end)
    }
}

fn splitter_config(config: &ChunkerConfig) -> ChunkConfig<text_splitter::Characters> {
    match config.overlap {
        Some(overlap) => {
            ChunkConfig::new(config.chunk_size.saturating_sub(overlap)..config.chunk_size)
        }
        None => ChunkConfig::new(config.chunk_size),
    }
}

/// Chunk text and report where each chunk came from.
///
/// Same segmentation as [`chunk_text`], but retaining the byte offsets that
/// `chunks()` discards. Line numbers are computed in one pass over the source.
pub fn chunk_spans(text: &str, config: &ChunkerConfig) -> Vec<ChunkSpan> {
    if text.is_empty() {
        return vec![];
    }
    let cfg = splitter_config(config);
    let indexed: Vec<(usize, &str)> = if config.markdown {
        MarkdownSplitter::new(cfg).chunk_indices(text).collect()
    } else {
        TextSplitter::new(cfg).chunk_indices(text).collect()
    };

    // Prefix line counts, so line numbers cost one pass rather than one per chunk.
    let mut newline_before: Vec<usize> = Vec::with_capacity(text.len() + 1);
    let mut seen = 0usize;
    for b in text.bytes() {
        newline_before.push(seen);
        if b == b'\n' {
            seen += 1;
        }
    }
    newline_before.push(seen);
    let line_at = |off: usize| newline_before[off.min(text.len())] + 1;

    indexed
        .into_iter()
        .enumerate()
        .map(|(index, (start, chunk))| {
            let end = start + chunk.len();
            ChunkSpan {
                index,
                start,
                end,
                line_start: line_at(start),
                line_end: line_at(end.saturating_sub(1)),
                text: chunk.to_string(),
            }
        })
        .collect()
}

/// The `n`-th chunk of `text`, or `None` if there are fewer than `n + 1`.
///
/// Chunking is deterministic for a given config, so this re-derives the same
/// boundaries the full pass produced — letting a caller reference "chunk 12 of
/// this file" and fetch it later, as long as the file is unchanged.
pub fn nth_chunk(text: &str, n: usize, config: &ChunkerConfig) -> Option<ChunkSpan> {
    chunk_spans(text, config).into_iter().nth(n)
}

/// Render spans for human review: index, line range, size, and a preview.
pub fn review_spans(spans: &[ChunkSpan], preview: usize) -> String {
    let mut out = String::new();
    for s in spans {
        let head: String = s.text.chars().take(preview).collect();
        out.push_str(&format!(
            "[{:>3}] lines {:>5}-{:<5} bytes {:>7}-{:<7} ({} B)\n      {}\n",
            s.index,
            s.line_start,
            s.line_end,
            s.start,
            s.end,
            s.len(),
            head.replace('\n', " ⏎ ")
        ));
    }
    out
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
    chunk_text(text, &ChunkerConfig::default().with_markdown(false))
}

#[cfg(test)]
mod tests {
    // --- span form: offsets for tooling, text for humans ----------------------

    #[test]
    fn chunk_spans_report_byte_offsets_that_slice_the_original() {
        let text = "First paragraph here.\n\nSecond paragraph follows.\n\nThird one ends it.";
        let spans = chunk_spans(text, &ChunkerConfig::new(30));
        assert!(spans.len() > 1, "expected multiple chunks");
        for s in &spans {
            assert_eq!(
                &text[s.start..s.end],
                s.text,
                "span must slice the source exactly"
            );
            assert!(s.end > s.start);
        }
    }

    #[test]
    fn chunk_spans_are_ordered_and_non_overlapping_without_overlap_config() {
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi";
        let spans = chunk_spans(text, &ChunkerConfig::new(20).without_overlap());
        for w in spans.windows(2) {
            assert!(
                w[0].end <= w[1].start,
                "chunks must not overlap: {:?} then {:?}",
                w[0],
                w[1]
            );
            assert!(w[0].index + 1 == w[1].index, "index must be sequential");
        }
    }

    #[test]
    fn chunk_spans_cover_all_non_whitespace_content() {
        let text = "one two three four five six seven eight nine ten eleven twelve";
        let spans = chunk_spans(text, &ChunkerConfig::new(15).without_overlap());
        let joined: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        for word in text.split_whitespace() {
            assert!(joined.contains(word), "content dropped: {word}");
        }
    }

    #[test]
    fn nth_chunk_returns_the_same_span_as_the_full_pass() {
        let text = "Alpha section text.\n\nBeta section text.\n\nGamma section text here.";
        let cfg = ChunkerConfig::new(25);
        let all = chunk_spans(text, &cfg);
        for (i, expected) in all.iter().enumerate() {
            let got = nth_chunk(text, i, &cfg).expect("chunk exists");
            assert_eq!(&got, expected, "nth_chunk({i}) must match the full pass");
        }
        assert!(
            nth_chunk(text, all.len(), &cfg).is_none(),
            "out of range yields None"
        );
    }

    #[test]
    fn spans_carry_line_numbers_for_human_review() {
        let text = "line one\nline two\n\nline four is longer than the rest of them\n\nline six";
        let spans = chunk_spans(text, &ChunkerConfig::new(30));
        assert_eq!(spans[0].line_start, 1, "line numbers are 1-based");
        for w in spans.windows(2) {
            assert!(w[1].line_start >= w[0].line_start);
        }
        let last = spans.last().unwrap();
        assert!(last.line_end >= last.line_start);
    }

    #[test]
    fn empty_text_yields_no_spans() {
        assert!(chunk_spans("", &ChunkerConfig::default()).is_empty());
        assert!(nth_chunk("", 0, &ChunkerConfig::default()).is_none());
    }

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
