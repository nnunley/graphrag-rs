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
    /// `sha256:` + 64 lowercase hex over the chunk text.
    ///
    /// Offsets locate a chunk; the fingerprint identifies it. A stored
    /// reference ("chunk 12 of this file") is positional and silently returns
    /// different content once the file is edited — carrying the fingerprint
    /// makes that detectable instead.
    pub fingerprint: String,
}

/// Fingerprint chunk content with xxh3-64.
///
/// Chunk identity is scoped to one file, where 64 bits is ample and the 16-hex
/// rendering stays readable in a TSV row or a stored reference. Use
/// [`fingerprint_wide`] when identity must hold across a whole corpus.
pub fn fingerprint_of(text: &str) -> String {
    format!("xxh3:{:016x}", xxhash_rust::xxh3::xxh3_64(text.as_bytes()))
}

/// Fingerprint with xxh3-128, for identity across a corpus rather than a file.
pub fn fingerprint_wide(text: &str) -> String {
    format!(
        "xxh3-128:{:032x}",
        xxhash_rust::xxh3::xxh3_128(text.as_bytes())
    )
}

/// Why a verified chunk lookup failed.
#[derive(Debug, thiserror::Error)]
pub enum ChunkError {
    #[error("chunk {index} does not exist (source yields {available})")]
    OutOfRange { index: usize, available: usize },
    #[error("chunk {index} no longer matches: expected {expected}, found {found}")]
    FingerprintMismatch {
        index: usize,
        expected: String,
        found: String,
    },
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

    spans_from_indices(text, indexed)
}

/// Build spans from `(offset, chunk)` pairs. Shared by the text and code
/// chunkers so both produce identical span semantics.
pub(crate) fn spans_from_indices(text: &str, indexed: Vec<(usize, &str)>) -> Vec<ChunkSpan> {
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
                fingerprint: fingerprint_of(chunk),
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

/// Fetch the `n`-th chunk and verify it still holds the expected content.
///
/// This is what makes a stored chunk reference safe across time: the index
/// locates it, the fingerprint proves the file has not shifted underneath.
pub fn nth_chunk_verified(
    text: &str,
    n: usize,
    config: &ChunkerConfig,
    expected_fingerprint: &str,
) -> Result<ChunkSpan, ChunkError> {
    let spans = chunk_spans(text, config);
    let available = spans.len();
    let span = spans.into_iter().nth(n).ok_or(ChunkError::OutOfRange {
        index: n,
        available,
    })?;
    if span.fingerprint != expected_fingerprint {
        return Err(ChunkError::FingerprintMismatch {
            index: n,
            expected: expected_fingerprint.to_string(),
            found: span.fingerprint,
        });
    }
    Ok(span)
}

/// Render spans for human review: index, line range, size, and a preview.
pub fn review_spans(spans: &[ChunkSpan], preview: usize) -> String {
    let mut out = String::new();
    for s in spans {
        let head: String = s.text.chars().take(preview).collect();
        out.push_str(&format!(
            "[{:>3}] lines {:>5}-{:<5} bytes {:>7}-{:<7} ({} B) {}\n      {}\n",
            s.index,
            s.line_start,
            s.line_end,
            s.start,
            s.end,
            s.len(),
            s.fingerprint,
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
    // --- fingerprints: make a chunk reference verifiable ----------------------

    #[test]
    fn span_fingerprint_has_the_canonical_shape() {
        // xxh3-64: short enough to read and paste, ample for chunk identity
        // within a file. sha256 would be 64 hex chars for no added benefit here.
        let spans = chunk_spans("some text to split into a chunk", &ChunkerConfig::new(100));
        let fp = &spans[0].fingerprint;
        assert!(fp.starts_with("xxh3:"), "got {fp}");
        let hex = fp.strip_prefix("xxh3:").unwrap();
        assert_eq!(hex.len(), 16, "xxh3-64 renders as 16 hex chars");
        assert!(
            hex.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "lowercase hex only: {hex}"
        );
    }

    #[test]
    fn wide_fingerprint_is_available_for_cross_corpus_identity() {
        let fp = fingerprint_wide("some text");
        assert!(fp.starts_with("xxh3-128:"), "got {fp}");
        assert_eq!(fp.strip_prefix("xxh3-128:").unwrap().len(), 32);
        assert_eq!(fingerprint_wide("some text"), fp, "deterministic");
        assert_ne!(fingerprint_wide("some other text"), fp);
    }

    #[test]
    fn fingerprint_is_deterministic_and_content_addressed() {
        let cfg = ChunkerConfig::new(40);
        let a = chunk_spans("alpha beta gamma delta epsilon zeta", &cfg);
        let b = chunk_spans("alpha beta gamma delta epsilon zeta", &cfg);
        assert_eq!(
            a[0].fingerprint, b[0].fingerprint,
            "same content, same fingerprint"
        );
        let c = chunk_spans("alpha beta gamma delta epsilon ZETA", &cfg);
        assert_ne!(
            a[0].fingerprint, c[0].fingerprint,
            "changed content must change it"
        );
    }

    #[test]
    fn fingerprint_covers_the_chunk_not_its_position() {
        // The same chunk text appearing at a different offset keeps its fingerprint:
        // it identifies content, so a reference survives edits elsewhere in the file.
        let cfg = ChunkerConfig::new(30).without_overlap();
        let one = chunk_spans("PREFIX.\n\nstable body text here.", &cfg);
        let two = chunk_spans("A MUCH LONGER PREFIX.\n\nstable body text here.", &cfg);
        let a = one.iter().find(|s| s.text.contains("stable body")).unwrap();
        let b = two.iter().find(|s| s.text.contains("stable body")).unwrap();
        assert_ne!(a.start, b.start, "offsets differ");
        assert_eq!(
            a.fingerprint, b.fingerprint,
            "fingerprint tracks content, not position"
        );
    }

    #[test]
    fn nth_chunk_verified_accepts_a_matching_fingerprint() {
        let text = "First part here.\n\nSecond part follows on.\n\nThird part ends.";
        let cfg = ChunkerConfig::new(25);
        let want = nth_chunk(text, 1, &cfg).unwrap();
        let got =
            nth_chunk_verified(text, 1, &cfg, &want.fingerprint).expect("fingerprint matches");
        assert_eq!(got, want);
    }

    #[test]
    fn nth_chunk_verified_detects_a_changed_file() {
        let cfg = ChunkerConfig::new(25);
        let before = "First part here.\n\nSecond part follows on.\n\nThird part ends.";
        let reference = nth_chunk(before, 1, &cfg).unwrap().fingerprint;
        // the file is edited; chunk 1 no longer holds what the reference recorded
        let after = "First part here.\n\nSecond part REWRITTEN entirely.\n\nThird part ends.";
        let err = nth_chunk_verified(after, 1, &cfg, &reference).unwrap_err();
        assert!(
            matches!(err, ChunkError::FingerprintMismatch { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn nth_chunk_verified_reports_a_missing_chunk() {
        let cfg = ChunkerConfig::new(1000);
        let err = nth_chunk_verified("short", 5, &cfg, "xxh3:00").unwrap_err();
        assert!(
            matches!(err, ChunkError::OutOfRange { index: 5, .. }),
            "{err:?}"
        );
    }

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
