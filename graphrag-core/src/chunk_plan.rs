//! Section-bounded chunk planning.
//!
//! Each section's own text (excluding its subsections) is chunked
//! separately, so no chunk crosses a section boundary and small adjacent
//! sections are never merged. A region that fits in `size` characters is one
//! chunk; a longer one is split with overlap inside that region only. Chunks
//! are spans; the text stays in the document.

use crate::fingerprints::ChunkerProfile;
use crate::sections::{SectionSpan, parse_sections};
use crate::token::fingerprint_text;
use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};

/// One planned chunk: a byte span of the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedChunk {
    /// Index into the sections Vec; `None` for a document without headings.
    pub section: Option<usize>,
    pub start: usize,
    pub end: usize,
    /// `fingerprint_text` of `text[start..end]`.
    pub content_fp: String,
}

/// Byte ranges of each section's own text: from its start to its first
/// subsection (or its end).
fn own_regions(sections: &[SectionSpan]) -> Vec<(Option<usize>, usize, usize)> {
    sections
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let first_child = sections
                .iter()
                .filter(|c| c.parent == Some(i))
                .map(|c| c.start)
                .min()
                .unwrap_or(s.end);
            (Some(i), s.start, first_child)
        })
        .collect()
}

fn trimmed(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let slice = &text[start..end];
    let lead = slice.len() - slice.trim_start().len();
    let body = slice.trim();
    (!body.is_empty()).then(|| (start + lead, start + lead + body.len()))
}

pub fn plan_chunks(text: &str, profile: &ChunkerProfile) -> (Vec<SectionSpan>, Vec<PlannedChunk>) {
    let markdown = profile.kind != "code";
    let sections = if markdown {
        parse_sections(text)
    } else {
        Vec::new()
    };
    let regions = if sections.is_empty() {
        vec![(None, 0, text.len())]
    } else {
        own_regions(&sections)
    };
    let config = || {
        ChunkConfig::new(profile.size)
            .with_overlap(profile.overlap.min(profile.size.saturating_sub(1)))
            .expect("overlap is below size")
    };
    let md = MarkdownSplitter::new(config());
    let plain = TextSplitter::new(config());

    let mut chunks = Vec::new();
    for (section, rs, re) in regions {
        let Some((start, end)) = trimmed(text, rs, re) else {
            continue;
        };
        let region = &text[start..end];
        let mut push = |s: usize, e: usize| {
            chunks.push(PlannedChunk {
                section,
                start: s,
                end: e,
                content_fp: fingerprint_text(&text[s..e]),
            })
        };
        if region.chars().count() <= profile.size {
            push(start, end);
            continue;
        }
        let pieces: Vec<(usize, &str)> = if markdown {
            md.chunk_indices(region).collect()
        } else {
            plain.chunk_indices(region).collect()
        };
        for (offset, piece) in pieces {
            push(start + offset, start + offset + piece.len());
        }
    }
    (sections, chunks)
}
