//! Markdown section tree with byte spans.
//!
//! ATX headings (`#`..`######`) open sections; a section runs until the next
//! heading of the same or a higher level, so it contains its subsections.
//! Headings inside fenced code blocks don't count. Text before the first
//! heading becomes an untitled level-0 section when it isn't blank. A
//! document with no headings has no sections.

/// One section of a markdown document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionSpan {
    /// Position among earlier sections with the same `heading_path`.
    pub ordinal: usize,
    /// Heading level 1–6; 0 for the untitled leading section.
    pub level: u8,
    pub heading: String,
    /// Headings from the top-level ancestor down to this section.
    pub heading_path: Vec<String>,
    /// Index of the enclosing section in the returned Vec.
    pub parent: Option<usize>,
    /// Byte offset of the heading line (or 0 for the leading section).
    pub start: usize,
    /// Byte offset one past the section's last byte.
    pub end: usize,
}

/// `(level, heading text)` when `line` is an ATX heading.
fn heading(line: &str) -> Option<(u8, String)> {
    let line = line.trim_end_matches(['\n', '\r']);
    let indent = line.len() - line.trim_start_matches(' ').len();
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let hashes = rest.len() - rest.trim_start_matches('#').len();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let after = &rest[hashes..];
    if !(after.is_empty() || after.starts_with([' ', '\t'])) {
        return None;
    }
    let mut text = after.trim();
    // Optional closing sequence: ` ###`.
    let stripped = text.trim_end_matches('#');
    if stripped.len() < text.len() && (stripped.is_empty() || stripped.ends_with([' ', '\t'])) {
        text = stripped.trim_end();
    }
    Some((hashes as u8, text.to_string()))
}

/// `(char, run length)` when `line` opens or closes a code fence.
fn fence(line: &str) -> Option<(char, usize)> {
    let t = line.trim_start_matches(' ');
    if line.len() - t.len() > 3 {
        return None;
    }
    for c in ['`', '~'] {
        let n = t.len() - t.trim_start_matches(c).len();
        if n >= 3 {
            return Some((c, n));
        }
    }
    None
}

pub fn parse_sections(text: &str) -> Vec<SectionSpan> {
    // (start, level, heading) of every heading outside fenced code.
    let mut heads: Vec<(usize, u8, String)> = Vec::new();
    let mut open_fence: Option<(char, usize)> = None;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        match (open_fence, fence(line)) {
            (None, Some(f)) => open_fence = Some(f),
            (Some((c, n)), Some((c2, n2))) if c == c2 && n2 >= n => open_fence = None,
            (None, None) => {
                if let Some((level, h)) = heading(line) {
                    heads.push((offset, level, h));
                }
            }
            _ => {}
        }
        offset += line.len();
    }
    if heads.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<SectionSpan> = Vec::new();
    let first = heads[0].0;
    if first > 0 {
        if text[..first].trim().is_empty() {
            heads[0].0 = 0;
        } else {
            out.push(SectionSpan {
                ordinal: 0,
                level: 0,
                heading: String::new(),
                heading_path: Vec::new(),
                parent: None,
                start: 0,
                end: first,
            });
        }
    }

    let mut stack: Vec<usize> = Vec::new();
    for (i, (start, level, h)) in heads.iter().enumerate() {
        let end = heads[i + 1..]
            .iter()
            .find(|(_, l, _)| l <= level)
            .map_or(text.len(), |(s, _, _)| *s);
        while stack.last().is_some_and(|&p| out[p].level >= *level) {
            stack.pop();
        }
        let parent = stack.last().copied();
        let mut heading_path = parent.map_or_else(Vec::new, |p| out[p].heading_path.clone());
        heading_path.push(h.clone());
        let ordinal = out
            .iter()
            .filter(|s| s.heading_path == heading_path)
            .count();
        out.push(SectionSpan {
            ordinal,
            level: *level,
            heading: h.clone(),
            heading_path,
            parent,
            start: *start,
            end,
        });
        stack.push(out.len() - 1);
    }
    out
}
