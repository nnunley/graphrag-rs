//! Task 4: section-bounded chunk planning.

use graphrag_core::chunk_plan::plan_chunks;
use graphrag_core::fingerprints::ChunkerProfile;

fn words(n: usize, seed: &str) -> String {
    let mut s = String::new();
    let mut i = 0;
    while s.len() < n {
        s.push_str(seed);
        s.push_str(&format!("{i} "));
        if i % 12 == 11 {
            s.push_str(".\n\n");
        }
        i += 1;
    }
    s
}

#[test]
fn small_sections_are_never_merged() {
    let text = format!(
        "# A\n{}\n# B\n{}\n# C\n{}\n",
        words(100, "a"),
        words(100, "b"),
        words(100, "c")
    );
    let (sections, chunks) = plan_chunks(&text, &ChunkerProfile::default_markdown());
    assert_eq!(sections.len(), 3);
    assert_eq!(chunks.len(), 3);
    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(c.section, Some(i));
    }
}

#[test]
fn long_section_splits_inside_itself() {
    let text = format!("# Short\nx\n# Long\n{}\n# After\ny\n", words(5000, "w"));
    let (sections, chunks) = plan_chunks(&text, &ChunkerProfile::default_markdown());
    let long = sections.iter().position(|s| s.heading == "Long").unwrap();
    let inside: Vec<_> = chunks.iter().filter(|c| c.section == Some(long)).collect();
    assert!(inside.len() >= 3, "got {}", inside.len());
    for c in &inside {
        assert!(sections[long].start <= c.start && c.end <= sections[long].end);
    }
    for c in &chunks {
        let s = &sections[c.section.unwrap()];
        assert!(
            s.start <= c.start && c.end <= s.end,
            "chunk crosses its section"
        );
    }
}

#[test]
fn long_section_chunks_overlap() {
    let text = words(5000, "w");
    let (_, chunks) = plan_chunks(&text, &ChunkerProfile::default_markdown());
    assert!(chunks.len() >= 3);
    assert!(
        chunks.windows(2).any(|w| w[1].start < w[0].end),
        "consecutive chunks of a long region share an overlap"
    );
}

#[test]
fn no_chunk_exceeds_hard_cap() {
    let profile = ChunkerProfile::default_markdown();
    let flat = "x".repeat(442_000);
    let prose = words(442_000, "lorem");
    let headed = format!("# H\n{}\n## I\n{}", "y".repeat(221_000), prose);
    for text in [flat, prose.clone(), headed] {
        let (_, chunks) = plan_chunks(&text, &profile);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(
                text[c.start..c.end].chars().count() <= 2000,
                "chunk too long"
            );
        }
    }
}

#[test]
fn heading_less_note_chunks_without_sections() {
    let (sections, chunks) = plan_chunks("a short note", &ChunkerProfile::default_markdown());
    assert!(sections.is_empty());
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].section, None);
    assert_eq!((chunks[0].start, chunks[0].end), (0, 12));
}

#[test]
fn parent_section_text_excludes_subsections() {
    let text = "# A\nintro of a\n## B\nbody of b\n";
    let (sections, chunks) = plan_chunks(text, &ChunkerProfile::default_markdown());
    assert_eq!(sections.len(), 2);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].section, Some(0));
    assert!(text[chunks[0].start..chunks[0].end].contains("intro of a"));
    assert!(!text[chunks[0].start..chunks[0].end].contains("body of b"));
    assert_eq!(chunks[1].section, Some(1));
}

#[test]
fn unchanged_section_keeps_identical_chunk_fps() {
    let one = words(3000, "one");
    let two = words(300, "two");
    let before = format!("# One\n{one}\n# Two\n{two}\n");
    let after = format!("# One\n{one}\n# Two\n{two} edited\n");
    let p = ChunkerProfile::default_markdown();
    let (sa, ca) = plan_chunks(&before, &p);
    let (sb, cb) = plan_chunks(&after, &p);
    let fps = |s: &[graphrag_core::sections::SectionSpan],
               c: &[graphrag_core::chunk_plan::PlannedChunk],
               h: &str| {
        let idx = s.iter().position(|x| x.heading == h).unwrap();
        c.iter()
            .filter(|x| x.section == Some(idx))
            .map(|x| x.content_fp.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(fps(&sa, &ca, "One"), fps(&sb, &cb, "One"));
    assert_ne!(fps(&sa, &ca, "Two"), fps(&sb, &cb, "Two"));
}

#[test]
fn multibyte_chunk_offsets_are_char_boundaries() {
    let body = "日本語のテキスト🙂 é ".repeat(800);
    let text = format!("# 見出し\n{body}\n");
    let (_, chunks) = plan_chunks(&text, &ChunkerProfile::default_markdown());
    assert!(chunks.len() > 1);
    for c in &chunks {
        assert!(text.is_char_boundary(c.start) && text.is_char_boundary(c.end));
        assert_eq!(
            c.content_fp,
            graphrag_core::token::fingerprint_text(&text[c.start..c.end])
        );
    }
}

#[test]
fn heading_only_section_is_one_chunk() {
    let (sections, chunks) = plan_chunks("# A\n\n# B\ntext\n", &ChunkerProfile::default_markdown());
    assert_eq!(sections.len(), 2);
    // "# A\n\n" still carries its heading text, so it is one chunk.
    assert_eq!(chunks.len(), 2);
}
