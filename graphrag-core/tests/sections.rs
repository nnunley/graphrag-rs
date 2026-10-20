//! Task 3: markdown section tree with byte spans.

use graphrag_core::sections::{SectionSpan, parse_sections};

fn paths(s: &[SectionSpan]) -> Vec<Vec<String>> {
    s.iter().map(|x| x.heading_path.clone()).collect()
}

fn v(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|x| x.to_string()).collect()
}

#[test]
fn nested_headings_build_paths() {
    let text = "# A\na\n## B\nb\n## C\nc\n# D\nd\n";
    let s = parse_sections(text);
    assert_eq!(
        paths(&s),
        vec![v(&["A"]), v(&["A", "B"]), v(&["A", "C"]), v(&["D"])]
    );
    assert_eq!(s[0].parent, None);
    assert_eq!(s[1].parent, Some(0));
    assert_eq!(s[2].parent, Some(0));
    assert_eq!(s[3].parent, None);
    assert_eq!(s[0].level, 1);
    assert_eq!(s[1].level, 2);
    // A spans through its subsections, up to D.
    assert_eq!(&text[s[0].start..s[0].end], "# A\na\n## B\nb\n## C\nc\n");
    assert_eq!(&text[s[1].start..s[1].end], "## B\nb\n");
    assert_eq!(&text[s[3].start..s[3].end], "# D\nd\n");
}

#[test]
fn leading_text_becomes_untitled_section() {
    let text = "intro line\n\n# A\nbody\n";
    let s = parse_sections(text);
    assert_eq!(s.len(), 2);
    assert_eq!(s[0].level, 0);
    assert_eq!(s[0].heading, "");
    assert!(s[0].heading_path.is_empty());
    assert_eq!(&text[s[0].start..s[0].end], "intro line\n\n");
    assert_eq!(s[1].heading, "A");
}

#[test]
fn whitespace_before_first_heading_is_not_a_section() {
    let s = parse_sections("\n\n# A\nx\n");
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].heading, "A");
    assert_eq!(
        s[0].start, 0,
        "leading whitespace belongs to the first section"
    );
}

#[test]
fn no_headings_no_sections() {
    assert!(parse_sections("just a note\nwith two lines\n").is_empty());
    assert!(parse_sections("").is_empty());
}

#[test]
fn spans_slice_the_text_exactly() {
    let fixtures = [
        "# A\na\n## B\nb\n# C\n",
        "lead\n# A\n### deep\nx\n## mid\ny\n",
        "# only heading",
        "pre\n```\n# no\n```\n# Yes\nz",
    ];
    for text in fixtures {
        let s = parse_sections(text);
        let top: String = s
            .iter()
            .filter(|x| x.parent.is_none())
            .map(|x| &text[x.start..x.end])
            .collect();
        assert_eq!(top, text, "fixture {text:?}");
        for x in &s {
            if let Some(p) = x.parent {
                assert!(s[p].start <= x.start && x.end <= s[p].end);
            }
        }
    }
}

#[test]
fn fenced_code_hash_is_not_a_heading() {
    let text = "# Real\n```bash\n# not a heading\necho hi\n```\n~~~\n## also not\n~~~\ntext\n";
    let s = parse_sections(text);
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].heading, "Real");
    assert_eq!(s[0].end, text.len());
}

#[test]
fn hash_without_space_is_not_a_heading() {
    let s = parse_sections("#hashtag\n# Real\n");
    assert_eq!(s.len(), 2);
    assert_eq!(s[0].level, 0);
    assert_eq!(s[1].heading, "Real");
}

#[test]
fn multibyte_headings_keep_char_boundaries() {
    let text = "前文 🙂\n# 日本語 🚀\n本文です。\n## é ü\n漢字\n";
    let s = parse_sections(text);
    assert_eq!(s.len(), 3);
    assert_eq!(s[1].heading, "日本語 🚀");
    for x in &s {
        assert!(text.is_char_boundary(x.start) && text.is_char_boundary(x.end));
    }
}

#[test]
fn duplicate_sibling_headings_get_ordinals() {
    let text = "# Top\n## Notes\na\n## Notes\nb\n## Other\n";
    let s = parse_sections(text);
    let notes: Vec<_> = s.iter().filter(|x| x.heading == "Notes").collect();
    assert_eq!(notes.len(), 2);
    assert_eq!(notes[0].ordinal, 0);
    assert_eq!(notes[1].ordinal, 1);
    assert_eq!(s.iter().find(|x| x.heading == "Other").unwrap().ordinal, 0);
}

#[test]
fn closing_hashes_are_trimmed() {
    let s = parse_sections("## Title ##\nx\n");
    assert_eq!(s[0].heading, "Title");
}
