//! Property-based invariants for chunking.
//!
//! Example tests prove behaviour on the inputs someone thought to write down.
//! These state what must hold for EVERY input, and let Hegel search for a
//! counterexample — which is the difference between "works on my cases" and a
//! completeness argument.
//!
//! Motivating failure: a build shipped where `--code`/`--lang` were parsed but
//! entirely ignored, and every positive example still passed because prose
//! chunking produced plausible-looking output. Only a negative case (an
//! unsupported extension that should error) exposed it.

use graphrag_core::chunker::{
    ChunkerConfig, chunk_spans, fingerprint_of, line_spans, nth_chunk, nth_chunk_verified,
    resize_span,
};
use hegel::TestCase;
use hegel::generators as gs;

/// Configs worth exploring: sizes that force many splits, with and without overlap.
fn draw_config(tc: &mut TestCase) -> ChunkerConfig {
    let size = tc.draw(gs::integers::<usize>().min_value(20).max_value(4000));
    let markdown = tc.draw(gs::booleans());
    let cfg = ChunkerConfig::new(size).with_markdown(markdown);
    if tc.draw(gs::booleans()) {
        cfg.without_overlap()
    } else {
        cfg.with_overlap(tc.draw(gs::integers::<usize>().min_value(0).max_value(size / 2)))
    }
}

#[hegel::test]
fn every_span_slices_its_source_exactly(mut tc: TestCase) {
    let text = tc.draw(gs::text().max_size(4000));
    let cfg = draw_config(&mut tc);
    for s in chunk_spans(&text, &cfg) {
        assert_eq!(
            &text[s.start..s.end],
            s.text,
            "span {} must slice the source",
            s.index
        );
    }
}

#[hegel::test]
fn spans_are_ordered_and_sequentially_indexed(mut tc: TestCase) {
    let text = tc.draw(gs::text().max_size(4000));
    let cfg = draw_config(&mut tc);
    let spans = chunk_spans(&text, &cfg);
    for (i, w) in spans.windows(2).enumerate() {
        assert!(w[0].start <= w[1].start, "spans must ascend by offset");
        assert_eq!(w[1].index, i + 1, "index must be sequential");
    }
}

#[hegel::test]
fn fingerprint_always_matches_the_chunk_body(mut tc: TestCase) {
    let text = tc.draw(gs::text().max_size(4000));
    let cfg = draw_config(&mut tc);
    for s in chunk_spans(&text, &cfg) {
        assert_eq!(
            s.fingerprint,
            fingerprint_of(&s.text),
            "fingerprint must be derived from the chunk it labels"
        );
    }
}

#[hegel::test]
fn nth_chunk_agrees_with_the_full_pass(mut tc: TestCase) {
    let text = tc.draw(gs::text().max_size(4000));
    let cfg = draw_config(&mut tc);
    let all = chunk_spans(&text, &cfg);
    for (i, want) in all.iter().enumerate() {
        assert_eq!(nth_chunk(&text, i, &cfg).as_ref(), Some(want));
    }
    // one past the end is always absent, whatever the input
    assert!(nth_chunk(&text, all.len(), &cfg).is_none());
}

#[hegel::test]
fn verification_accepts_the_truth_and_rejects_everything_else(mut tc: TestCase) {
    let text = tc.draw(gs::text().max_size(4000));
    let cfg = draw_config(&mut tc);
    let all = chunk_spans(&text, &cfg);
    for (i, s) in all.iter().enumerate() {
        // positive: the recorded fingerprint verifies
        assert!(nth_chunk_verified(&text, i, &cfg, &s.fingerprint).is_ok());
        // negative: any other fingerprint must be refused
        assert!(
            nth_chunk_verified(&text, i, &cfg, "xxh3:0000000000000000").is_err(),
            "a wrong fingerprint must never verify"
        );
    }
}

#[hegel::test]
fn line_numbers_are_one_based_and_monotonic(mut tc: TestCase) {
    let text = tc.draw(gs::text().max_size(4000));
    let cfg = draw_config(&mut tc);
    let spans = chunk_spans(&text, &cfg);
    for s in &spans {
        assert!(s.line_start >= 1, "lines are 1-based");
        assert!(s.line_end >= s.line_start);
        assert!(s.line_start <= text.lines().count().max(1));
    }
    for w in spans.windows(2) {
        assert!(
            w[1].line_start >= w[0].line_start,
            "line numbers must not go backwards"
        );
    }
}

#[hegel::test]
fn chunking_is_deterministic(mut tc: TestCase) {
    let text = tc.draw(gs::text().max_size(4000));
    let cfg = draw_config(&mut tc);
    assert_eq!(
        chunk_spans(&text, &cfg),
        chunk_spans(&text, &cfg),
        "same input and config must always yield the same spans"
    );
}

#[hegel::test]
fn line_chunking_is_lossless_and_never_splits_a_line(tc: TestCase) {
    let text = tc.draw(gs::text().max_size(4000));
    let budget = tc.draw(gs::integers::<usize>().min_value(1).max_value(500));
    let spans = line_spans(&text, budget);
    let joined: String = spans.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(
        joined, text,
        "concatenating line spans must reproduce the source"
    );
    for s in &spans {
        assert_eq!(&text[s.start..s.end], s.text);
        let body = s.text.trim_end_matches('\n');
        // within budget, or a single line that alone exceeds it
        assert!(
            s.len() <= budget || !body.contains('\n'),
            "an over-budget chunk must be one indivisible line"
        );
    }
}

#[hegel::test]
fn resize_is_total_and_always_slices_exactly(tc: TestCase) {
    let text = tc.draw(gs::text().max_size(2000));
    let budget = tc.draw(gs::integers::<usize>().min_value(1).max_value(400));
    let delta = tc.draw(gs::integers::<i32>().min_value(-20).max_value(20)) as isize;
    for s in line_spans(&text, budget) {
        let got = resize_span(&text, &s, delta);
        assert!(got.end <= text.len(), "never past the end of the source");
        assert!(got.start <= got.end, "never inverted");
        assert_eq!(&text[got.start..got.end], got.text, "must slice exactly");
        assert_eq!(got.fingerprint, fingerprint_of(&got.text));
        if delta > 0 {
            assert!(
                got.start <= s.start && got.end >= s.end,
                "expansion is a superset"
            );
        }
        if delta < 0 && !s.text.trim().is_empty() {
            assert!(
                !got.text.trim().is_empty(),
                "contraction keeps at least one line"
            );
        }
    }
}
