fn main() {
    let text = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
    for (label, chunks) in [
        ("chunk_plain", graphrag_core::chunk_plain(&text)),
        ("chunk_markdown", graphrag_core::chunk_markdown(&text)),
    ] {
        let n = chunks.len();
        let lens: Vec<usize> = chunks.iter().map(|c| c.len()).collect();
        let mid_sentence = chunks
            .iter()
            .filter(|c| {
                let t = c.trim_end();
                !t.is_empty() && !t.ends_with(['.', '!', '?', ':', ')', '"', '`'])
            })
            .count();
        println!(
            "{label:16} {n:3} chunks  mean {:5.0}  mid-sentence cuts {}/{}",
            lens.iter().sum::<usize>() as f64 / n as f64,
            mid_sentence,
            n
        );
    }
}
