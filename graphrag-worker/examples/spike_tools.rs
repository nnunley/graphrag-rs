//! Helpers for scripts/extraction-spike: the production chunker and the
//! production extraction prompt, so the spike measures exactly what ships.
//!
//!   spike_tools plan SIZE OVERLAP   < text   → [{start, end, heading_path}]
//!   spike_tools prompt              < {"chunk": "...", "heading_path": [...], "known_classes": [...]}
//!                                   → {"system", "user", "max_tokens", "schema", "thinking"}

use graphrag_core::chunk_plan::plan_chunks;
use graphrag_core::fingerprints::ChunkerProfile;
use graphrag_llm::paper_json;
use serde_json::{Value, json};
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let out = match args.get(1).map(String::as_str) {
        Some("plan") => {
            let size = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
            let overlap = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
            let profile = ChunkerProfile {
                size,
                overlap,
                ..ChunkerProfile::default_markdown()
            };
            let (sections, chunks) = plan_chunks(&input, &profile);
            Value::Array(
                chunks
                    .iter()
                    .map(|c| {
                        json!({
                            "start": c.start,
                            "end": c.end,
                            "heading_path": c.section.map(|i| sections[i].heading_path.clone()).unwrap_or_default(),
                        })
                    })
                    .collect(),
            )
        }
        Some("prompt") => {
            let v: Value = serde_json::from_str(&input).expect("json input");
            let strs = |k: &str| -> Vec<String> {
                v[k].as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default()
            };
            let p = paper_json::build_prompt(
                v["chunk"].as_str().unwrap_or(""),
                &strs("heading_path"),
                &strs("known_classes"),
            );
            json!({
                "system": p.system,
                "user": p.user,
                "max_tokens": p.max_tokens,
                "schema": paper_json::response_schema(),
                "thinking": paper_json::ENABLE_THINKING,
                "prompt_version": paper_json::PROMPT_VERSION,
            })
        }
        _ => {
            eprintln!("usage: spike_tools plan SIZE OVERLAP | prompt");
            std::process::exit(64);
        }
    };
    println!("{out}");
}
