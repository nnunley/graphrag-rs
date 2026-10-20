//! Fakes for worker tests: a deterministic LLM and embedder.
#![allow(dead_code)]

use graphrag_core::GraphRagError;
use graphrag_core::text_embedder::TextEmbedder;
use graphrag_llm::{ChatClient, ChatPrompt};
use graphrag_worker::WorkerLlm;
use serde_json::{Value, json};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Normal,
    Transport,
    Malformed,
}

pub struct FakeLlm {
    pub models: Vec<String>,
    pub calls: AtomicUsize,
    pub mode: Mutex<Mode>,
    /// The chunk text of each extraction call, in call order.
    pub extracted: Mutex<Vec<String>>,
}

impl FakeLlm {
    pub fn new() -> Self {
        Self {
            models: vec!["qwen3.8-27b".into()],
            calls: AtomicUsize::new(0),
            mode: Mutex::new(Mode::Normal),
            extracted: Mutex::new(Vec::new()),
        }
    }
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
    pub fn set_mode(&self, m: Mode) {
        *self.mode.lock().unwrap() = m;
    }
}

/// Capitalized words become entities (kind "thing"); consecutive ones relate.
fn extraction(text: &str) -> String {
    let mut names: Vec<String> = Vec::new();
    for w in text.split(|c: char| !c.is_alphanumeric()) {
        if w.len() > 1 && w.chars().next().unwrap().is_uppercase() && !names.iter().any(|n| n == w)
        {
            names.push(w.to_string());
        }
    }
    let entities: Vec<Value> = names
        .iter()
        .map(|n| json!({"name": n, "type": "Concept", "description": format!("{n} in the text"), "kind": "topic", "is_class": false}))
        .collect();
    let rels: Vec<Value> = names
        .windows(2)
        .map(|w| json!({"source": w[0], "target": w[1], "relation": "associated_with", "description": "co-occur", "strength": 5}))
        .collect();
    json!({"entities": entities, "relationships": rels}).to_string()
}

impl ChatClient for FakeLlm {
    fn complete(&self, prompt: &ChatPrompt) -> Result<String, String> {
        self.complete_with_format(prompt, None)
    }

    fn complete_with_format(
        &self,
        prompt: &ChatPrompt,
        _format: Option<&Value>,
    ) -> Result<String, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match *self.mode.lock().unwrap() {
            Mode::Transport => return Err("transport: connection refused".into()),
            Mode::Malformed => return Ok("this is not json".into()),
            Mode::Normal => {}
        }
        if prompt.system.contains("extract a knowledge graph") {
            let text = prompt
                .user
                .split("this text:\n\n")
                .nth(1)
                .unwrap_or("")
                .to_string();
            self.extracted.lock().unwrap().push(text.clone());
            return Ok(extraction(&text));
        }
        if prompt.system.contains("summarize one community") {
            let first = prompt
                .user
                .lines()
                .find_map(|l| {
                    l.strip_prefix("- ")
                        .map(|r| r.split(" (").next().unwrap_or("").to_string())
                })
                .unwrap_or_default();
            return Ok(format!(
                "This community centers on {first} and the items it relates to."
            ));
        }
        Ok(json!({"mappings": []}).to_string())
    }
}

impl WorkerLlm for FakeLlm {
    fn served_models(&self) -> Result<Vec<String>, String> {
        Ok(self.models.clone())
    }
}

pub struct FakeEmbedder {
    pub calls: AtomicUsize,
}

impl FakeEmbedder {
    pub fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl TextEmbedder for FakeEmbedder {
    fn id(&self) -> String {
        "fake:hash-8".into()
    }
    fn dim(&self) -> usize {
        8
    }
    fn embed(&self, text: &str) -> Result<Vec<f32>, GraphRagError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut v = vec![0.0f32; 8];
        for (i, b) in text.bytes().enumerate() {
            v[i % 8] += b as f32;
        }
        Ok(v)
    }
}
