//! Per-step input fingerprints. Each step hashes its inputs under its own
//! domain tag, so equal bytes fed to two steps never collide. A step is
//! current only when its stored fingerprint equals the one computed here
//! from its current inputs.

use crate::paper::{
    ExtractionAttemptIdentityParts, ExtractionAttemptIdentityV1, PaperContractError,
};
use crate::token::fingerprint_parts;
use serde::{Deserialize, Serialize};

/// How a document is cut into chunks. Part of the chunk-plan fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkerProfile {
    /// `"markdown-sections"` or `"code"`.
    pub kind: &'static str,
    /// Maximum chunk length in characters (also the hard cap).
    pub size: usize,
    /// Characters shared by consecutive chunks of one long region.
    pub overlap: usize,
}

impl ChunkerProfile {
    pub fn default_markdown() -> Self {
        Self {
            kind: "markdown-sections",
            size: 2000,
            overlap: 200,
        }
    }
}

/// Inputs to a community run besides the graph itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommunityConfig {
    /// Weight per active non-structural assertion (e.g. `part_of`).
    pub asserted_weight: f64,
    /// Weight per derived `mentions` edge.
    pub mention_weight: f64,
    /// Weight per structural `has_part` edge.
    pub structure_weight: f64,
    /// Weight for `isa`/`genls` and admitted lifted class edges.
    pub class_weight: f64,
    pub max_levels: i32,
    pub min_size_to_split: usize,
    pub seed: u64,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            asserted_weight: 0.0,
            mention_weight: 0.0,
            structure_weight: 0.0,
            class_weight: 0.0,
            max_levels: 4,
            min_size_to_split: 10,
            seed: 42,
        }
    }
}

/// The extraction settings that, with a chunk, identify an attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionConfig {
    pub model: String,
    pub profile: String,
    pub prompt_version: String,
    pub schema_version: String,
    pub glean_max_rounds: u32,
}

/// Everything a community summary is written from.
#[derive(Debug, Clone, PartialEq)]
pub struct SummaryInput {
    /// `(name, type, description)` per member entity.
    pub members: Vec<(String, String, String)>,
    /// `(head, tail, relation, weight, description)` per internal edge.
    pub edges: Vec<(String, String, String, f64, String)>,
    /// Fingerprints of child communities' summaries.
    pub child_fps: Vec<String>,
    /// Groups the grounding excerpts come from.
    pub groups: Vec<String>,
}

/// Identity of a document's text (`documents.content_fp`).
pub fn document_fp(content: &str) -> String {
    fingerprint_parts(&[b"graphrag-step-document-v1", content.as_bytes()])
}

pub fn chunk_plan_fp(document_content: &str, chunker: &ChunkerProfile) -> String {
    fingerprint_parts(&[
        b"graphrag-step-chunk-v1",
        document_content.as_bytes(),
        chunker.kind.as_bytes(),
        &(chunker.size as u64).to_be_bytes(),
        &(chunker.overlap as u64).to_be_bytes(),
    ])
}

pub fn embed_fp(chunk_text: &str, embedder_id: &str, dim: usize) -> String {
    fingerprint_parts(&[
        b"graphrag-step-embed-v1",
        chunk_text.as_bytes(),
        embedder_id.as_bytes(),
        &(dim as u64).to_be_bytes(),
    ])
}

/// The extraction step reuses the M0 attempt identity unchanged.
pub fn extract_fp(
    parts: &ExtractionAttemptIdentityParts<'_>,
) -> Result<String, PaperContractError> {
    Ok(ExtractionAttemptIdentityV1::derive(parts)?.fingerprint)
}

/// Fingerprint of the weighted graph a community run clusters. Edge and
/// isolate order don't matter; weights are hashed bit-exactly.
pub fn graph_fp(
    edges: &[(i64, i64, f64)],
    isolates: &[i64],
    config: &CommunityConfig,
    seed: u64,
) -> String {
    let mut edges = edges.to_vec();
    edges.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)).then(a.2.total_cmp(&b.2)));
    let mut isolates = isolates.to_vec();
    isolates.sort_unstable();
    let mut buf = Vec::with_capacity(edges.len() * 24 + isolates.len() * 8);
    for (a, b, w) in &edges {
        buf.extend_from_slice(&a.to_be_bytes());
        buf.extend_from_slice(&b.to_be_bytes());
        buf.extend_from_slice(&w.to_bits().to_be_bytes());
    }
    let mut iso = Vec::with_capacity(isolates.len() * 8);
    for n in &isolates {
        iso.extend_from_slice(&n.to_be_bytes());
    }
    let cfg = [
        config.asserted_weight.to_bits(),
        config.mention_weight.to_bits(),
        config.structure_weight.to_bits(),
        config.class_weight.to_bits(),
        config.max_levels as i64 as u64,
        config.min_size_to_split as u64,
        config.seed,
    ];
    let cfg: Vec<u8> = cfg.iter().flat_map(|v| v.to_be_bytes()).collect();
    fingerprint_parts(&[
        b"graphrag-step-communities-v1",
        &buf,
        &iso,
        &cfg,
        &seed.to_be_bytes(),
    ])
}

pub fn summary_fp(input: &SummaryInput, model: &str, prompt_version: &str) -> String {
    let mut parts: Vec<Vec<u8>> = vec![b"graphrag-step-summary-v1".to_vec()];
    for (n, t, d) in &input.members {
        parts.extend([n, t, d].map(|s| s.as_bytes().to_vec()));
    }
    parts.push(b"|edges".to_vec());
    for (h, t, r, w, d) in &input.edges {
        parts.extend([h, t, r].map(|s| s.as_bytes().to_vec()));
        parts.push(w.to_bits().to_be_bytes().to_vec());
        parts.push(d.as_bytes().to_vec());
    }
    parts.push(b"|children".to_vec());
    parts.extend(input.child_fps.iter().map(|s| s.as_bytes().to_vec()));
    parts.push(b"|groups".to_vec());
    parts.extend(input.groups.iter().map(|s| s.as_bytes().to_vec()));
    parts.push(model.as_bytes().to_vec());
    parts.push(prompt_version.as_bytes().to_vec());
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    fingerprint_parts(&refs)
}
