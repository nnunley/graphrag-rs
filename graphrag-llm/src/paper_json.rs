//! `paper-json-v1.1`: paper-shaped JSON extraction (entities with
//! descriptions and classes; relationships with descriptions and strength),
//! enforced with llama.cpp `json_schema`, thinking on. Chosen by the
//! 2026-10-19 blind spike (arm C) with the judges' fixes applied.

use crate::ChatPrompt;
use graphrag_core::attempts::{AttemptOutcome, ExtractedEntity, ExtractedRelation};
use graphrag_core::entity_types::CANONICAL_ENTITY_TYPES;
use graphrag_core::synonyms::canonical_relations;
use serde::Deserialize;
use serde_json::{Value, json};

pub const PROFILE: &str = "paper-json";
pub const PROMPT_VERSION: &str = "paper-json-v1.1";
pub const SCHEMA_VERSION: &str = "graph-v1";
/// Thinking uses part of the budget; arm C averaged ~1.2K output tokens.
pub const MAX_TOKENS: u32 = 12000;
pub const ENABLE_THINKING: bool = true;

/// Entity types offered to the model: the canonical ones, minus graph node
/// kinds it should never produce.
fn entity_types() -> Vec<&'static str> {
    CANONICAL_ENTITY_TYPES
        .iter()
        .copied()
        .filter(|t| !matches!(*t, "group" | "section"))
        .collect()
}

fn relations() -> Vec<&'static str> {
    canonical_relations()
        .into_iter()
        .filter(|r| !matches!(*r, "same_as" | "disjoint_with" | "links_to"))
        .collect()
}

fn system_prompt(known_classes: &[String]) -> String {
    let mut s = format!(
        "You extract a knowledge graph from a text for later retrieval.

1. Identify every significant named or specific entity. For each give:
   - name: the canonical name, as it would be referred to elsewhere (not a sentence or clause)
   - type: one of {types}
   - description: a self-contained description of the entity based only on this text
   - kind: the class this entity is an instance of, as a short singular lowercase common-noun phrase (jj -> \"version control system\", llm-proxy -> \"proxy service\")
   - is_class: true when the name itself denotes a category of things rather than one specific thing; then kind is the name as a singular common noun
   - broader: the next more general class of kind (\"version control system\" -> \"software tool\"), if one is clear
2. Identify every pair of entities from step 1 that are clearly related in the text. For each give:
   - source, target: entity names exactly as in step 1
   - relation: a short snake_case verb phrase; use one of these when it fits: {rels}
   - description: why the source and target are related, based only on this text
   - strength: integer 1-10 for how strong and explicit the relationship is in the text

Rules:
- tools, version-control systems, libraries and AI models are Software, never Person.
- ignore illustrative example names (e.g. alice, bob, foo) and bibliographic citations.
- prefer a specific relation; use associated_with only when nothing else fits.
- use isa when something is an instance of a kind, genls when one kind is a more specific kind of another.
- Only use information stated in the text. If there are no entities, return empty lists.
",
        types = entity_types().join(", "),
        rels = relations().join(", "),
    );
    if !known_classes.is_empty() {
        s.push_str(&format!(
            "\nKnown classes (reuse one for kind or broader when it fits): {}\n",
            known_classes.join("; ")
        ));
    }
    s.push_str("\nReturn JSON only.");
    s
}

pub fn build_prompt(chunk: &str, heading_path: &[String], known_classes: &[String]) -> ChatPrompt {
    let context = if heading_path.is_empty() {
        String::new()
    } else {
        format!("Section: {}\n\n", heading_path.join(" › "))
    };
    ChatPrompt {
        system: system_prompt(known_classes),
        user: format!("{context}Extract entities and relationships from this text:\n\n{chunk}"),
        max_tokens: MAX_TOKENS,
    }
}

pub fn response_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "description": {"type": "string"},
                "kind": {"type": "string"},
                "is_class": {"type": "boolean"},
                "broader": {"type": "string"}},
                "required": ["name", "type", "description", "kind", "is_class"]}},
            "relationships": {"type": "array", "items": {"type": "object", "properties": {
                "source": {"type": "string"},
                "target": {"type": "string"},
                "relation": {"type": "string"},
                "description": {"type": "string"},
                "strength": {"type": "integer"}},
                "required": ["source", "target", "relation", "description", "strength"]}},
        },
        "required": ["entities", "relationships"],
    })
}

/// A name that is a sentence or clause rather than a referent: empty,
/// numeric, too long, more than 6 words, terminal punctuation, or
/// clause-shaped (5+ words starting lowercase, like "use JWT tokens for
/// session management").
pub fn is_fragment(name: &str) -> bool {
    let n = name.trim();
    let words = n.split_whitespace().count();
    n.is_empty()
        || words > 6
        || (words >= 5 && n.starts_with(|c: char| c.is_lowercase()))
        || n.ends_with(['.', '!', '?'])
        || n.chars().count() > 60
        || n.chars()
            .all(|c| c.is_ascii_digit() || c.is_whitespace() || c == '.' || c == ',')
}

#[derive(Deserialize)]
struct RawEntity {
    name: String,
    #[serde(rename = "type", default)]
    entity_type: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    is_class: bool,
    #[serde(default)]
    broader: Option<String>,
}

#[derive(Deserialize)]
struct RawRelation {
    source: String,
    target: String,
    relation: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    strength: Option<i64>,
}

#[derive(Deserialize)]
struct RawGraph {
    entities: Vec<RawEntity>,
    relationships: Vec<RawRelation>,
}

fn nonempty(s: Option<String>) -> Option<String> {
    s.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

/// Parse a response. Malformed JSON is `Err`; valid JSON with nothing left
/// after fragment rejection is `Empty`. Returns the rejected-fragment count.
pub fn parse(raw: &str) -> Result<(AttemptOutcome, usize), String> {
    let g: RawGraph = serde_json::from_str(raw.trim()).map_err(|e| format!("malformed: {e}"))?;
    let mut rejected = 0;
    let mut dropped: Vec<String> = Vec::new();
    let mut entities = Vec::new();
    for e in g.entities {
        if is_fragment(&e.name) {
            rejected += 1;
            dropped.push(e.name.trim().to_lowercase());
            continue;
        }
        entities.push(ExtractedEntity {
            name: e.name.trim().to_string(),
            entity_type: e.entity_type,
            description: e.description,
            kind: nonempty(e.kind),
            is_class: e.is_class,
            broader: nonempty(e.broader),
        });
    }
    let mut relations = Vec::new();
    for r in g.relationships {
        let bad = |n: &str| is_fragment(n) || dropped.contains(&n.trim().to_lowercase());
        if bad(&r.source) || bad(&r.target) || r.relation.trim().is_empty() {
            continue;
        }
        relations.push(ExtractedRelation {
            source: r.source.trim().to_string(),
            target: r.target.trim().to_string(),
            relation: r.relation.trim().to_string(),
            description: r.description,
            strength: r.strength.map(|s| s.clamp(1, 10) as u8),
        });
    }
    if entities.is_empty() && relations.is_empty() {
        return Ok((AttemptOutcome::Empty, rejected));
    }
    Ok((
        AttemptOutcome::Ok {
            entities,
            relations,
        },
        rejected,
    ))
}
