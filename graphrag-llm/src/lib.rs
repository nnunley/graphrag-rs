//! Shared LLM contracts for GraphRAG metadata extraction.
//!
//! This crate deliberately keeps provider details small. MCP can satisfy the
//! prompt through sampling, while the standalone CLI can call a local Ollama
//! server directly.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// A single extracted relationship triple.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityTriple {
    pub head: String,
    pub head_type: Option<String>,
    pub relation: String,
    pub tail: String,
    pub tail_type: Option<String>,
}

/// Response shape expected from entity extraction prompts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityExtraction {
    #[serde(default)]
    pub entities: Vec<EntityTriple>,
}

/// Provider-agnostic chat prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatPrompt {
    pub system: String,
    pub user: String,
    pub max_tokens: u32,
}

/// Synchronous chat completion provider.
pub trait ChatClient {
    fn complete(&self, prompt: &ChatPrompt) -> Result<String, String>;
    /// Complete with an optional provider-native output-format constraint (e.g. Ollama `format`
    /// = a JSON schema). An `ExtractionStrategy` supplies it via `response_format()`. Default
    /// implementation ignores the constraint (instruction-only).
    fn complete_with_format(
        &self,
        prompt: &ChatPrompt,
        _format: Option<&Value>,
    ) -> Result<String, String> {
        self.complete(prompt)
    }
}

/// Build the first-pass entity extraction prompt for one chunk.
pub fn entity_extraction_prompt(
    chunk: &str,
    chunk_index: usize,
    total_chunks: usize,
) -> ChatPrompt {
    ChatPrompt {
        system: ENTITY_EXTRACTION_SYSTEM.to_string(),
        user: format!(
            "Extract entities from this text (chunk {}/{}):\n\n{}",
            chunk_index + 1,
            total_chunks,
            chunk
        ),
        max_tokens: 1000,
    }
}

/// Build the self-reflection prompt used to decide whether another extraction
/// pass is worth doing.
pub fn reflection_prompt(chunk: &str, entities: &[EntityTriple]) -> ChatPrompt {
    let entity_list = entities
        .iter()
        .map(format_triple)
        .collect::<Vec<_>>()
        .join("\n");

    ChatPrompt {
        system: "You are reviewing entity extraction results. Answer with ONLY 'yes' or 'no'."
            .to_string(),
        user: format!(
            "Given this text:\n\n{}\n\nThese entities were extracted:\n{}\n\nWere any important entities or relationships missed? Answer only 'yes' or 'no'.",
            chunk, entity_list
        ),
        max_tokens: 10,
    }
}

/// Build the prompt for extracting additional entities after reflection.
pub fn continuation_extraction_prompt(chunk: &str, already_found: &[EntityTriple]) -> ChatPrompt {
    let entity_list = already_found
        .iter()
        .flat_map(|entity| [entity.head.as_str(), entity.tail.as_str()])
        .collect::<Vec<_>>()
        .join(", ");

    ChatPrompt {
        system: CONTINUATION_EXTRACTION_SYSTEM.to_string(),
        user: format!(
            "Already extracted entities: {}\n\nExtract ADDITIONAL entities from this text:\n\n{}",
            entity_list, chunk
        ),
        max_tokens: 1000,
    }
}

/// Parse the model's entity-extraction response.
///
/// The parser accepts strict JSON and a common markdown-fenced JSON wrapper.
pub fn parse_entity_extraction(response: &str) -> Result<EntityExtraction, String> {
    let cleaned = strip_markdown_json_fence(response.trim());
    serde_json::from_str::<EntityExtraction>(cleaned)
        .map_err(|e| format!("parse entity extraction: {e}"))
}

/// Ollama `/api/chat` client that requests a JSON-schema-constrained response.
#[derive(Debug, Clone)]
pub struct OllamaChatClient {
    base_url: String,
    model: String,
}

impl OllamaChatClient {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
        }
    }

    pub fn from_env() -> Result<Self, String> {
        let base_url = std::env::var("GRAPHRAG_OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model = std::env::var("GRAPHRAG_OLLAMA_MODEL")
            .or_else(|_| std::env::var("OLLAMA_MODEL"))
            .map_err(|_| "GRAPHRAG_OLLAMA_MODEL or OLLAMA_MODEL must be set".to_string())?;
        Ok(Self::new(base_url, model))
    }

    pub fn entity_extraction_request(&self, prompt: &ChatPrompt) -> Value {
        ollama_chat_request(&self.model, prompt, Some(entity_extraction_json_schema()))
    }

    /// Build an Ollama chat request with an optional output-format constraint (the strategy's
    /// `response_format()` — `None` means instruction-only, no schema).
    pub fn chat_request(&self, prompt: &ChatPrompt, format: Option<Value>) -> Value {
        ollama_chat_request(&self.model, prompt, format)
    }

    /// The configured model name (for per-model extraction-strategy resolution).
    pub fn model(&self) -> &str {
        &self.model
    }

    fn chat_url(&self) -> String {
        format!("{}/api/chat", self.base_url.trim_end_matches('/'))
    }
}

impl ChatClient for OllamaChatClient {
    fn complete(&self, prompt: &ChatPrompt) -> Result<String, String> {
        // Back-compat: JSON-schema-constrained (existing callers rely on this).
        let schema = entity_extraction_json_schema();
        self.complete_with_format(prompt, Some(&schema))
    }

    fn complete_with_format(
        &self,
        prompt: &ChatPrompt,
        format: Option<&Value>,
    ) -> Result<String, String> {
        let response: Value = ureq::post(&self.chat_url())
            .send_json(self.chat_request(prompt, format.cloned()))
            .map_err(|e| format!("ollama chat request: {e}"))?
            .into_json()
            .map_err(|e| format!("ollama chat response: {e}"))?;

        response
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .ok_or_else(|| format!("ollama response missing message.content: {response}"))
    }
}

fn ollama_chat_request(model: &str, prompt: &ChatPrompt, format: Option<Value>) -> Value {
    let mut request = json!({
        "model": model,
        "stream": false,
        "messages": [
            {
                "role": "system",
                "content": prompt.system
            },
            {
                "role": "user",
                "content": prompt.user
            }
        ],
        "options": {
            "num_predict": prompt.max_tokens
        }
    });

    if let Some(format) = format {
        request["format"] = format;
    }

    request
}

pub fn entity_extraction_json_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["entities"],
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["head", "relation", "tail"],
                    "properties": {
                        "head": { "type": "string" },
                        "head_type": {
                            "type": ["string", "null"],
                            "enum": ["Person", "Software", "Concept", "Organization", "Location", "Other", null]
                        },
                        "relation": { "type": "string" },
                        "tail": { "type": "string" },
                        "tail_type": {
                            "type": ["string", "null"],
                            "enum": ["Person", "Software", "Concept", "Organization", "Location", "Other", null]
                        }
                    }
                }
            }
        }
    })
}

fn strip_markdown_json_fence(response: &str) -> &str {
    response
        .strip_prefix("```json")
        .or_else(|| response.strip_prefix("```"))
        .and_then(|rest| rest.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(response)
}

fn format_triple(entity: &EntityTriple) -> String {
    format!("{} -[{}]-> {}", entity.head, entity.relation, entity.tail)
}

const ENTITY_EXTRACTION_SYSTEM: &str = r#"You are an entity extraction system. Extract entities and their relationships from the given text.

Output ONLY valid JSON in this exact format (no markdown, no explanation):
{
  "entities": [
    {"head": "Entity1", "head_type": "Person|Software|Concept|Organization|Location|Other", "relation": "relationship_verb", "tail": "Entity2", "tail_type": "Person|Software|Concept|Organization|Location|Other"}
  ]
}

Rules:
- Extract concrete, specific entities (not generic terms)
- Relations should be active verbs (uses, implements, created, manages, etc.)
- If no clear entities exist, return {"entities": []}
- Output ONLY the JSON, nothing else"#;

const CONTINUATION_EXTRACTION_SYSTEM: &str = r#"You are an entity extraction system. MANY entities were missed in the previous extraction. Extract additional entities and relationships that were missed.

Output ONLY valid JSON in this exact format (no markdown, no explanation):
{
  "entities": [
    {"head": "Entity1", "head_type": "Person|Software|Concept|Organization|Location|Other", "relation": "relationship_verb", "tail": "Entity2", "tail_type": "Person|Software|Concept|Organization|Location|Other"}
  ]
}

Rules:
- Focus on entities NOT in the already-extracted list
- Extract concrete, specific entities (not generic terms)
- Relations should be active verbs (uses, implements, created, manages, etc.)
- Output ONLY the JSON, nothing else"#;

// ─── Extraction strategies (pluggable per-model call-out strategy) ──────────────────
//
// The prompt/format/parse triad differs by model (2026-08-03 spike): qwen/gemma-class do best
// with line-oriented triples; nemotron-class with JSON. An `ExtractionStrategy` encapsulates
// one such choice, orthogonal to the `ChatClient` provider. Strategies are resolved per-model
// from a data-driven table (`strategy_for_model`) — no hardcoded branches at the call sites.

/// Result of parsing one extraction response: recovered triples + a count of malformed items
/// (skipped, never failing the whole chunk — line-level tolerance for the triple format).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedExtraction {
    pub triples: Vec<EntityTriple>,
    pub malformed: usize,
}

/// A pluggable entity/relation extraction strategy: how to prompt a model for one chunk and how
/// to parse its response into triples. Selected per-model. Single-block (no sandwich) for
/// extraction — the closing "bread" is neutral-to-harmful on short chunks (2026-08-03 spike).
pub trait ExtractionStrategy: Send + Sync {
    fn name(&self) -> &'static str;
    fn build_prompt(&self, chunk: &str, chunk_index: usize, total_chunks: usize) -> ChatPrompt;
    fn parse(&self, response: &str) -> ParsedExtraction;
    /// Optional Ollama `format` constraint (JSON schema). `None` = instruction-only + parse-retry.
    fn response_format(&self) -> Option<Value> {
        None
    }
}

/// Line-oriented RDF/RDL-style extraction: one `head (Type) -[relation]-> tail (Type)` per line.
/// Best for qwen/gemma-class; degrades gracefully (a malformed line is skipped + counted, not a
/// whole-chunk loss like a broken JSON blob).
pub struct TripleLineStrategy;

/// Split a `name (Type)` fragment into `(name, Option<type>)`. A missing/empty `(Type)` → None.
fn split_typed(s: &str) -> (String, Option<String>) {
    let s = s.trim();
    // `name (type)` → strip the trailing ')', split at the last '(': name before, type after.
    match s.strip_suffix(')').and_then(|inner| inner.rfind('(').map(|i| (i, inner))) {
        Some((i, inner)) => {
            let name = inner[..i].trim().to_string();
            let ty = inner[i + 1..].trim().to_string();
            (name, if ty.is_empty() { None } else { Some(ty) })
        }
        None => (s.to_string(), None),
    }
}

/// Parse one `head (Type) -[relation]-> tail (Type)` line into a triple, or None if malformed.
fn parse_triple_line(line: &str) -> Option<EntityTriple> {
    let (lhs, tail_part) = line.split_once("]->")?;
    let (head_part, relation) = lhs.split_once("-[")?;
    let (head, head_type) = split_typed(head_part);
    let (tail, tail_type) = split_typed(tail_part);
    if head.is_empty() || tail.is_empty() || relation.trim().is_empty() {
        return None;
    }
    Some(EntityTriple { head, head_type, relation: relation.trim().to_string(), tail, tail_type })
}

const TRIPLE_LINE_SYSTEM: &str = r#"You are an entity extraction system. Extract entities and their relationships from the text.
Entity types MUST be one of: Person | Software | Concept | Organization | Location | Other.

Output ONE triple per line and NOTHING else, in exactly this format:
head (head_type) -[relation]-> tail (tail_type)

Example:
tokio (Software) -[chosen_over]-> async-std (Software)

Rules:
- Extract concrete, specific entities (not generic terms)
- Relations should be active verbs (uses, implements, created, manages, ...)
- If no clear entities exist, output nothing"#;

impl ExtractionStrategy for TripleLineStrategy {
    fn name(&self) -> &'static str {
        "triple-lines"
    }
    fn build_prompt(&self, chunk: &str, chunk_index: usize, total_chunks: usize) -> ChatPrompt {
        ChatPrompt {
            system: TRIPLE_LINE_SYSTEM.to_string(),
            user: format!(
                "Extract entities and relationships from this text (chunk {}/{}):\n\n{}",
                chunk_index + 1,
                total_chunks,
                chunk
            ),
            max_tokens: 1000,
        }
    }
    fn parse(&self, response: &str) -> ParsedExtraction {
        let mut triples = Vec::new();
        let mut malformed = 0usize;
        for line in response.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('#') || l.starts_with("//") || l.starts_with("Example") {
                continue;
            }
            match parse_triple_line(l) {
                Some(t) => triples.push(t),
                None => malformed += 1,
            }
        }
        ParsedExtraction { triples, malformed }
    }
}

/// JSON-schema-constrained extraction — today's behavior behind the trait. Best for
/// nemotron-class (its triple-line output is noisier). A broken JSON blob loses the whole chunk.
pub struct JsonStrategy;

impl ExtractionStrategy for JsonStrategy {
    fn name(&self) -> &'static str {
        "json"
    }
    fn build_prompt(&self, chunk: &str, chunk_index: usize, total_chunks: usize) -> ChatPrompt {
        entity_extraction_prompt(chunk, chunk_index, total_chunks)
    }
    fn parse(&self, response: &str) -> ParsedExtraction {
        match parse_entity_extraction(response) {
            Ok(e) => ParsedExtraction { triples: e.entities, malformed: 0 },
            Err(_) => ParsedExtraction { triples: Vec::new(), malformed: 1 },
        }
    }
    fn response_format(&self) -> Option<Value> {
        Some(entity_extraction_json_schema())
    }
}

/// A per-model → extraction-strategy rule. Data-driven: add a row to extend; no hardcoded
/// strategy branches at the call sites.
struct StrategyRule {
    model_substr: &'static str,
    make: fn() -> Box<dyn ExtractionStrategy>,
}

/// Model→strategy rules (2026-08-03 extraction spike). First case-insensitive substring match
/// wins; the default (triple-lines) is validated best for qwen/gemma-class models.
static MODEL_STRATEGY_RULES: &[StrategyRule] =
    &[StrategyRule { model_substr: "nemotron", make: || Box::new(JsonStrategy) }];

/// Resolve a model name to its best-performing extraction strategy.
pub fn strategy_for_model(model: &str) -> Box<dyn ExtractionStrategy> {
    let lower = model.to_lowercase();
    for rule in MODEL_STRATEGY_RULES {
        if lower.contains(rule.model_substr) {
            return (rule.make)();
        }
    }
    Box::new(TripleLineStrategy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_entity_extraction_json() {
        let parsed = parse_entity_extraction(
            r#"{"entities":[{"head":"Alice","head_type":"Person","relation":"uses","tail":"GraphRAG","tail_type":"Software"}]}"#,
        )
        .expect("valid extraction parses");

        assert_eq!(parsed.entities.len(), 1);
        assert_eq!(parsed.entities[0].head, "Alice");
        assert_eq!(parsed.entities[0].tail_type.as_deref(), Some("Software"));
    }

    #[test]
    fn parses_markdown_fenced_json() {
        let parsed = parse_entity_extraction(
            "```json\n{\"entities\":[{\"head\":\"Alice\",\"relation\":\"uses\",\"tail\":\"GraphRAG\"}]}\n```",
        )
        .expect("fenced extraction parses");

        assert_eq!(parsed.entities[0].relation, "uses");
    }

    #[test]
    fn extraction_prompt_mentions_chunk_position() {
        let prompt = entity_extraction_prompt("hello", 1, 3);

        assert!(prompt.user.contains("chunk 2/3"));
        assert!(prompt.system.contains("Output ONLY valid JSON"));
    }

    #[test]
    fn ollama_request_forces_json_schema_response() {
        let client = OllamaChatClient::new("http://localhost:11434", "llama3.2");
        let prompt = entity_extraction_prompt("Alice uses GraphRAG.", 0, 1);
        let request = client.entity_extraction_request(&prompt);

        assert_eq!(request["model"], "llama3.2");
        assert_eq!(request["stream"], false);
        assert_eq!(request["format"]["type"], "object");
        assert_eq!(request["format"]["required"], json!(["entities"]));
        assert_eq!(request["format"]["properties"]["entities"]["type"], "array");
        assert_eq!(request["messages"][0]["role"], "system");
        assert_eq!(request["messages"][1]["role"], "user");
        assert_eq!(request["options"]["num_predict"], prompt.max_tokens);
    }

    #[test]
    fn chat_request_omits_format_when_none() {
        let client = OllamaChatClient::new("http://localhost:11434", "qwen3.6-hermes");
        let prompt = entity_extraction_prompt("x", 0, 1);
        let req = client.chat_request(&prompt, None);
        assert!(req.get("format").is_none(), "no format constraint when strategy supplies none");
        let req2 = client.chat_request(&prompt, Some(entity_extraction_json_schema()));
        assert_eq!(req2["format"]["type"], "object", "format applied when supplied");
    }

    #[test]
    fn strategy_for_model_resolves_per_model() {
        // 2026-08-03 spike: nemotron-class → JSON; qwen/gemma/default → triple-lines.
        assert_eq!(strategy_for_model("nemotron3:33b").name(), "json");
        assert_eq!(strategy_for_model("Nemotron-4-large").name(), "json", "case-insensitive");
        assert_eq!(strategy_for_model("qwen3.6-hermes:latest").name(), "triple-lines");
        assert_eq!(strategy_for_model("gemma4:31b").name(), "triple-lines");
        assert_eq!(strategy_for_model("some-unknown-model").name(), "triple-lines", "default");
    }

    #[test]
    fn json_strategy_wraps_existing_json_behavior() {
        let s = JsonStrategy;
        assert_eq!(s.name(), "json");
        let p = s.build_prompt("Alice uses GraphRAG.", 0, 1);
        assert!(p.system.contains("Output ONLY valid JSON"));
        assert!(p.user.contains("Alice uses GraphRAG."));
        let parsed =
            s.parse(r#"{"entities":[{"head":"Alice","relation":"uses","tail":"GraphRAG"}]}"#);
        assert_eq!(parsed.triples.len(), 1);
        assert_eq!(parsed.triples[0].head, "Alice");
        assert_eq!(parsed.malformed, 0);
        assert!(s.response_format().is_some(), "JSON strategy constrains via response_format");
    }

    #[test]
    fn json_strategy_counts_unparseable_as_malformed() {
        let s = JsonStrategy;
        let parsed = s.parse("not json at all");
        assert_eq!(parsed.triples.len(), 0);
        assert_eq!(parsed.malformed, 1, "a broken JSON blob loses the whole chunk (malformed=1)");
    }

    #[test]
    fn triple_line_strategy_builds_single_block_prompt() {
        let s = TripleLineStrategy;
        let p = s.build_prompt("Norman chose tokio.", 1, 3);
        let full = format!("{}\n{}", p.system, p.user);
        assert!(full.contains("-["), "teaches the head -[relation]-> tail format");
        assert!(
            full.contains("Software") && full.contains("Person"),
            "lists canonical entity types"
        );
        assert!(full.to_lowercase().contains("example"), "has a worked example");
        assert!(p.user.contains("Norman chose tokio."), "includes the chunk text");
        // Single-block: the core format instruction appears exactly once (no sandwich for extract).
        assert_eq!(
            full.matches("ONE triple per line").count(),
            1,
            "single-block (no sandwich) for extraction"
        );
    }

    #[test]
    fn triple_line_strategy_parses_and_counts_malformed() {
        let s = TripleLineStrategy;
        let out = "tokio (Software) -[chosen_over]-> async-std (Software)\n\
                   not a triple line\n\
                   Alice (Person) -[uses]-> GraphRAG (Software)";
        let p = s.parse(out);
        assert_eq!(p.triples.len(), 2);
        assert_eq!(p.triples[0].head, "tokio");
        assert_eq!(p.triples[0].relation, "chosen_over");
        assert_eq!(p.triples[0].head_type.as_deref(), Some("Software"));
        assert_eq!(p.triples[0].tail_type.as_deref(), Some("Software"));
        assert_eq!(p.triples[1].head, "Alice");
        assert_eq!(p.malformed, 1);
    }
}
