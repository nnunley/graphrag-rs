use crate::GraphRagPlugin;
use crate::error_ext::GraphRagErrorExt;
use graphrag_core::assertions::Asserter;
use graphrag_core::documents::{AddDocument, AddOutcome, Origin};
use graphrag_core::fingerprints::ChunkerProfile;
use graphrag_core::{Database, GraphRagError};
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Record, Signature, SyntaxShape, Type, Value};

/// Everything is added to the one memory store; groups scope it.
const MEMORY_STORE: &str = "memory";

pub struct GraphRagAdd;

impl PluginCommand for GraphRagAdd {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag add"
    }

    fn description(&self) -> &str {
        "Add content to memory. Accepts a record with: content, embedding (optional), groups (optional list), source, doc_key, entities (optional asserted triples). The background worker embeds and extracts."
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag add")
            .required(
                "group",
                SyntaxShape::String,
                "Group to add to (formerly the store name)",
            )
            .input_output_types(vec![
                (Type::record(), Type::record()),
                (
                    Type::List(Box::new(Type::record())),
                    Type::List(Box::new(Type::record())),
                ),
            ])
            .category(Category::Database)
    }

    fn run(
        &self,
        plugin: &Self::Plugin,
        _engine: &EngineInterface,
        call: &EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, nu_protocol::LabeledError> {
        let group: String = call.req(0)?;
        let span = call.head;
        let db = Database::open(&plugin.db_path).map_err(|e| e.into_labeled_error(span))?;

        let records: Vec<Record> = match input {
            PipelineData::Value(Value::Record { val, .. }, _) => vec![val.into_owned()],
            PipelineData::Value(Value::List { vals, .. }, _) => vals
                .into_iter()
                .filter_map(|v| match v {
                    Value::Record { val, .. } => Some(val.into_owned()),
                    _ => None,
                })
                .collect(),
            _ => {
                return Err(GraphRagError::Other(
                    "Expected record or list of records with: content, embedding (optional), groups (optional), source (optional)".to_string(),
                )
                .into_labeled_error(span));
            }
        };
        let mut results = Vec::new();
        for record in &records {
            let (document_id, chunks) =
                process_record(&db, &group, record).map_err(|e| e.into_labeled_error(span))?;
            let mut out = Record::new();
            out.push("document_id", Value::int(document_id, span));
            out.push("chunks", Value::int(chunks as i64, span));
            out.push("group", Value::string(&group, span));
            results.push(Value::record(out, span));
        }
        let value = if results.len() == 1 {
            results.remove(0)
        } else {
            Value::list(results, span)
        };
        Ok(PipelineData::Value(value, None))
    }
}

fn str_field<'a>(record: &'a Record, key: &str) -> Option<&'a str> {
    record.get(key).and_then(|v| v.as_str().ok())
}

fn process_record(
    db: &Database,
    group: &str,
    record: &Record,
) -> Result<(i64, usize), GraphRagError> {
    let content = str_field(record, "content")
        .ok_or_else(|| GraphRagError::MissingField("content".to_string()))?;
    let embedding: Option<Vec<f32>> = match record.get("embedding") {
        Some(Value::List { vals, .. }) => Some(
            vals.iter()
                .filter_map(|v| v.as_float().ok().map(|f| f as f32))
                .collect(),
        ),
        _ => None,
    };
    let mut groups: Vec<String> = match record.get("groups") {
        Some(Value::List { vals, .. }) => vals
            .iter()
            .filter_map(|v| v.as_str().ok().map(String::from))
            .collect(),
        _ => Vec::new(),
    };
    groups.insert(0, group.to_string());
    let groups: Vec<&str> = groups.iter().map(String::as_str).collect();

    if db.get_store(MEMORY_STORE).is_err() {
        let dim = embedding.as_ref().map_or(768, Vec::len);
        db.create_store(MEMORY_STORE, dim)?;
    }
    let outcome = db.add_document(&AddDocument {
        store: MEMORY_STORE,
        content,
        origin: Origin::Source,
        source: str_field(record, "source"),
        doc_key: str_field(record, "doc_key"),
        title: str_field(record, "title"),
        groups: &groups,
        surface: "nushell",
        profile: &ChunkerProfile::default_markdown(),
    })?;
    let document_id = match outcome {
        AddOutcome::Created { document_id }
        | AddOutcome::Existing { document_id, .. }
        | AddOutcome::Replaced { document_id, .. } => document_id,
    };
    let chunks = db.document_chunks(document_id)?;
    // A caller-supplied embedding covers the whole text, so it can stand in
    // for a single-chunk document until the worker embeds it with its own
    // embedder (the embed step stays pending, so the worker still runs).
    if let (Some(v), [only]) = (embedding, chunks.as_slice()) {
        db.set_chunk_embedding(*only, &v)?;
    }
    // Structured triples are the caller's assertions, not an extraction
    // attempt: re-extraction must not erase them.
    if let Some(Value::List { vals, .. }) = record.get("entities") {
        let by = Asserter::new("agent:nushell")?;
        for v in vals {
            let Value::Record { val, .. } = v else {
                continue;
            };
            let (Some(head), Some(rel), Some(tail)) = (
                str_field(val, "head"),
                str_field(val, "relation"),
                str_field(val, "tail"),
            ) else {
                return Err(GraphRagError::MissingField(
                    "head/relation/tail".to_string(),
                ));
            };
            let h =
                db.get_or_create_entity(MEMORY_STORE, head, str_field(val, "head_type"), None)?;
            let t =
                db.get_or_create_entity(MEMORY_STORE, tail, str_field(val, "tail_type"), None)?;
            db.assert_edge(h, rel, t, &by, Some("nushell add"), None)?;
        }
    }
    Ok((document_id, chunks.len()))
}
