use crate::GraphRagPlugin;
use crate::error_ext::GraphRagErrorExt;
use graphrag_core::{
    Database, EntityInput, ExtractResult, GraphRagError, apply_extracted_entities,
    extraction_provenance_nonce,
};
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Record, Signature, SyntaxShape, Type, Value};

pub struct GraphRagAdd;

impl PluginCommand for GraphRagAdd {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag add"
    }

    fn description(&self) -> &str {
        "Add content to a GraphRAG store. Accepts a record with: content, embedding, entities (optional), source (optional)"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag add")
            .required("store", SyntaxShape::String, "Name of the store")
            .input_output_types(vec![
                (Type::Record(Box::new([])), Type::Record(Box::new([]))),
                (
                    Type::List(Box::new(Type::Record(Box::new([])))),
                    Type::List(Box::new(Type::Record(Box::new([])))),
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
        let store_name: String = call.req(0)?;
        let span = call.head;

        let db = Database::open(&plugin.db_path).map_err(|e| e.into_labeled_error(span))?;

        // Verify store exists and get dimension
        let store = db
            .get_store(&store_name)
            .map_err(|e| e.into_labeled_error(span))?;

        // Process input
        let result = match input {
            PipelineData::Value(Value::Record { val, .. }, _) => {
                let (chunk_id, provenance_nonce) =
                    process_record(&db, &store_name, store.dim, &val, span)
                        .map_err(|e| e.into_labeled_error(span))?;
                add_result_value(&store_name, chunk_id, provenance_nonce, span)
            }
            PipelineData::Value(Value::List { vals, .. }, _) => {
                let mut results = Vec::new();
                for val in vals {
                    if let Value::Record { val: record, .. } = val {
                        let (chunk_id, provenance_nonce) =
                            process_record(&db, &store_name, store.dim, &record, span)
                                .map_err(|e| e.into_labeled_error(span))?;
                        results.push(add_result_value(
                            &store_name,
                            chunk_id,
                            provenance_nonce,
                            span,
                        ));
                    }
                }
                Value::list(results, span)
            }
            _ => {
                return Err(GraphRagError::Other(
                    "Expected record or list of records with: content, embedding, entities (optional), source (optional)".to_string()
                ).into_labeled_error(span));
            }
        };

        Ok(PipelineData::Value(result, None))
    }
}

fn add_result_value(
    store_name: &str,
    chunk_id: i64,
    provenance_nonce: Option<String>,
    span: nu_protocol::Span,
) -> Value {
    let mut record = Record::new();
    record.push("chunk_id", Value::int(chunk_id, span));
    record.push("store", Value::string(store_name, span));
    if let Some(nonce) = provenance_nonce {
        record.push("provenance_nonce", Value::string(nonce, span));
    }
    Value::record(record, span)
}

fn process_record(
    db: &Database,
    store_name: &str,
    expected_dim: usize,
    record: &Record,
    _span: nu_protocol::Span,
) -> Result<(i64, Option<String>), GraphRagError> {
    // Extract content (required)
    let content = record
        .get("content")
        .and_then(|v| v.as_str().ok())
        .ok_or_else(|| GraphRagError::MissingField("content".to_string()))?;

    // Extract embedding (required)
    let embedding: Vec<f32> = record
        .get("embedding")
        .and_then(|v| {
            if let Value::List { vals, .. } = v {
                Some(
                    vals.iter()
                        .filter_map(|v| v.as_float().ok().map(|f| f as f32))
                        .collect(),
                )
            } else {
                None
            }
        })
        .ok_or_else(|| GraphRagError::MissingField("embedding".to_string()))?;

    // Verify dimension
    if embedding.len() != expected_dim {
        return Err(GraphRagError::DimensionMismatch {
            expected: expected_dim,
            got: embedding.len(),
        });
    }

    // Extract optional fields
    let source = record.get("source").and_then(|v| v.as_str().ok());

    let metadata = record
        .get("metadata")
        .map(|v| serde_json::to_string(v).unwrap_or_default());

    // Add chunk to database
    let chunk_id = db.add_chunk(store_name, content, source, metadata.as_deref())?;

    // Store the canonical embedding
    db.set_chunk_embedding(chunk_id, &embedding)?;

    // Structured entity input is an extraction result and therefore must use
    // the same atomic checkpoint/provenance path as CLI and MCP extraction.
    let provenance_nonce = if let Some(Value::List { vals, .. }) = record.get("entities") {
        let triples = vals
            .iter()
            .filter_map(|value| match value {
                Value::Record { val, .. } => Some(parse_entity(val)),
                _ => None,
            })
            .collect::<Result<Vec<_>, _>>()?;
        let model = record
            .get("model")
            .and_then(|value| value.as_str().ok())
            .unwrap_or("nushell-input");
        let nonce = extraction_provenance_nonce(store_name, chunk_id, model, content);
        let result = ExtractResult {
            chunk_id,
            model: model.into(),
            provenance_nonce: nonce.clone(),
            response: serde_json::to_string(&triples).unwrap_or_default(),
        };
        apply_extracted_entities(db, store_name, &result, &triples)?;
        Some(nonce)
    } else {
        None
    };

    Ok((chunk_id, provenance_nonce))
}

fn parse_entity(record: &Record) -> Result<EntityInput, GraphRagError> {
    let head = record
        .get("head")
        .and_then(|value| value.as_str().ok())
        .ok_or_else(|| GraphRagError::MissingField("head".to_string()))?;
    let relation = record
        .get("relation")
        .and_then(|value| value.as_str().ok())
        .ok_or_else(|| GraphRagError::MissingField("relation".to_string()))?;
    let tail = record
        .get("tail")
        .and_then(|value| value.as_str().ok())
        .ok_or_else(|| GraphRagError::MissingField("tail".to_string()))?;
    Ok(EntityInput {
        head: head.into(),
        head_type: record
            .get("head_type")
            .and_then(|value| value.as_str().ok())
            .map(str::to_owned),
        relation: relation.into(),
        tail: tail.into(),
        tail_type: record
            .get("tail_type")
            .and_then(|value| value.as_str().ok())
            .map(str::to_owned),
        properties: record
            .get("properties")
            .and_then(|value| serde_json::to_value(value).ok()),
    })
}
