use crate::GraphRagPlugin;
use crate::error_ext::GraphRagErrorExt;
use graphrag_core::Database;
use graphrag_core::GraphRagError;
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
                let chunk_id = process_record(&db, &store_name, store.dim, &val, span)
                    .map_err(|e| e.into_labeled_error(span))?;

                Value::record(
                    nu_protocol::record! {
                        "chunk_id" => Value::int(chunk_id, span),
                        "store" => Value::string(&store_name, span),
                    },
                    span,
                )
            }
            PipelineData::Value(Value::List { vals, .. }, _) => {
                let mut results = Vec::new();
                for val in vals {
                    if let Value::Record { val: record, .. } = val {
                        let chunk_id = process_record(&db, &store_name, store.dim, &record, span)
                            .map_err(|e| e.into_labeled_error(span))?;

                        results.push(Value::record(
                            nu_protocol::record! {
                                "chunk_id" => Value::int(chunk_id, span),
                                "store" => Value::string(&store_name, span),
                            },
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

fn process_record(
    db: &Database,
    store_name: &str,
    expected_dim: usize,
    record: &Record,
    _span: nu_protocol::Span,
) -> Result<i64, GraphRagError> {
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

    // Process entities if provided
    if let Some(Value::List { vals, .. }) = record.get("entities") {
        for entity_val in vals {
            if let Value::Record {
                val: entity_record, ..
            } = entity_val
            {
                process_entity(db, store_name, chunk_id, entity_record)?;
            }
        }
    }

    Ok(chunk_id)
}

fn process_entity(
    db: &Database,
    store_name: &str,
    chunk_id: i64,
    record: &Record,
) -> Result<(), GraphRagError> {
    // Extract entity fields
    let head = record
        .get("head")
        .and_then(|v| v.as_str().ok())
        .ok_or_else(|| GraphRagError::MissingField("head".to_string()))?;

    let relation = record
        .get("relation")
        .and_then(|v| v.as_str().ok())
        .ok_or_else(|| GraphRagError::MissingField("relation".to_string()))?;

    let tail = record
        .get("tail")
        .and_then(|v| v.as_str().ok())
        .ok_or_else(|| GraphRagError::MissingField("tail".to_string()))?;

    let head_type = record.get("head_type").and_then(|v| v.as_str().ok());
    let tail_type = record.get("tail_type").and_then(|v| v.as_str().ok());

    let properties = record
        .get("properties")
        .map(|v| serde_json::to_string(v).unwrap_or_default());

    // Get or create entities
    let head_id = db.get_or_create_entity(store_name, head, head_type, None)?;
    let tail_id = db.get_or_create_entity(store_name, tail, tail_type, None)?;

    // Create relation
    db.add_relation(
        store_name,
        head_id,
        tail_id,
        relation,
        properties.as_deref(),
    )?;

    // Link entities to chunk
    db.link_chunk_entity(chunk_id, head_id)?;
    db.link_chunk_entity(chunk_id, tail_id)?;

    Ok(())
}
