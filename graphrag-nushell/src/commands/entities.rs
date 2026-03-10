use graphrag_core::Database;
use crate::error_ext::GraphRagErrorExt;
use crate::GraphRagPlugin;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, SyntaxShape, Type, Value};

pub struct GraphRagEntities;

impl PluginCommand for GraphRagEntities {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag entities"
    }

    fn description(&self) -> &str {
        "List entities in a GraphRAG store"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag entities")
            .required("store", SyntaxShape::String, "Name of the store")
            .input_output_types(vec![(Type::Nothing, Type::table())])
            .category(Category::Database)
    }

    fn run(
        &self,
        plugin: &Self::Plugin,
        _engine: &EngineInterface,
        call: &EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, nu_protocol::LabeledError> {
        let store_name: String = call.req(0)?;
        let span = call.head;

        let db = Database::open(&plugin.db_path)
            .map_err(|e| e.into_labeled_error(span))?;

        let entities = db.list_entities(&store_name)
            .map_err(|e| e.into_labeled_error(span))?;

        let values: Vec<Value> = entities
            .into_iter()
            .map(|entity| {
                Value::record(
                    nu_protocol::record! {
                        "id" => Value::int(entity.id, span),
                        "name" => Value::string(&entity.name, span),
                        "type" => entity.entity_type
                            .map(|t| Value::string(t, span))
                            .unwrap_or(Value::nothing(span)),
                        "properties" => entity.properties
                            .and_then(|p| serde_json::from_str::<serde_json::Value>(&p).ok())
                            .map(|v| json_to_value(&v, span))
                            .unwrap_or(Value::nothing(span)),
                    },
                    span,
                )
            })
            .collect();

        Ok(PipelineData::Value(Value::list(values, span), None))
    }
}

fn json_to_value(json: &serde_json::Value, span: nu_protocol::Span) -> Value {
    match json {
        serde_json::Value::Null => Value::nothing(span),
        serde_json::Value::Bool(b) => Value::bool(*b, span),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::int(i, span)
            } else if let Some(f) = n.as_f64() {
                Value::float(f, span)
            } else {
                Value::string(n.to_string(), span)
            }
        }
        serde_json::Value::String(s) => Value::string(s, span),
        serde_json::Value::Array(arr) => {
            Value::list(arr.iter().map(|v| json_to_value(v, span)).collect(), span)
        }
        serde_json::Value::Object(obj) => {
            let mut record = nu_protocol::Record::new();
            for (k, v) in obj {
                record.push(k.clone(), json_to_value(v, span));
            }
            Value::record(record, span)
        }
    }
}
