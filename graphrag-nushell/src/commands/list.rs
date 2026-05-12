use crate::GraphRagPlugin;
use crate::error_ext::GraphRagErrorExt;
use graphrag_core::Database;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, Type, Value};

pub struct GraphRagList;

impl PluginCommand for GraphRagList {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag list"
    }

    fn description(&self) -> &str {
        "List all GraphRAG stores"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag list")
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
        let span = call.head;

        let db = Database::open(&plugin.db_path).map_err(|e| e.into_labeled_error(span))?;

        let stores = db.list_stores().map_err(|e| e.into_labeled_error(span))?;

        let values: Vec<Value> = stores
            .into_iter()
            .map(|store| {
                // Get stats for each store
                let (chunks, entities, relations) =
                    db.store_stats(&store.name).unwrap_or((0, 0, 0));

                Value::record(
                    nu_protocol::record! {
                        "name" => Value::string(store.name, span),
                        "dim" => Value::int(store.dim as i64, span),
                        "chunks" => Value::int(chunks, span),
                        "entities" => Value::int(entities, span),
                        "relations" => Value::int(relations, span),
                        "created_at" => Value::string(store.created_at, span),
                    },
                    span,
                )
            })
            .collect();

        Ok(PipelineData::Value(Value::list(values, span), None))
    }
}
