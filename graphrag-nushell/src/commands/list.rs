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
        "List the groups in memory, with document counts"
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

        let groups = db
            .list_groups("memory")
            .map_err(|e| e.into_labeled_error(span))?;
        let values: Vec<Value> = groups
            .into_iter()
            .map(|(name, documents)| {
                Value::record(
                    nu_protocol::record! {
                        "group" => Value::string(name, span),
                        "documents" => Value::int(documents as i64, span),
                    },
                    span,
                )
            })
            .collect();

        Ok(PipelineData::Value(Value::list(values, span), None))
    }
}
