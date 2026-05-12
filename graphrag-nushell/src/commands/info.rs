use crate::GraphRagPlugin;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, Type, Value};

pub struct GraphRagInfo;

impl PluginCommand for GraphRagInfo {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag info"
    }

    fn description(&self) -> &str {
        "Show GraphRAG configuration (data directory, database path)"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag info")
            .input_output_types(vec![(Type::Nothing, Type::Record(Box::new([])))])
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

        let data_dir = plugin
            .db_path
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        let result = Value::record(
            nu_protocol::record! {
                "data_dir" => Value::string(&data_dir, span),
                "db_path" => Value::string(plugin.db_path.to_string_lossy(), span),
                "index_dir" => Value::string(plugin.index_dir.to_string_lossy(), span),
                "env_var" => Value::string("GRAPHRAG_DATA_DIR", span),
            },
            span,
        );

        Ok(PipelineData::Value(result, None))
    }
}
