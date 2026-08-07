use crate::GraphRagPlugin;
use crate::error_ext::GraphRagErrorExt;
use graphrag_core::Database;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, SyntaxShape, Type, Value};

pub struct GraphRagCreate;

impl PluginCommand for GraphRagCreate {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag create"
    }

    fn description(&self) -> &str {
        "Create a new GraphRAG store"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag create")
            .required("name", SyntaxShape::String, "Name of the store")
            .named(
                "dim",
                SyntaxShape::Int,
                "Embedding dimension (default: 1536 for OpenAI)",
                Some('d'),
            )
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
        let name: String = call.req(0)?;
        let dim: usize = call.get_flag("dim")?.unwrap_or(1536);
        let span = call.head;

        // Create database and store
        let db = Database::open(&plugin.db_path).map_err(|e| e.into_labeled_error(span))?;

        let store = db
            .create_store(&name, dim)
            .map_err(|e| e.into_labeled_error(span))?;

        // Return store info as record
        let record = Value::record(
            nu_protocol::record! {
                "name" => Value::string(store.name, span),
                "dim" => Value::int(store.dim as i64, span),
                "created_at" => Value::string(store.created_at, span),
            },
            span,
        );

        Ok(PipelineData::Value(record, None))
    }
}
