use crate::db::Database;
use crate::GraphRagPlugin;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, SyntaxShape, Type, Value};

pub struct GraphRagDelete;

impl PluginCommand for GraphRagDelete {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag delete"
    }

    fn description(&self) -> &str {
        "Delete a GraphRAG store"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag delete")
            .required("name", SyntaxShape::String, "Name of the store to delete")
            .input_output_types(vec![(Type::Nothing, Type::Nothing)])
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
        let span = call.head;

        let db = Database::open(&plugin.db_path)
            .map_err(|e| e.into_labeled_error(span))?;

        db.delete_store(&name)
            .map_err(|e| e.into_labeled_error(span))?;

        // Delete HNSW index file if it exists
        let index_path = plugin.index_dir.join(format!("{}.usearch", name));
        if index_path.exists() {
            std::fs::remove_file(&index_path)
                .map_err(|e| crate::error::GraphRagError::Io(e).into_labeled_error(span))?;
        }

        Ok(PipelineData::Value(Value::nothing(span), None))
    }
}
