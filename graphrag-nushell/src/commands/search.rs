use crate::GraphRagPlugin;
use crate::error_ext::GraphRagErrorExt;
use graphrag_core::Database;
use graphrag_core::HnswIndex;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, SyntaxShape, Type, Value};

pub struct GraphRagSearch;

impl PluginCommand for GraphRagSearch {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag search"
    }

    fn description(&self) -> &str {
        "Search for similar vectors in a GraphRAG store"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag search")
            .required("store", SyntaxShape::String, "Name of the store")
            .required(
                "embedding",
                SyntaxShape::List(Box::new(SyntaxShape::Number)),
                "Query embedding vector",
            )
            .named(
                "top",
                SyntaxShape::Int,
                "Number of results to return (default: 10)",
                Some('k'),
            )
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
        let embedding_val: Value = call.req(1)?;
        let top_k: usize = call.get_flag("top")?.unwrap_or(10);
        let span = call.head;

        // Parse embedding
        let embedding: Vec<f32> = match embedding_val {
            Value::List { vals, .. } => vals
                .iter()
                .filter_map(|v| v.as_float().ok().map(|f| f as f32))
                .collect(),
            _ => {
                return Err(graphrag_core::GraphRagError::Other(
                    "embedding must be a list of numbers".to_string(),
                )
                .into_labeled_error(span));
            }
        };

        let db = Database::open(&plugin.db_path).map_err(|e| e.into_labeled_error(span))?;

        // Verify store exists
        let store = db
            .get_store(&store_name)
            .map_err(|e| e.into_labeled_error(span))?;

        // Load HNSW index
        let index_path = plugin.index_dir.join(format!("{}.usearch", store_name));
        let hnsw =
            HnswIndex::load(&index_path, store.dim).map_err(|e| e.into_labeled_error(span))?;

        // Search
        let results = hnsw
            .search(&embedding, top_k)
            .map_err(|e| e.into_labeled_error(span))?;

        // Get chunk details
        let chunk_ids: Vec<i64> = results.iter().map(|r| r.key as i64).collect();
        let chunks = db
            .get_chunks_by_ids(&chunk_ids)
            .map_err(|e| e.into_labeled_error(span))?;

        // Build result table
        let values: Vec<Value> = results
            .iter()
            .filter_map(|result| {
                let chunk = chunks.iter().find(|c| c.id == result.key as i64)?;
                Some(Value::record(
                    nu_protocol::record! {
                        "chunk_id" => Value::int(chunk.id, span),
                        "distance" => Value::float(result.distance as f64, span),
                        "content" => Value::string(&chunk.content, span),
                        "source" => chunk.source.as_ref()
                            .map(|s| Value::string(s, span))
                            .unwrap_or(Value::nothing(span)),
                    },
                    span,
                ))
            })
            .collect();

        Ok(PipelineData::Value(Value::list(values, span), None))
    }
}
