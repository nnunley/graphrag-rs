use crate::GraphRagPlugin;
use crate::error_ext::GraphRagErrorExt;
use graphrag_core::Community;
use graphrag_core::Database;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, SyntaxShape, Type, Value};

pub struct GraphRagCommunities;

impl PluginCommand for GraphRagCommunities {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag communities"
    }

    fn description(&self) -> &str {
        "Communities of the memory graph (the current fingerprinted run, built if missing)"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag communities")
            .required(
                "group",
                SyntaxShape::String,
                "Group (communities span the whole memory graph)",
            )
            .named(
                "max-iterations",
                SyntaxShape::Int,
                "Maximum Leiden iterations (default: 100)",
                Some('m'),
            )
            .named(
                "tolerance",
                SyntaxShape::Number,
                "Convergence tolerance (default: 1e-6)",
                Some('t'),
            )
            .input_output_types(vec![(Type::Nothing, Type::record())])
            .category(Category::Database)
    }

    fn run(
        &self,
        plugin: &Self::Plugin,
        _engine: &EngineInterface,
        call: &EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, nu_protocol::LabeledError> {
        let _group: String = call.req(0)?;
        let span = call.head;
        let db = Database::open(&plugin.db_path).map_err(|e| e.into_labeled_error(span))?;
        let value = super::store_communities::community_run_value(&db, false, 4, 10, span)
            .map_err(|e| e.into_labeled_error(span))?;
        Ok(PipelineData::Value(value, None))
    }
}

/// Helper to convert Community to a nested Value for hierarchical output
#[allow(dead_code)]
fn community_to_value(community: &Community, db: &Database, span: nu_protocol::Span) -> Value {
    match community {
        Community::Leaf(node_ids) => {
            let entities: Vec<Value> = node_ids
                .iter()
                .filter_map(|&id| {
                    db.get_entity_by_id(id).ok().map(|e| {
                        Value::record(
                            nu_protocol::record! {
                                "id" => Value::int(e.id, span),
                                "name" => Value::string(&e.name, span),
                                "type" => e.entity_type
                                    .map(|t| Value::string(t, span))
                                    .unwrap_or(Value::nothing(span)),
                            },
                            span,
                        )
                    })
                })
                .collect();
            Value::list(entities, span)
        }
        Community::Nested(subs) => {
            let sub_values: Vec<Value> = subs
                .iter()
                .map(|sub| community_to_value(sub, db, span))
                .collect();
            Value::list(sub_values, span)
        }
    }
}
