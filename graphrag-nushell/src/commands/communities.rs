use crate::GraphRagPlugin;
use crate::error_ext::GraphRagErrorExt;
use graphrag_core::Database;
use graphrag_core::{Community, CommunityGraph};
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, SyntaxShape, Type, Value};

pub struct GraphRagCommunities;

impl PluginCommand for GraphRagCommunities {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag communities"
    }

    fn description(&self) -> &str {
        "Detect communities in the entity graph using the Leiden algorithm"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag communities")
            .required("store", SyntaxShape::String, "Name of the store")
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
        let max_iterations: usize = call.get_flag("max-iterations")?.unwrap_or(100);
        let tolerance: f64 = call.get_flag("tolerance")?.unwrap_or(1e-6);
        let span = call.head;

        let db = Database::open(&plugin.db_path).map_err(|e| e.into_labeled_error(span))?;

        // Get all entities and relations
        let entities = db
            .list_entities(&store_name)
            .map_err(|e| e.into_labeled_error(span))?;
        let relations = db
            .list_relations(&store_name)
            .map_err(|e| e.into_labeled_error(span))?;

        // Build community graph
        let mut graph = CommunityGraph::new();

        // Add all entities as nodes
        for entity in &entities {
            graph.add_node(entity.id);
        }

        // Add relations as edges (weight = 1.0 for now)
        for relation in &relations {
            graph.add_edge(relation.head_id, relation.tail_id, 1.0);
        }

        // Run Leiden algorithm
        let result = graph.leiden(Some(max_iterations), tolerance);

        // Build output: one row per community with entity details
        let mut community_values: Vec<Value> = Vec::new();

        for (idx, community) in result.communities.iter().enumerate() {
            let node_ids = community.collect_nodes();

            // Get entity names for this community
            let entity_names: Vec<Value> = node_ids
                .iter()
                .filter_map(|&id| {
                    db.get_entity_by_id(id)
                        .ok()
                        .map(|e| Value::string(e.name, span))
                })
                .collect();

            let entity_types: Vec<Value> = node_ids
                .iter()
                .filter_map(|&id| {
                    db.get_entity_by_id(id)
                        .ok()
                        .and_then(|e| e.entity_type.map(|t| Value::string(t, span)))
                })
                .collect();

            community_values.push(Value::record(
                nu_protocol::record! {
                    "community_id" => Value::int(idx as i64, span),
                    "size" => Value::int(node_ids.len() as i64, span),
                    "entities" => Value::list(entity_names, span),
                    "types" => Value::list(entity_types, span),
                    "entity_ids" => Value::list(
                        node_ids.iter().map(|&id| Value::int(id, span)).collect(),
                        span
                    ),
                },
                span,
            ));
        }

        // Add summary record
        let summary = Value::record(
            nu_protocol::record! {
                "total_communities" => Value::int(result.communities.len() as i64, span),
                "total_entities" => Value::int(entities.len() as i64, span),
                "total_relations" => Value::int(relations.len() as i64, span),
                "modularity" => Value::float(result.modularity, span),
            },
            span,
        );

        // Return communities table
        Ok(PipelineData::Value(
            Value::record(
                nu_protocol::record! {
                    "communities" => Value::list(community_values, span),
                    "stats" => summary,
                },
                span,
            ),
            None,
        ))
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
