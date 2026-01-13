use crate::db::Database;
use crate::leiden::CommunityGraph;
use crate::GraphRagPlugin;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, SyntaxShape, Type, Value};

pub struct GraphRagStoreCommunities;

impl PluginCommand for GraphRagStoreCommunities {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag store-communities"
    }

    fn description(&self) -> &str {
        "Detect and persist communities in the entity graph using Leiden algorithm"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag store-communities")
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
            .switch(
                "clear",
                "Clear existing communities before detecting new ones",
                Some('c'),
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
        let store_name: String = call.req(0)?;
        let max_iterations: usize = call.get_flag("max-iterations")?.unwrap_or(100);
        let tolerance: f64 = call.get_flag("tolerance")?.unwrap_or(1e-6);
        let clear: bool = call.has_flag("clear")?;
        let span = call.head;

        let db = Database::open(&plugin.db_path)
            .map_err(|e| e.into_labeled_error(span))?;

        // Optionally clear existing communities
        if clear {
            db.clear_communities(&store_name)
                .map_err(|e| e.into_labeled_error(span))?;
        }

        // Get all entities and relations
        let entities = db.list_entities(&store_name)
            .map_err(|e| e.into_labeled_error(span))?;
        let relations = db.list_relations(&store_name)
            .map_err(|e| e.into_labeled_error(span))?;

        // Build community graph
        let mut graph = CommunityGraph::new();

        // Add all entities as nodes
        for entity in &entities {
            graph.add_node(entity.id);
        }

        // Add relations as edges
        for relation in &relations {
            graph.add_edge(relation.head_id, relation.tail_id, 1.0);
        }

        // Run Leiden algorithm
        let result = graph.leiden(Some(max_iterations), tolerance);

        // Store communities
        let mut community_ids: Vec<i64> = Vec::new();

        for community in &result.communities {
            let node_ids = community.collect_nodes();

            // Create community record
            let community_id = db.create_community(&store_name, 0, result.modularity)
                .map_err(|e| e.into_labeled_error(span))?;

            community_ids.push(community_id);

            // Link entities to community
            for entity_id in node_ids {
                db.link_entity_community(entity_id, community_id)
                    .map_err(|e| e.into_labeled_error(span))?;
            }
        }

        // Build output
        let community_values: Vec<Value> = community_ids
            .iter()
            .enumerate()
            .map(|(idx, &id)| {
                let entities_in_comm = db.get_community_entities(id).unwrap_or_default();
                let entity_names: Vec<Value> = entities_in_comm
                    .iter()
                    .map(|e| Value::string(&e.name, span))
                    .collect();

                Value::record(
                    nu_protocol::record! {
                        "community_id" => Value::int(id, span),
                        "index" => Value::int(idx as i64, span),
                        "size" => Value::int(entity_names.len() as i64, span),
                        "entities" => Value::list(entity_names, span),
                    },
                    span,
                )
            })
            .collect();

        let result_record = Value::record(
            nu_protocol::record! {
                "communities" => Value::list(community_values, span),
                "total_communities" => Value::int(community_ids.len() as i64, span),
                "total_entities" => Value::int(entities.len() as i64, span),
                "total_relations" => Value::int(relations.len() as i64, span),
                "modularity" => Value::float(result.modularity, span),
            },
            span,
        );

        Ok(PipelineData::Value(result_record, None))
    }
}
