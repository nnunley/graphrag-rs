use graphrag_core::Database;
use crate::error_ext::GraphRagErrorExt;
use crate::GraphRagPlugin;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, SyntaxShape, Type, Value};

pub struct GraphRagGetCommunity;

impl PluginCommand for GraphRagGetCommunity {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag get-community"
    }

    fn description(&self) -> &str {
        "Get details of a specific community including entities and summary"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag get-community")
            .required("store", SyntaxShape::String, "Name of the store")
            .required("community_id", SyntaxShape::Int, "Community ID")
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
        let _store_name: String = call.req(0)?;
        let community_id: i64 = call.req(1)?;
        let span = call.head;

        let db = Database::open(&plugin.db_path)
            .map_err(|e| e.into_labeled_error(span))?;

        // Get community entities
        let entities = db.get_community_entities(community_id)
            .map_err(|e| e.into_labeled_error(span))?;

        // Get community record
        let communities = db.list_communities(&_store_name)
            .map_err(|e| e.into_labeled_error(span))?;

        let community = communities.iter().find(|c| c.id == community_id);

        let entity_values: Vec<Value> = entities
            .iter()
            .map(|e| {
                Value::record(
                    nu_protocol::record! {
                        "id" => Value::int(e.id, span),
                        "name" => Value::string(&e.name, span),
                        "type" => e.entity_type.as_ref()
                            .map(|t| Value::string(t, span))
                            .unwrap_or(Value::nothing(span)),
                    },
                    span,
                )
            })
            .collect();

        // Get child communities if any
        let children = db.get_child_communities(community_id)
            .map_err(|e| e.into_labeled_error(span))?;
        let child_ids: Vec<Value> = children.iter().map(|c| Value::int(c.id, span)).collect();

        let result = Value::record(
            nu_protocol::record! {
                "community_id" => Value::int(community_id, span),
                "level" => community.map(|c| Value::int(c.level as i64, span)).unwrap_or(Value::nothing(span)),
                "parent_id" => community.and_then(|c| c.parent_id)
                    .map(|p| Value::int(p, span))
                    .unwrap_or(Value::nothing(span)),
                "children" => Value::list(child_ids, span),
                "summary" => community.and_then(|c| c.summary.as_ref())
                    .map(|s| Value::string(s, span))
                    .unwrap_or(Value::nothing(span)),
                "modularity" => community.and_then(|c| c.modularity)
                    .map(|m| Value::float(m, span))
                    .unwrap_or(Value::nothing(span)),
                "entity_count" => Value::int(entities.len() as i64, span),
                "entities" => Value::list(entity_values, span),
            },
            span,
        );

        Ok(PipelineData::Value(result, None))
    }
}

pub struct GraphRagUpdateSummary;

impl PluginCommand for GraphRagUpdateSummary {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag update-summary"
    }

    fn description(&self) -> &str {
        "Update the summary for a community (typically from LLM output)"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag update-summary")
            .required("community_id", SyntaxShape::Int, "Community ID")
            .required("summary", SyntaxShape::String, "Summary text")
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
        let community_id: i64 = call.req(0)?;
        let summary: String = call.req(1)?;
        let span = call.head;

        let db = Database::open(&plugin.db_path)
            .map_err(|e| e.into_labeled_error(span))?;

        db.update_community_summary(community_id, &summary)
            .map_err(|e| e.into_labeled_error(span))?;

        let result = Value::record(
            nu_protocol::record! {
                "community_id" => Value::int(community_id, span),
                "summary" => Value::string(&summary, span),
                "updated" => Value::bool(true, span),
            },
            span,
        );

        Ok(PipelineData::Value(result, None))
    }
}

pub struct GraphRagListCommunities;

impl PluginCommand for GraphRagListCommunities {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag list-communities"
    }

    fn description(&self) -> &str {
        "List all stored communities for a store"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag list-communities")
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

        let communities = db.list_communities(&store_name)
            .map_err(|e| e.into_labeled_error(span))?;

        let values: Vec<Value> = communities
            .iter()
            .map(|c| {
                let entities = db.get_community_entities(c.id).unwrap_or_default();
                let entity_names: Vec<Value> = entities
                    .iter()
                    .map(|e| Value::string(&e.name, span))
                    .collect();

                Value::record(
                    nu_protocol::record! {
                        "id" => Value::int(c.id, span),
                        "level" => Value::int(c.level as i64, span),
                        "parent_id" => c.parent_id
                            .map(|p| Value::int(p, span))
                            .unwrap_or(Value::nothing(span)),
                        "size" => Value::int(entities.len() as i64, span),
                        "has_summary" => Value::bool(c.summary.is_some(), span),
                        "summary" => c.summary.as_ref()
                            .map(|s| Value::string(s, span))
                            .unwrap_or(Value::nothing(span)),
                        "entities" => Value::list(entity_names, span),
                    },
                    span,
                )
            })
            .collect();

        Ok(PipelineData::Value(Value::list(values, span), None))
    }
}
