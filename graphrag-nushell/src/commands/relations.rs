use crate::GraphRagPlugin;
use crate::error_ext::GraphRagErrorExt;
use graphrag_core::Database;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, SyntaxShape, Type, Value};

pub struct GraphRagRelations;

impl PluginCommand for GraphRagRelations {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag relations"
    }

    fn description(&self) -> &str {
        "Get relations for an entity in a GraphRAG store"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag relations")
            .required("store", SyntaxShape::String, "Name of the store")
            .required("entity", SyntaxShape::String, "Name of the entity")
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
        let entity_name: String = call.req(1)?;
        let span = call.head;

        let db = Database::open(&plugin.db_path).map_err(|e| e.into_labeled_error(span))?;

        // Get entity by name
        let entity = db
            .get_entity_by_name(&store_name, &entity_name)
            .map_err(|e| e.into_labeled_error(span))?;

        // Get relations
        let relations = db
            .get_relations_for_entity(entity.id)
            .map_err(|e| e.into_labeled_error(span))?;

        // Get all entity IDs we need to look up
        let mut entity_ids: Vec<i64> = Vec::new();
        for rel in &relations {
            if !entity_ids.contains(&rel.head_id) {
                entity_ids.push(rel.head_id);
            }
            if !entity_ids.contains(&rel.tail_id) {
                entity_ids.push(rel.tail_id);
            }
        }

        // Get all entities for lookup
        let all_entities = db.list_entities(&store_name).unwrap_or_default();

        let get_entity = |id: i64| -> (String, Option<String>) {
            all_entities
                .iter()
                .find(|e| e.id == id)
                .map(|e| (e.name.clone(), e.entity_type.clone()))
                .unwrap_or_else(|| (format!("entity_{}", id), None))
        };

        let values: Vec<Value> = relations
            .into_iter()
            .map(|rel| {
                let (head_name, head_type) = get_entity(rel.head_id);
                let (tail_name, tail_type) = get_entity(rel.tail_id);

                // Determine direction relative to queried entity
                let (direction, other_entity, other_type) = if rel.head_id == entity.id {
                    ("outgoing", tail_name.clone(), tail_type.clone())
                } else {
                    ("incoming", head_name.clone(), head_type.clone())
                };

                Value::record(
                    nu_protocol::record! {
                        "relation" => Value::string(&rel.relation, span),
                        "direction" => Value::string(direction, span),
                        "head" => Value::string(&head_name, span),
                        "head_type" => head_type.map(|t| Value::string(t, span)).unwrap_or(Value::nothing(span)),
                        "tail" => Value::string(&tail_name, span),
                        "tail_type" => tail_type.map(|t| Value::string(t, span)).unwrap_or(Value::nothing(span)),
                        "other_entity" => Value::string(other_entity, span),
                        "other_type" => other_type.map(|t| Value::string(t, span)).unwrap_or(Value::nothing(span)),
                    },
                    span,
                )
            })
            .collect();

        Ok(PipelineData::Value(Value::list(values, span), None))
    }
}
