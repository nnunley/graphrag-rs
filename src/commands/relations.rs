use crate::db::Database;
use crate::GraphRagPlugin;
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

        let db = Database::open(&plugin.db_path)
            .map_err(|e| e.into_labeled_error(span))?;

        // Get entity by name
        let entity = db.get_entity_by_name(&store_name, &entity_name)
            .map_err(|e| e.into_labeled_error(span))?;

        // Get relations
        let relations = db.get_relations_for_entity(entity.id)
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

        // Get entity names (simple approach - query each)
        let get_entity_name = |id: i64| -> String {
            db.list_entities(&store_name)
                .ok()
                .and_then(|entities| entities.into_iter().find(|e| e.id == id))
                .map(|e| e.name)
                .unwrap_or_else(|| format!("entity_{}", id))
        };

        let values: Vec<Value> = relations
            .into_iter()
            .map(|rel| {
                let head_name = get_entity_name(rel.head_id);
                let tail_name = get_entity_name(rel.tail_id);

                // Determine direction relative to queried entity
                let (direction, other_entity) = if rel.head_id == entity.id {
                    ("outgoing", tail_name)
                } else {
                    ("incoming", head_name)
                };

                Value::record(
                    nu_protocol::record! {
                        "relation" => Value::string(&rel.relation, span),
                        "direction" => Value::string(direction, span),
                        "head" => Value::string(get_entity_name(rel.head_id), span),
                        "tail" => Value::string(get_entity_name(rel.tail_id), span),
                        "other_entity" => Value::string(other_entity, span),
                    },
                    span,
                )
            })
            .collect();

        Ok(PipelineData::Value(Value::list(values, span), None))
    }
}
