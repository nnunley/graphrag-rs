use crate::db::Database;
use crate::hnsw::HnswIndex;
use crate::GraphRagPlugin;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, SyntaxShape, Type, Value};
use std::collections::HashSet;

pub struct GraphRagQuery;

impl PluginCommand for GraphRagQuery {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag query"
    }

    fn description(&self) -> &str {
        "Combined vector search + graph expansion query. Returns chunks and related entities."
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag query")
            .required("store", SyntaxShape::String, "Name of the store")
            .required(
                "embedding",
                SyntaxShape::List(Box::new(SyntaxShape::Number)),
                "Query embedding vector",
            )
            .named(
                "top",
                SyntaxShape::Int,
                "Number of initial vector results (default: 10)",
                Some('k'),
            )
            .named(
                "expand",
                SyntaxShape::Int,
                "Graph expansion depth (default: 1)",
                Some('e'),
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
        let embedding_val: Value = call.req(1)?;
        let top_k: usize = call.get_flag("top")?.unwrap_or(10);
        let expand_depth: usize = call.get_flag("expand")?.unwrap_or(1);
        let span = call.head;

        // Parse embedding
        let embedding: Vec<f32> = match embedding_val {
            Value::List { vals, .. } => vals
                .iter()
                .filter_map(|v| v.as_float().ok().map(|f| f as f32))
                .collect(),
            _ => {
                return Err(crate::error::GraphRagError::Other(
                    "embedding must be a list of numbers".to_string(),
                )
                .into_labeled_error(span));
            }
        };

        let db = Database::open(&plugin.db_path)
            .map_err(|e| e.into_labeled_error(span))?;

        let store = db.get_store(&store_name)
            .map_err(|e| e.into_labeled_error(span))?;

        // Load HNSW index
        let index_path = plugin.index_dir.join(format!("{}.usearch", store_name));
        let hnsw = HnswIndex::load(&index_path, store.dim)
            .map_err(|e| e.into_labeled_error(span))?;

        // Step 1: Vector search
        let vector_results = hnsw.search(&embedding, top_k)
            .map_err(|e| e.into_labeled_error(span))?;

        let initial_chunk_ids: Vec<i64> = vector_results.iter().map(|r| r.key as i64).collect();

        // Step 2: Get entities from initial chunks
        let mut all_entity_ids: HashSet<i64> = HashSet::new();
        for chunk_id in &initial_chunk_ids {
            if let Ok(entities) = db.get_entities_for_chunk(*chunk_id) {
                for entity in entities {
                    all_entity_ids.insert(entity.id);
                }
            }
        }

        // Step 3: Graph expansion - find related entities and their chunks
        let mut expanded_chunk_ids: HashSet<i64> = initial_chunk_ids.iter().cloned().collect();
        let mut frontier_entities: HashSet<i64> = all_entity_ids.clone();

        for _ in 0..expand_depth {
            let mut new_entities: HashSet<i64> = HashSet::new();

            for entity_id in &frontier_entities {
                // Get related entities via relations
                if let Ok(relations) = db.get_relations_for_entity(*entity_id) {
                    for rel in relations {
                        if !all_entity_ids.contains(&rel.head_id) {
                            new_entities.insert(rel.head_id);
                        }
                        if !all_entity_ids.contains(&rel.tail_id) {
                            new_entities.insert(rel.tail_id);
                        }
                    }
                }
            }

            // Get chunks for new entities
            for entity_id in &new_entities {
                if let Ok(chunk_ids) = db.get_chunks_for_entity(*entity_id) {
                    for chunk_id in chunk_ids {
                        expanded_chunk_ids.insert(chunk_id);
                    }
                }
            }

            all_entity_ids.extend(new_entities.iter());
            frontier_entities = new_entities;

            if frontier_entities.is_empty() {
                break;
            }
        }

        // Step 4: Fetch all chunks
        let all_chunk_ids: Vec<i64> = expanded_chunk_ids.into_iter().collect();
        let chunks = db.get_chunks_by_ids(&all_chunk_ids)
            .map_err(|e| e.into_labeled_error(span))?;

        // Step 5: Build result
        let chunk_values: Vec<Value> = chunks
            .iter()
            .map(|chunk| {
                // Find distance if this was in initial results
                let distance = vector_results
                    .iter()
                    .find(|r| r.key as i64 == chunk.id)
                    .map(|r| r.distance as f64);

                let from_graph = !initial_chunk_ids.contains(&chunk.id);

                Value::record(
                    nu_protocol::record! {
                        "chunk_id" => Value::int(chunk.id, span),
                        "content" => Value::string(&chunk.content, span),
                        "source" => chunk.source.as_ref()
                            .map(|s| Value::string(s, span))
                            .unwrap_or(Value::nothing(span)),
                        "distance" => distance
                            .map(|d| Value::float(d, span))
                            .unwrap_or(Value::nothing(span)),
                        "from_graph" => Value::bool(from_graph, span),
                    },
                    span,
                )
            })
            .collect();

        // Get entity details
        let entities = db.list_entities(&store_name)
            .map_err(|e| e.into_labeled_error(span))?;

        let entity_values: Vec<Value> = entities
            .into_iter()
            .filter(|e| all_entity_ids.contains(&e.id))
            .map(|entity| {
                Value::record(
                    nu_protocol::record! {
                        "id" => Value::int(entity.id, span),
                        "name" => Value::string(&entity.name, span),
                        "type" => entity.entity_type
                            .map(|t| Value::string(t, span))
                            .unwrap_or(Value::nothing(span)),
                    },
                    span,
                )
            })
            .collect();

        // Return combined result
        let result = Value::record(
            nu_protocol::record! {
                "chunks" => Value::list(chunk_values, span),
                "entities" => Value::list(entity_values, span),
                "stats" => Value::record(
                    nu_protocol::record! {
                        "initial_chunks" => Value::int(initial_chunk_ids.len() as i64, span),
                        "total_chunks" => Value::int(all_chunk_ids.len() as i64, span),
                        "entities_found" => Value::int(all_entity_ids.len() as i64, span),
                        "expand_depth" => Value::int(expand_depth as i64, span),
                    },
                    span,
                ),
            },
            span,
        );

        Ok(PipelineData::Value(result, None))
    }
}
