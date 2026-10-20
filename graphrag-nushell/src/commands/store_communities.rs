use crate::GraphRagPlugin;
use crate::error_ext::GraphRagErrorExt;
use graphrag_core::Database;
use graphrag_core::fingerprints::CommunityConfig;
use nu_plugin::{EngineInterface, EvaluatedCall, PluginCommand};
use nu_protocol::{Category, PipelineData, Signature, SyntaxShape, Type, Value};

const MEMORY_STORE: &str = "memory";

pub struct GraphRagStoreCommunities;

impl PluginCommand for GraphRagStoreCommunities {
    type Plugin = GraphRagPlugin;

    fn name(&self) -> &str {
        "graphrag store-communities"
    }

    fn description(&self) -> &str {
        "Build (or reuse) the fingerprinted, evidence-weighted community run over the memory graph"
    }

    fn signature(&self) -> Signature {
        Signature::build("graphrag store-communities")
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
            .named(
                "levels",
                SyntaxShape::Int,
                "Maximum hierarchy levels for hierarchical detection (default: 1 = flat)",
                Some('l'),
            )
            .named(
                "min-size",
                SyntaxShape::Int,
                "Minimum community size to partition further (default: 5)",
                Some('s'),
            )
            .switch(
                "clear",
                "Clear existing communities before detecting new ones",
                Some('c'),
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
        let max_levels: i32 = call.get_flag("levels")?.unwrap_or(4);
        let min_size: usize = call.get_flag("min-size")?.unwrap_or(10);
        let clear: bool = call.has_flag("clear")?;
        let span = call.head;
        let db = Database::open(&plugin.db_path).map_err(|e| e.into_labeled_error(span))?;
        let value = community_run_value(&db, clear, max_levels, min_size, span)
            .map_err(|e| e.into_labeled_error(span))?;
        Ok(PipelineData::Value(value, None))
    }
}

/// Build (or reuse) the fingerprinted community run over the whole memory
/// graph and describe it. Shared by `graphrag communities`.
pub(crate) fn community_run_value(
    db: &Database,
    clear: bool,
    max_levels: i32,
    min_size: usize,
    span: nu_protocol::Span,
) -> Result<Value, graphrag_core::GraphRagError> {
    if clear {
        db.clear_communities(MEMORY_STORE)?;
    }
    let cfg = CommunityConfig {
        max_levels: max_levels.max(1),
        min_size_to_split: min_size.max(1),
        ..CommunityConfig::default()
    };
    let run = db.build_community_run(MEMORY_STORE, &cfg)?;
    let communities = db.list_communities(MEMORY_STORE)?;
    let values: Vec<Value> = communities
        .iter()
        .map(|c| {
            let names: Vec<Value> = db
                .get_community_entities(c.id)
                .unwrap_or_default()
                .iter()
                .map(|e| Value::string(&e.name, span))
                .collect();
            Value::record(
                nu_protocol::record! {
                    "community_id" => Value::int(c.id, span),
                    "level" => Value::int(c.level as i64, span),
                    "parent_id" => c.parent_id.map(|p| Value::int(p, span)).unwrap_or(Value::nothing(span)),
                    "size" => Value::int(names.len() as i64, span),
                    "summary" => c.summary.clone().map(|t| Value::string(t, span)).unwrap_or(Value::nothing(span)),
                    "entities" => Value::list(names, span),
                },
                span,
            )
        })
        .collect();
    Ok(Value::record(
        nu_protocol::record! {
            "run_id" => Value::int(run.run_id, span),
            "reused" => Value::bool(run.reused, span),
            "communities" => Value::list(values, span),
            "total_communities" => Value::int(run.communities as i64, span),
            "depth" => Value::int(run.levels as i64, span),
            "modularity" => Value::float(run.modularity, span),
        },
        span,
    ))
}
