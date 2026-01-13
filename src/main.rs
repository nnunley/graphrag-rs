mod commands;
mod db;
mod error;
mod hnsw;
mod leiden;

use nu_plugin::{serve_plugin, MsgPackSerializer, Plugin, PluginCommand};

pub struct GraphRagPlugin {
    db_path: std::path::PathBuf,
    index_dir: std::path::PathBuf,
}

impl GraphRagPlugin {
    pub fn new() -> Self {
        let data_dir = dirs::data_local_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("graphrag");

        Self {
            db_path: data_dir.join("graphrag.db"),
            index_dir: data_dir.join("indexes"),
        }
    }
}

impl Default for GraphRagPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for GraphRagPlugin {
    fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    fn commands(&self) -> Vec<Box<dyn PluginCommand<Plugin = Self>>> {
        vec![
            Box::new(commands::GraphRagCreate),
            Box::new(commands::GraphRagList),
            Box::new(commands::GraphRagDelete),
            Box::new(commands::GraphRagAdd),
            Box::new(commands::GraphRagSearch),
            Box::new(commands::GraphRagEntities),
            Box::new(commands::GraphRagRelations),
            Box::new(commands::GraphRagQuery),
            Box::new(commands::GraphRagCommunities),
            Box::new(commands::GraphRagStoreCommunities),
            Box::new(commands::GraphRagGetCommunity),
            Box::new(commands::GraphRagUpdateSummary),
            Box::new(commands::GraphRagListCommunities),
        ]
    }
}

fn main() {
    serve_plugin(&GraphRagPlugin::new(), MsgPackSerializer);
}
