mod commands;
mod error_ext;

use nu_plugin::{serve_plugin, MsgPackSerializer, Plugin, PluginCommand};

pub struct GraphRagPlugin {
    db_path: std::path::PathBuf,
    index_dir: std::path::PathBuf,
}

impl GraphRagPlugin {
    pub fn new() -> Self {
        // Check for GRAPHRAG_DATA_DIR environment variable first
        let data_dir = std::env::var("GRAPHRAG_DATA_DIR")
            .map(std::path::PathBuf::from)
            .ok()
            .unwrap_or_else(|| {
                dirs::data_local_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join("graphrag")
            });

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
            Box::new(commands::GraphRagInfo),
        ]
    }
}

fn main() {
    serve_plugin(&GraphRagPlugin::new(), MsgPackSerializer);
}
