//! The self-driving ingestion worker.
//!
//! `run` drains ready work in priority order until nothing is left: embed and
//! extract chunks, re-chunk documents whose chunker changed, generalize,
//! cluster, summarize. "Ready" is computed from step fingerprints in core;
//! there is no queue. One process at a time (an advisory lock), LLM calls
//! run concurrently on scoped threads, and every database write happens on
//! the calling thread.

pub mod lock;
pub mod spawn;
mod steps;

use graphrag_core::fingerprints::{ChunkerProfile, CommunityConfig, ExtractionConfig};
use graphrag_core::generalize::LiftConfig;
use graphrag_core::salience::SalienceConfig;
use graphrag_core::status::StaticConfig;
use graphrag_core::text_embedder::TextEmbedder;
use graphrag_llm::ChatClient;
use graphrag_llm::openai::OpenAiChatClient;
use graphrag_llm::paper_json;
use std::path::{Path, PathBuf};

pub use spawn::spawn_detached;

/// An LLM the worker can pin: chat plus the list of served models.
pub trait WorkerLlm: ChatClient {
    fn served_models(&self) -> Result<Vec<String>, String>;
}

impl WorkerLlm for OpenAiChatClient {
    fn served_models(&self) -> Result<Vec<String>, String> {
        OpenAiChatClient::served_models(self).map_err(|e| e.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub data_dir: PathBuf,
    pub store: String,
    pub model: String,
    pub prompt_version: String,
    pub concurrency: usize,
    pub max_items: Option<usize>,
    /// Reserved: scope work to one group (all groups today).
    pub group: Option<String>,
    pub retry_failed: bool,
    pub accept_reextract: bool,
    /// Consecutive transport failures before giving up with `Outage`.
    pub outage_threshold: u32,
    pub chunker: ChunkerProfile,
    pub community: CommunityConfig,
    pub lift: LiftConfig,
    pub salience: SalienceConfig,
    /// Classes listed in the extraction prompt.
    pub known_classes: usize,
    /// Cosine at which a new class name maps to an existing class.
    pub class_min_cos: f32,
}

impl WorkerConfig {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            store: "memory".into(),
            model: graphrag_llm::openai::DEFAULT_MODEL.into(),
            prompt_version: paper_json::PROMPT_VERSION.into(),
            concurrency: 4,
            max_items: None,
            group: None,
            retry_failed: false,
            accept_reextract: false,
            outage_threshold: 5,
            chunker: ChunkerProfile::default_markdown(),
            community: CommunityConfig::default(),
            lift: LiftConfig::default(),
            salience: SalienceConfig::default(),
            known_classes: 30,
            class_min_cos: 0.92,
        }
    }

    pub fn extraction(&self) -> ExtractionConfig {
        ExtractionConfig {
            model: self.model.clone(),
            profile: paper_json::PROFILE.into(),
            prompt_version: self.prompt_version.clone(),
            schema_version: paper_json::SCHEMA_VERSION.into(),
            glean_max_rounds: 0,
        }
    }

    /// The fingerprint expectations this configuration implies.
    pub fn static_config(&self, embedder: &dyn TextEmbedder) -> StaticConfig {
        StaticConfig {
            store: self.store.clone(),
            embedder_id: embedder.id(),
            dim: embedder.dim(),
            chunker: self.chunker.clone(),
            extraction: self.extraction(),
            community: self.community.clone(),
            lift: self.lift.clone(),
        }
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("graphrag.db")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitStatus {
    Done,
    AlreadyRunning,
    Usage,
    Unavailable,
    Outage,
    /// An unexpected local error (database, I/O).
    Error(String),
}

impl ExitStatus {
    pub fn code(&self) -> i32 {
        match self {
            ExitStatus::Done | ExitStatus::AlreadyRunning => 0,
            ExitStatus::Usage => 64,
            ExitStatus::Unavailable => 69,
            ExitStatus::Error(_) => 70,
            ExitStatus::Outage => 75,
        }
    }
}

/// While this file exists the worker does nothing (e.g. during big builds).
pub fn pause_flag(data_dir: &Path) -> PathBuf {
    data_dir.join("worker.pause")
}

pub fn run(cfg: &WorkerConfig, llm: &dyn WorkerLlm, embedder: &dyn TextEmbedder) -> ExitStatus {
    let _guard = match lock::acquire(&cfg.data_dir) {
        Ok(Some(g)) => g,
        Ok(None) => {
            println!("already running");
            return ExitStatus::AlreadyRunning;
        }
        Err(e) => return ExitStatus::Error(format!("worker lock: {e}")),
    };
    if pause_flag(&cfg.data_dir).exists() {
        println!("paused ({} exists)", pause_flag(&cfg.data_dir).display());
        return ExitStatus::Done;
    }
    match steps::Driver::open(cfg, llm, embedder).and_then(|mut d| d.run()) {
        Ok(status) => status,
        Err(e) => ExitStatus::Error(e.to_string()),
    }
}
