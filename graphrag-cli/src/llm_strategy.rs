//! Adapter: graphrag-llm extraction strategies -> the core `ExtractionStrategy`
//! trait.
//!
//! This lives in the CLI rather than in graphrag-core so the core crate keeps
//! no LLM/HTTP dependency. The CLI is the layer that legitimately sees both.

use graphrag_core::db::EntityInput;
use graphrag_core::mr::{ExtractPrompt, ExtractionStrategy};
use graphrag_llm::strategy_for_model;

/// Wraps the model-resolved graphrag-llm strategy (triple-lines by default,
/// JSON for nemotron-class) behind the core trait.
pub struct LlmStrategy {
    inner: Box<dyn graphrag_llm::ExtractionStrategy>,
}

impl LlmStrategy {
    pub fn for_model(model: &str) -> Self {
        Self {
            inner: strategy_for_model(model),
        }
    }
}

impl ExtractionStrategy for LlmStrategy {
    fn prompt(&self, chunk: &str, chunk_index: usize, total: usize) -> ExtractPrompt {
        let p = self.inner.build_prompt(chunk, chunk_index, total);
        ExtractPrompt {
            system: p.system,
            user: p.user,
            format: self.inner.response_format(),
        }
    }

    fn parse(&self, response: &str) -> Vec<EntityInput> {
        self.inner
            .parse(response)
            .triples
            .into_iter()
            .map(|t| EntityInput {
                head: t.head,
                head_type: t.head_type,
                relation: t.relation,
                tail: t.tail,
                tail_type: t.tail_type,
                properties: None,
            })
            .collect()
    }
}
