//! Provider-agnostic chat prompt. Core builds prompt text (summaries);
//! `graphrag-llm` re-exports this type and owns the clients that send it.

/// A system + user message pair with an output budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatPrompt {
    pub system: String,
    pub user: String,
    pub max_tokens: u32,
}
