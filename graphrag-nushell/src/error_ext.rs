use graphrag_core::GraphRagError;
use nu_protocol::{LabeledError, Span};

/// Extension trait for converting GraphRagError to nushell's LabeledError
pub trait GraphRagErrorExt {
    fn into_labeled_error(self, span: Span) -> LabeledError;
}

impl GraphRagErrorExt for GraphRagError {
    fn into_labeled_error(self, span: Span) -> LabeledError {
        LabeledError::new(self.to_string()).with_label("graphrag error", span)
    }
}
