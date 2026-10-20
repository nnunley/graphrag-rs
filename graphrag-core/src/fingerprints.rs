//! Per-step input fingerprints. Each step hashes its inputs under its own
//! domain tag, so equal bytes fed to two steps never collide.

use crate::token::fingerprint_parts;

/// Identity of a document's text (`documents.content_fp`).
pub fn document_fp(content: &str) -> String {
    fingerprint_parts(&[b"graphrag-step-document-v1", content.as_bytes()])
}
