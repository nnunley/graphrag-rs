//! BM25 lexical retrieval over chunks, backed by leit.
//!
//! leit Phase 1 is build-and-search-in-memory only — `SegmentView::open` is
//! a structural validator, not a searcher, and `InMemoryIndex::new` is
//! `pub(crate)`. So there's no persistence layer here; callers either rebuild
//! per query (CLI, one-shot) or cache the [`LexicalIndex`] in process memory
//! (MCP server, long-lived).
//!
//! When leit ships Phase 2 cursor-based persistence, swap [`build_from_chunks`]
//! for a reload path and add an `.leitseg` sidecar next to the `.usearch` file.

use crate::db::Chunk;
use crate::error::GraphRagError;
use leit_core::{FieldId, NoFilter};
use leit_index::{ExecutionWorkspace, InMemoryIndex, InMemoryIndexBuilder, SearchScorer};
use leit_text::{Analyzer, CaseMapping, FieldAnalyzers, UnicodeNormalizer, WhitespaceTokenizer};

/// Single content field. We don't (yet) project metadata or source into
/// separate fields for BM25F — single-field BM25 is the floor that already
/// closes the keyword-recall gap that vector-only retrieval misses.
const CONTENT: FieldId = FieldId::new(1);

/// A built-and-ready BM25 index over a chunk corpus.
///
/// Hold this in memory and call [`search`](Self::search). Rebuild from
/// [`Chunk`] slice whenever the corpus changes — leit doesn't (yet) support
/// incremental adds-after-build.
pub struct LexicalIndex {
    inner: InMemoryIndex,
}

impl LexicalIndex {
    /// Build a BM25 index over `chunks.content`. Chunk IDs are mapped to
    /// `u32` document IDs (leit's native). Caller is responsible for
    /// translating IDs back to chunk IDs after search.
    ///
    /// Returns `Ok(None)` when the corpus is empty — distinguishes "no
    /// matches" from "no index built" upstream.
    pub fn build_from_chunks(chunks: &[Chunk]) -> Result<Option<Self>, GraphRagError> {
        if chunks.is_empty() {
            return Ok(None);
        }

        let mut analyzers = FieldAnalyzers::new();
        analyzers.set(
            CONTENT,
            Analyzer::new(WhitespaceTokenizer::new()).with_normalizer(
                UnicodeNormalizer::builder()
                    .case_mapping(CaseMapping::Fold)
                    .build(),
            ),
        );
        let mut builder = InMemoryIndexBuilder::new(analyzers);
        builder.register_field_alias(CONTENT, "content");

        for chunk in chunks {
            let doc_id = u32::try_from(chunk.id).map_err(|_| {
                GraphRagError::Other(format!(
                    "chunk id {} exceeds u32 — leit document ids are u32",
                    chunk.id
                ))
            })?;
            builder
                .index_document(doc_id, &[(CONTENT, chunk.content.as_str())])
                .map_err(|e| GraphRagError::Other(format!("leit index_document: {e}")))?;
        }

        Ok(Some(Self {
            inner: builder.build_index(),
        }))
    }

    /// Serialize the underlying index. Today this is **write-only** — leit's
    /// Phase 1 has no public reload path (`SegmentView` is a validator, not a
    /// searcher; `InMemoryIndex::new` is `pub(crate)`). We write the bytes
    /// anyway as a forward-compat sidecar: when leit Phase 2 ships
    /// cursor-based persistence, callers swap [`build_from_chunks`] for a
    /// reload with no on-disk migration.
    pub fn to_segment_bytes(&self) -> Result<Vec<u8>, GraphRagError> {
        self.inner
            .to_segment_bytes()
            .map_err(|e| GraphRagError::Other(format!("leit to_segment_bytes: {e}")))
    }

    /// Run a BM25 query, return (chunk_id, score) pairs ranked high-first.
    ///
    /// Natural-language queries are auto-OR'd. leit's query parser defaults
    /// to AND across whitespace-separated terms; for brain-style "what do
    /// I know about X" queries that's the wrong default — one unrelated
    /// term ("non-primary worktree" against a doc that only says "worktree")
    /// kills the entire match. We split on whitespace and join with " OR "
    /// unless the user already supplied explicit operators (OR/AND/parens/
    /// field-qualifiers), in which case the query is passed through verbatim.
    pub fn search(&self, query: &str, top: usize) -> Result<Vec<(i64, f32)>, GraphRagError> {
        let effective = if query_has_explicit_operators(query) {
            query.to_string()
        } else {
            query.split_whitespace().collect::<Vec<_>>().join(" OR ")
        };
        let mut workspace = ExecutionWorkspace::new();
        let hits = workspace
            .search(
                &self.inner,
                &effective,
                top,
                SearchScorer::bm25(),
                &NoFilter,
            )
            .map_err(|e| GraphRagError::Other(format!("leit search: {e}")))?;
        Ok(hits
            .into_iter()
            .map(|h| (i64::from(h.id), f32::from(h.score)))
            .collect())
    }
}

/// Detect whether the query already uses leit's operator syntax.
/// We treat as "explicit" if any of OR/AND/parens/`field:` appear as
/// recognisable tokens. Conservative — false positives just mean the user
/// gets verbatim parsing, which is what they typed.
fn query_has_explicit_operators(query: &str) -> bool {
    if query.contains('(') || query.contains(')') {
        return true;
    }
    if query.contains(':') {
        // field:value qualifier
        return true;
    }
    query
        .split_whitespace()
        .any(|tok| tok == "OR" || tok == "AND" || tok == "NOT")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Chunk;

    fn chunk(id: i64, content: &str) -> Chunk {
        Chunk {
            id,
            store: "test".to_string(),
            content: content.to_string(),
            source: None,
            metadata: None,
            created_at: "2026-05-29".to_string(),
        }
    }

    #[test]
    fn empty_corpus_returns_none() {
        let result = LexicalIndex::build_from_chunks(&[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn bm25_finds_specific_keyword() {
        let corpus = vec![
            chunk(1, "jj worktrees don't respect the safe-push alias"),
            chunk(2, "rust ownership semantics for vector buffers"),
            chunk(3, "git push to remote with explicit branch"),
        ];
        let index = LexicalIndex::build_from_chunks(&corpus).unwrap().unwrap();
        let hits = index.search("safe-push", 5).unwrap();
        assert!(!hits.is_empty(), "exact-token query should match doc 1");
        assert_eq!(
            hits[0].0, 1,
            "doc 1 has 'safe-push' literally; should rank #1"
        );
    }

    #[test]
    fn multi_word_query_or_defaults() {
        // Natural-language query has terms not all present in any doc.
        // Without auto-OR, leit's AND-default returns nothing. With it,
        // doc 1 wins on partial overlap.
        let corpus = vec![
            chunk(1, "jj worktrees do not respect the safe-push alias"),
            chunk(2, "rust ownership"),
        ];
        let index = LexicalIndex::build_from_chunks(&corpus).unwrap().unwrap();
        let hits = index
            .search("jj safe-push non-primary worktree", 5)
            .unwrap();
        assert!(
            !hits.is_empty(),
            "auto-OR should let doc 1 match on jj+safe-push+worktree"
        );
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn explicit_operators_pass_through() {
        let corpus = vec![chunk(1, "alpha beta"), chunk(2, "beta gamma")];
        let index = LexicalIndex::build_from_chunks(&corpus).unwrap().unwrap();
        // Verbatim parsing for queries with parens/field-qualifiers/OR.
        let hits = index.search("alpha AND beta", 5).unwrap();
        assert_eq!(hits.len(), 1, "AND should keep verbatim semantics");
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn bm25_ranks_term_overlap() {
        let corpus = vec![
            chunk(1, "rust"),
            chunk(2, "rust rust rust"),
            chunk(3, "unrelated content here"),
        ];
        let index = LexicalIndex::build_from_chunks(&corpus).unwrap().unwrap();
        let hits = index.search("rust", 3).unwrap();
        assert_eq!(hits[0].0, 2, "term repetition should win BM25");
    }
}
