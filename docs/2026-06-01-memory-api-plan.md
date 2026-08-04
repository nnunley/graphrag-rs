# GraphRAG Memory API Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shift `graphrag-rs` toward a durable-write-first memory API with native semantic MCP reads, while keeping `note`/`ask` compatibility during the transition.

**Architecture:** Keep compatibility writes at the edge, but split them from enrichment work internally. Introduce a small persisted work queue with lane/urgency metadata, make `note` durably append before embeddings or graph work, and make MCP reads explicitly GraphRAG-shaped instead of centering `brain.ask` semantics.

**Tech Stack:** Rust workspace, `graphrag-core`, `graphrag-cli`, `graphrag-mcp`, SQLite, HNSW/usearch, leit BM25 fusion.

---

## Current State

- `graphrag-cli` still exposes compatibility-first human verbs: `note`, `ask`, `log`, `backfill-embeddings`, `enrich`.
- `cmd_note_with_embedder` currently embeds before `add_chunk`, then synchronously updates HNSW and rewrites the lexical sidecar. This does not satisfy the "durable write first, enrich after" direction.
- `cmd_ask` and MCP `graphrag_recall` already do hybrid recall by fusing HNSW and leit BM25.
- `leit` now provides a real persisted segment format plus `SegmentView`/`MmapSegment` reopen and validation surfaces, so the `.leitseg` sidecar is no longer speculative.
- `leit` still appears to execute queries only against `InMemoryIndex`; persisted segment reopen is available for validation/access, but not yet as a drop-in search target for `ExecutionWorkspace`.
- `graphrag-mcp` already has some native-ish read tools (`graphrag_entities`, `graphrag_relations`, `graphrag_communities`, `graphrag_query_global`), but the main read path is still `graphrag_recall`, and tool responses are still mostly CLI-shaped text instead of machine-oriented structured hints.
- There is no persisted backlog/work-lane model yet. There is no optional worker surface yet.
- Verification on June 1, 2026: `cargo test --workspace` passes on this branch.

## Scope

This plan is intentionally minimal. It does not try to solve remote delegation, full semantic synthesis, or distributed workers. It defines the smallest implementation slice that moves the architecture in the direction described in [docs/2026-06-01-memory-api-notes.md](/Users/ndn/nu_libs/graphrag-rs/docs/2026-06-01-memory-api-notes.md).

## Chunk 1: Durable Write First

### Task 1: Add persisted work items for post-write processing

**Files:**
- Modify: `graphrag-core/src/db.rs`
- Modify: `graphrag-core/src/lib.rs`
- Test: `graphrag-core/src/db.rs`

- [ ] Add a minimal `work_items` table in SQLite with fields for `id`, `store`, `chunk_id`, `lane`, `status`, `urgency`, `attempts`, `payload`, `created_at`, and `updated_at`.
- [ ] Add small DB methods to enqueue work, list pending work by lane, mark running/completed/failed, and update urgency.
- [ ] Keep the schema narrow. Start with the four lanes from the notes: `write`, `cheap_enrich`, `semantic`, `global`.
- [ ] Add DB tests that prove enqueue/list/transition behavior works and is stable across reopen.

### Task 2: Make `note` durably append before enrichment

**Files:**
- Modify: `graphrag-cli/src/main.rs`
- Modify: `graphrag-core/src/db.rs`
- Test: `graphrag-cli/src/main.rs`
- Test: `graphrag-cli/tests/cli_note.rs`

- [ ] Refactor the `note` path so it creates the store if needed, writes the chunk row immediately, then enqueues follow-up work instead of requiring embeddings up front.
- [ ] Preserve the existing compatibility command name and a simple human-facing success response.
- [ ] Keep inline work budgeted and optional. In the first pass, only try the `cheap_enrich` lane inline if it can finish quickly; otherwise leave backlog behind.
- [ ] Add tests that prove a note is persisted even if embedding/enrichment work is skipped or fails later.

## Chunk 2: Minimal Lane Drainer

### Task 3: Introduce a single-process lane drainer

**Files:**
- Create: `graphrag-core/src/work.rs`
- Modify: `graphrag-core/src/lib.rs`
- Modify: `graphrag-cli/src/main.rs`
- Test: `graphrag-core/src/work.rs`
- Test: `graphrag-cli/tests/cli_enrich.rs`

- [ ] Add a small executor that can drain pending work items by lane within a caller-supplied time or item budget.
- [ ] Implement only the `cheap_enrich` lane in the first slice: embeddings, HNSW update, lexical segment rebuild, and MCP lexical cache invalidation.
- [ ] Leave `semantic` and `global` lanes as explicit stubs that are persisted but not yet executed automatically.
- [ ] Add a CLI entry point such as `graphrag drain --lane cheap_enrich --max-items N` or adapt `enrich` so it can operate on queued work instead of only rebuilding from all chunks.

### Task 4: Preserve compatibility while surfacing backlog state

**Files:**
- Modify: `graphrag-cli/src/main.rs`
- Test: `graphrag-cli/tests/cli_note.rs`

- [ ] Update CLI output for `note` to stay short and human-oriented, but include whether enrichment finished inline or remains queued.
- [ ] Avoid machine-oriented JSON in the CLI default path.
- [ ] Keep total response latency aimed at the notes target: quick success, then optional background/manual drain.

## Chunk 3: Native MCP Read Surface

### Task 5: Separate compatibility recall from native semantic reads

**Files:**
- Modify: `graphrag-mcp/src/main.rs`
- Modify: `README.md`
- Test: `graphrag-mcp/src/main.rs`

- [ ] Keep `graphrag_recall` as a compatibility tool for now.
- [ ] Add a first minimal set of native read tools that expose GraphRAG concepts directly instead of a generic "ask" surface.
- [ ] Recommended minimal set:
- [ ] `graphrag_recent_chunks`
- [ ] `graphrag_search_chunks`
- [ ] `graphrag_describe_entity`
- [ ] `graphrag_get_community_context`
- [ ] Reuse existing DB queries and formatting paths where possible. Avoid a large retrieval rewrite in this phase.

### Task 6: Return machine-usable structured hints from MCP

**Files:**
- Modify: `graphrag-mcp/src/main.rs`
- Test: `graphrag-mcp/src/main.rs`

- [ ] Change native MCP read tools to return structured JSON payloads first, with optional human-readable text as a secondary rendering.
- [ ] Include a lightweight `next_operation` or equivalent hint for the caller when a follow-up tool is the obvious next step.
- [ ] Keep CLI-style prose out of the native MCP response schema.

## Chunk 4: Queue Visibility and Deferral

### Task 7: Add backlog inspection APIs before adding long-lived workers

**Files:**
- Modify: `graphrag-cli/src/main.rs`
- Modify: `graphrag-mcp/src/main.rs`
- Test: `graphrag-cli/tests/cli_help.rs`

- [ ] Add a small backlog inspection command/tool so humans and agents can see pending work by lane and urgency.
- [ ] Do not add a daemon yet. The first minimal slice only needs persisted work plus explicit drain/inspection.
- [ ] Document the deferred worker model as follow-on work rather than implementing it now.

## Risks To Watch

- The current `note` path assumes a store dimension exists before the first chunk. A durable-write-first design needs a store bootstrap strategy that does not depend on immediate embedding success.
- HNSW and leit indexes are currently treated as the source of read readiness. After the refactor, indexed readiness becomes eventual, so native MCP reads must communicate partial freshness clearly.
- `leit` persistence is ahead of `leit` persisted-search execution. We can safely persist and reopen `.leitseg`, but unless `leit` adds query execution over `SegmentView`, `graphrag-rs` still needs either in-memory rebuilds/caches for BM25 or a local adapter over leit's segment readers.
- MCP already has several overlapping read tools. Adding native verbs without a naming cleanup will create a confusing surface unless compatibility vs native tools are labeled explicitly.

## Suggested Order

1. Persist work items in `graphrag-core`.
2. Refactor CLI `note` to append first and queue work.
3. Add the minimal lane drainer for `cheap_enrich`.
4. Expose backlog visibility.
5. Add native MCP read verbs and structured hints.
6. Revisit optional worker/remote delegation after the above is stable.
