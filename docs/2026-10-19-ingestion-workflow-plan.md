# One Memory Graph: Ingestion Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn graphrag-rs into one content-addressed memory graph where documents, sections, groups and extracted entities are nodes. A lock-guarded background worker drives every ingestion step (chunk, embed, extract, weight, communities, summaries) to completion against llama.cpp, tracked by per-step input fingerprints.

**Architecture:**
- `graphrag-core` owns the schema, the fingerprints, step status, section-bounded chunking, content-addressed documents, assertions, extraction attempts, evidence weights, community runs and summary persistence, with no LLM dependency.
- `graphrag-llm` owns the OpenAI-compatible client and the `paper-json-v1` extraction profile.
- A new `graphrag-worker` crate owns the drain loop, locking, model pinning, retries and spawning.
- CLI and MCP become thin surfaces that write, escalate and spawn.
- A one-shot `graphrag migrate v2` merges the old stores into one.

**Tech Stack:** Rust 2024 workspace, rusqlite 0.32 (bundled), text-splitter 0.29 (markdown), ureq 2.12, sha2, the existing `leiden.rs`, fastembed (Nomic, 768-dim), llama.cpp `/v1/chat/completions`.

**Spec:** `docs/2026-10-19-ingestion-workflow-design.md` (revision 4, approved by Norman 2026-10-19). Read it before every task. When this plan and the spec disagree, the spec wins; stop and report the conflict.

## Global Constraints

- Release builds only: `cargo test --release`, `cargo build --release`. Run one cargo job at a time; the shared `CARGO_TARGET_DIR` breaks under parallel builds.
- Every task ends with `cargo fmt --all --check` and `cargo clippy --release --workspace --all-targets -- -D warnings` clean.
- Fingerprints: lowercase `sha256:<64 hex>` from `token::fingerprint_parts`, with a distinct domain tag per step (`graphrag-step-<name>-v1`).
- LLM: `GRAPHRAG_LLM_BASE_URL` default `http://ndn.lan:8080/v1`; `GRAPHRAG_LLM_MODEL` default `qwen3.8-27b`; `GRAPHRAG_LLM_CONCURRENCY` default `4`.
- Chunking: 2,000 characters, 200 overlap, markdown-aware, never crossing a section boundary, with a hard cap of 2,000 characters per chunk.
- Priorities: `2` = touched by recall, `1` = newly added, `0` = backlog.
- Worker exit codes: `0` done, `69` model not served / endpoint unavailable, `75` LLM outage (M consecutive transport failures), `64` usage.
- Re-extract guard: `--accept-reextract` is required when more than 10% of chunks would be superseded.
- Summaries only for communities with at least 3 member entities. Smaller ones are `skipped_small`. Hierarchy capped at 4 levels.
- Every connection: `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`, `PRAGMA foreign_keys=ON`.
- Commit messages: conventional prefix, no AI attribution (a hook enforces this). Don't push without Norman's go-ahead.
- Nothing touches the live DB at `~/Library/Application Support/graphrag/graphrag.db` until Task 20, and only after a fresh backup.

## Review Focus

1. **Multibyte text at chunk and section boundaries** (CJK, emoji, combining marks): byte offsets must land on char boundaries and slice the document exactly. Test in Task 3.
2. **Markdown headings inside fenced code blocks** (`# comment` in a ```` ```bash ```` block) must not open sections. Test in Task 3.
3. **A second process writing while the worker writes** (the MCP server adding a note mid-drain): neither write is lost and neither fails with `SQLITE_BUSY`. Test in Task 1.
4. **The same entity spelled differently across groups** (`GraphRAG` / `graphrag ` / `GRAPHRAG`): it merges into one node, so notes and papers connect. Test in Task 8.
5. **A relationship whose endpoint isn't in the response's entity list:** the endpoint is created with type `other`, the relation is kept, and the event is counted in `unmapped`/hygiene stats rather than failing the attempt. Test in Task 11.

---

## File Structure

| Path | Responsibility |
|---|---|
| `graphrag-core/src/db.rs` (modify) | connection pragmas; `schema_v2` call from `open` |
| `graphrag-core/src/schema_v2.rs` (create) | v2 DDL, `PRAGMA user_version` gate |
| `graphrag-core/src/fingerprints.rs` (create) | one pure fn per step fingerprint |
| `graphrag-core/src/sections.rs` (create) | markdown → section tree with byte spans |
| `graphrag-core/src/chunk_plan.rs` (create) | section-bounded chunk planning |
| `graphrag-core/src/nodes.rs` (create) | document/section/group node rows, `same_as` resolution |
| `graphrag-core/src/assertions.rs` (create) | assert/retract edges with records |
| `graphrag-core/src/documents.rs` (create) | content-addressed add, `doc_key` versioning, re-chunk identity |
| `graphrag-core/src/status.rs` (create) | `step_status`, next-work query, escalation, report |
| `graphrag-core/src/attempts.rs` (create) | extraction attempts, instances, supersession, GC |
| `graphrag-core/src/weights.rs` (create) | support counts, Leiden projection |
| `graphrag-core/src/community_runs.rs` (create) | run build/replace, gate |
| `graphrag-core/src/summaries.rs` (create) | summary planning, gates, reuse |
| `graphrag-core/src/synonyms.rs`, `entity_types.rs` (modify) | vocabulary additions |
| `graphrag-core/src/migrate_v2.rs` (create) | old stores → one graph |
| `graphrag-llm/src/openai.rs` (create) | `OpenAiChatClient` |
| `graphrag-llm/src/paper_json.rs` (create) | `paper-json-v1` prompt, schema, parser |
| `graphrag-worker/` (create crate) | drain loop, lock, pin, retry, spawn |
| `graphrag-cli/src/main.rs` (modify) | `work`, `status`, `note --group`, `ask --in`, `migrate v2` |
| `graphrag-mcp/src/main.rs` (modify) | drop sampling extraction, spawn, escalate, `in` scope |
| `graphrag-nushell/`, `graphrag-py/` (modify) | compile against the new API, `in` scope |
| `scripts/launchd/com.ndn.graphrag-work.plist` (create) | 15-minute backstop |
| `scripts/extraction-spike/` (modify) | production-size re-spike arms |

Core tests go in `graphrag-core/tests/<module>.rs`, one file per new module, opening a fresh DB in `tempfile::tempdir()` the way `mr_contract.rs:68` does.

---

## Phase A — Storage foundation

### Task 1: Connection pragmas and the v2 schema gate

**Files:**
- Create: `graphrag-core/src/schema_v2.rs`
- Modify: `graphrag-core/src/db.rs:97-108` (`open`), `graphrag-core/src/lib.rs`
- Test: `graphrag-core/tests/schema_v2.rs`

**Interfaces:**
- Produces:
  - `pub(crate) fn apply_v2(conn: &rusqlite::Connection) -> Result<(), GraphRagError>`, idempotent, sets `PRAGMA user_version = 2`.
  - `Database::open` applies the pragmas, then the v1 init/migrate, then `apply_v2`.
  - `pub fn Database::schema_version(&self) -> Result<i64, GraphRagError>`.
  - `pub(crate) fn Database::conn(&self) -> &Connection` for sibling modules.
- v2 DDL created here (later tasks use these exact names):
  - `documents(id, content, content_fp UNIQUE, doc_key, origin TEXT NOT NULL DEFAULT 'source' CHECK(origin IN ('source','compiled')), source, title, node_id, created_at)`
  - `document_adds(id, document_id, source, surface, added_at)`
  - `sections(id, document_id, parent_id, heading, level, ordinal, heading_path, start, end, content_fp, node_id)`
  - `chunks` gains `document_id, section_id, start, end, content_fp`
  - `assertions(id, head_id, relation, tail_id, asserter, reason, context, asserted_at, retracted_at)`, where `context` is nullable
  - `step_status(subject_kind, subject_id, step, fp, state, attempts, last_error, next_try_at, priority, updated_at, PRIMARY KEY(subject_kind, subject_id, step))`
  - `attempts(attempt_fp PK, chunk_id, model, profile, prompt_version, schema_version, outcome, response_sha256, raw_response, unmapped_relations, rejected_entities, created_at)`
  - `attempt_entities(attempt_fp, entity_id, entity_type, description)`
  - `attempt_relations(attempt_fp, relation_id, raw_relation, description, strength, evidence_kind, evidence_id)`, where evidence is `section` or `document`
  - `community_runs(run_id, graph_fp, config_json, seed, modularity, created_at, current)`
  - `communities` gains `run_id, fp`
  - `summaries(community_fp PK, summary, model, prompt_version, created_at)`
  - `entities` gains `kind TEXT NOT NULL DEFAULT 'entity'`
  - `worker_state(key TEXT PRIMARY KEY, value TEXT, updated_at)`, which holds `last_llm_ok_at`, `last_llm_error`, `last_run_started_at` and `last_run_finished_at`

- [ ] **Step 1: Write the failing tests**
  - `open_sets_wal_and_busy_timeout`: `journal_mode` == `"wal"`, `busy_timeout` == 5000, `foreign_keys` == 1.
  - `open_twice_is_idempotent_at_version_2`: open, close, reopen; `schema_version()` == 2 both times; every v2 table exists (`sqlite_master`).
  - `v1_database_upgrades_in_place`: create the v1 tables with raw `rusqlite` using the current `init_schema` DDL (copy it into the test as a constant), insert 2 chunks, then open with `Database::open`. The chunks survive and `schema_version()` == 2.
  - `concurrent_writer_waits_instead_of_failing` (Review Focus 3): two `Database` handles on one file, thread A holds a write transaction for 300 ms while thread B calls `add_chunk`; B succeeds and both rows exist.
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --test schema_v2`. Expected: FAIL (`schema_version` not found).
- [ ] **Step 3: Implement** `apply_v2` (DDL above, `CREATE ... IF NOT EXISTS`, `ALTER TABLE ... ADD COLUMN` guarded by `PRAGMA table_info`) and the pragmas in `open`.
- [ ] **Step 4: Run** the same command, plus `cargo test --release -p graphrag-core` (existing suites stay green). Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): v2 schema gate with WAL and busy timeout`.

### Task 2: Step fingerprints

**Files:**
- Create: `graphrag-core/src/fingerprints.rs`
- Modify: `graphrag-core/src/token.rs:125` (make `fingerprint_parts` `pub`), `lib.rs`
- Test: `graphrag-core/tests/fingerprints.rs`

**Interfaces:**
- Produces (all `-> String`, `sha256:` form):
  - `document_fp(content: &str)`
  - `chunk_plan_fp(document_content: &str, chunker: &ChunkerProfile)`
  - `embed_fp(chunk_text: &str, embedder_id: &str, dim: usize)`
  - `extract_fp(parts: &ExtractionAttemptIdentityParts) -> Result<String, PaperContractError>`, which delegates to `ExtractionAttemptIdentityV1::derive`
  - `graph_fp(edges: &[(i64, i64, f64)], isolates: &[i64], config: &CommunityConfig, seed: u64)`
  - `summary_fp(input: &SummaryInput, model: &str, prompt_version: &str)`
- Also produces:
  - `pub struct ChunkerProfile { kind: &'static str /* "markdown-sections" | "code" */, size: usize, overlap: usize }` with `ChunkerProfile::default_markdown()` = `("markdown-sections", 2000, 200)`.
  - `pub struct CommunityConfig { asserted_weight: f64, mention_weight: f64, structure_weight: f64, max_levels: i32 /* 4 */, min_size_to_split: usize, seed: u64 }` with `Default` = all weights `0.0`.
  - `pub struct ExtractionConfig { pub model: String, pub profile: String, pub prompt_version: String, pub schema_version: String, pub glean_max_rounds: u32 }`, which feeds `extract_fp` together with the chunk's store, id and text.
  - `SummaryInput` is used in Task 14; this task declares it as a forward struct in `fingerprints.rs` with the fields `members: Vec<(String, String, String)>, edges: Vec<(String, String, String, f64, String)>, child_fps: Vec<String>, groups: Vec<String>`. Task 14 uses this one.
- Edge and isolate lists are sorted inside `graph_fp`, so caller order doesn't matter. Weights are encoded with `f64::to_bits().to_be_bytes()`.

- [ ] **Step 1: Write the failing tests:**
  - `each_step_has_a_distinct_domain`: identical inputs to `document_fp` and `embed_fp(.., "", 0)` differ;
  - `graph_fp_ignores_edge_order`;
  - `graph_fp_changes_with_any_weight_or_config_field` (one assertion per field, including each of the three weights);
  - `extract_fp_matches_m0_identity` (equals `ExtractionAttemptIdentityV1::derive(..).fingerprint`);
  - `embed_fp_changes_with_dim`.
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --test fingerprints`. Expected: FAIL.
- [ ] **Step 3: Implement** the functions in `fingerprints.rs`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): per-step input fingerprints`.

### Task 3: Markdown section tree

**Files:**
- Create: `graphrag-core/src/sections.rs`
- Test: `graphrag-core/tests/sections.rs`

**Interfaces:**
- Produces:
  - `pub struct SectionSpan { pub ordinal: usize, pub level: u8 /* 0 = leading untitled */, pub heading: String, pub heading_path: Vec<String>, pub parent: Option<usize> /* index into the Vec */, pub start: usize, pub end: usize }`
  - `pub fn parse_sections(text: &str) -> Vec<SectionSpan>`
- Rules:
  - ATX headings `#`..`######` at line start open sections.
  - Setext headings are ignored (the corpus has none; YAGNI).
  - A section ends where the next heading of the same or a higher level starts.
  - Text before the first heading becomes a level-0 section with an empty heading, only if it's non-whitespace.
  - A document with no headings returns an empty Vec.
  - `ordinal` counts siblings that share the same `heading_path`.

- [ ] **Step 1: Write the failing tests:**
  - `nested_headings_build_paths`: `# A`/`## B`/`## C`/`# D` gives paths `[A]`, `[A,B]`, `[A,C]`, `[D]` and the right parents;
  - `leading_text_becomes_untitled_section`;
  - `no_headings_no_sections`;
  - `spans_slice_the_text_exactly` (property-style over the fixtures: concatenating the top-level spans reproduces the text);
  - `fenced_code_hash_is_not_a_heading` (Review Focus 2: `` ```bash\n# not a heading\n``` ``);
  - `multibyte_headings_keep_char_boundaries` (Review Focus 1: `# 日本語 🚀`, then body text; every `start`/`end` satisfies `text.is_char_boundary`);
  - `duplicate_sibling_headings_get_ordinals` (two `## Notes` under one `#` get ordinals 0 and 1).
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --test sections`. Expected: FAIL.
- [ ] **Step 3: Implement** a line scanner that tracks fenced-block state (```` ``` ```` and `~~~`).
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): markdown section tree with byte spans`.

### Task 4: Section-bounded chunk planning

**Files:**
- Create: `graphrag-core/src/chunk_plan.rs`
- Test: `graphrag-core/tests/chunk_plan.rs`

**Interfaces:**
- Consumes: `parse_sections` (Task 3), `chunker::chunk_spans`, `ChunkerProfile` (Task 2), `code_chunker` for `kind == "code"`.
- Produces:
  - `pub struct PlannedChunk { pub section: Option<usize> /* index into sections */, pub start: usize, pub end: usize, pub text: String, pub content_fp: String /* fingerprint_text */ }`
  - `pub fn plan_chunks(text: &str, profile: &ChunkerProfile) -> (Vec<SectionSpan>, Vec<PlannedChunk>)`
- Rule: chunk each *leaf-most text region* separately: a section's own text, excluding its subsections' spans. Regions ≤ `size` become one chunk. Longer regions go through `chunk_spans` with `size`/`overlap`, applied to that region only.

- [ ] **Step 1: Write the failing tests:**
  - `small_sections_are_never_merged` (three 100-char sections give 3 chunks);
  - `long_section_splits_inside_itself` (a 5,000-char section gives ≥3 chunks, all inside its span);
  - `no_chunk_exceeds_hard_cap` (generated 442,000-char text, with and without headings, gives every chunk `len() <= 2000`);
  - `heading_less_note_chunks_without_sections`;
  - `unchanged_section_keeps_identical_chunk_fps` (edit one section; the other sections' chunk `content_fp`s are unchanged);
  - `multibyte_chunk_offsets_are_char_boundaries` (Review Focus 1).
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --features code --test chunk_plan`. Expected: FAIL.
- [ ] **Step 3: Implement** `plan_chunks`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): section-bounded chunk planning`.

### Task 5: Graph nodes and assertions

**Files:**
- Create: `graphrag-core/src/nodes.rs`, `graphrag-core/src/assertions.rs`
- Test: `graphrag-core/tests/assertions.rs`

**Interfaces:**
- Produces:
  - `pub enum NodeKind { Entity, Document, Section, Group }`, stored in `entities.kind` as `entity|document|section|group`.
  - `pub fn Database::get_or_create_group(&self, store: &str, name: &str) -> Result<i64, GraphRagError>`, which returns the node id and creates the row with `kind='group'`, `entity_type='group'`.
  - `pub fn Database::create_node(&self, store: &str, kind: NodeKind, name: &str) -> Result<i64, GraphRagError>`. Document and section names are unique per store via a `"doc:<fp-prefix>"` / `"sec:<id>"` naming scheme, so `UNIQUE(store, name)` keeps holding.
  - `pub struct Asserter(pub String)`, validated to `human:<name>` or `agent:<name>` (for example `human:norman`, `agent:claude`, `agent:migration`, `agent:cli`, `agent:mcp`, `agent:structure`). Anything else is `GraphRagError::InvalidInput`.
  - `pub fn Database::assert_edge(&self, head: i64, relation: &str, tail: i64, by: &Asserter, reason: Option<&str>, context: Option<&str>) -> Result<i64, GraphRagError>`. This is idempotent while active: an identical active assertion returns the existing id.
  - `pub fn Database::retract_edge(&self, assertion_id: i64, by: &Asserter) -> Result<(), GraphRagError>`, which sets `retracted_at` and keeps the row.
  - `pub fn Database::active_assertions(&self, node: i64) -> Result<Vec<AssertionRecord>, GraphRagError>`.
  - `pub fn Database::groups_of(&self, node: i64) -> Result<Vec<String>, GraphRagError>`, transitive `part_of` up through groups. It's computed by a recursive query; no closure rows are stored.
  - `pub fn Database::merge_as_same(&self, from: i64, into: i64, by: &Asserter) -> Result<i64, GraphRagError>`, which asserts `from -[same_as]-> into`. `pub fn Database::resolve_entity(&self, id: i64) -> Result<i64, GraphRagError>` follows active `same_as` chains to the representative, which is the end of the chain. A cycle is an error at assert time. `merge_entities` (db.rs:1366) becomes a thin wrapper over `merge_as_same`.
  - `pub fn Database::members_of_group(&self, store: &str, group: &str) -> Result<Vec<i64> /* document node ids, incl. subgroups */, GraphRagError>`.

- [ ] **Step 1: Write the failing tests:**
  - `assert_is_idempotent_while_active`;
  - `retract_keeps_record_and_hides_edge` (row exists with `retracted_at`; `active_assertions` excludes it);
  - `reassert_after_retract_creates_new_record`;
  - `groups_are_transitive` (doc `part_of` `graphrag`, `graphrag` `part_of` `papers` → `groups_of(doc)` = `{graphrag, papers}`);
  - `document_in_two_groups`;
  - `asserter_must_be_prefixed`;
  - `context_is_stored_and_optional`;
  - `same_as_resolves_and_retract_unmerges` (after `merge_as_same(a, b)`, `resolve_entity(a) == b`; after retracting, `resolve_entity(a) == a`, and a's rows are intact);
  - `same_as_cycle_rejected`.
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --test assertions`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): node kinds, retractable assertions and same_as merges`.

### Task 6: Content-addressed documents

**Files:**
- Create: `graphrag-core/src/documents.rs`
- Test: `graphrag-core/tests/documents.rs`

**Interfaces:**
- Consumes: `plan_chunks` (Task 4), `create_node`/`assert_edge`/`get_or_create_group` (Task 5), `document_fp` (Task 2).
- Produces:
  - `pub struct AddDocument<'a> { pub store: &'a str, pub content: &'a str, pub origin: Origin /* Source | Compiled */, pub source: Option<&'a str>, pub doc_key: Option<&'a str>, pub title: Option<&'a str>, pub groups: &'a [&'a str], pub surface: &'a str, pub profile: &'a ChunkerProfile }`
  - `pub enum AddOutcome { Created { document_id: i64 }, Existing { document_id: i64, new_groups: usize }, Replaced { document_id: i64, replaced: i64 } }`
  - `pub fn Database::add_document(&self, req: &AddDocument) -> Result<AddOutcome, GraphRagError>`. One transaction does all of this:
    - inserts or reuses the document;
    - records a `document_adds` row;
    - asserts `part_of` to each group (asserter = `agent:<surface>`);
    - on a `doc_key` match with different content, creates the new document and asserts `new -[replaces]-> old`;
    - writes the sections (with section nodes and structural `has_part` edges stored as assertions with asserter `agent:structure`) and the chunks with `document_id`/`section_id`/`start`/`end`/`content_fp`;
    - sets `step_status(document, id, "chunk")` to `current` with `chunk_plan_fp`.
  - `pub fn Database::rechunk_document(&self, document_id: i64, profile: &ChunkerProfile) -> Result<RechunkReport, GraphRagError>`. A chunk whose `content_fp` is still in the new plan keeps its id. New chunks are inserted, and chunks that disappeared are deleted (cascading their attempts, links and status). `pub struct RechunkReport { kept: usize, added: usize, removed: usize }`.
  - `pub fn Database::active_documents(&self, store: &str) -> ...`, which excludes documents that are the tail of an active `replaces` assertion.
- Empty or whitespace-only content returns `GraphRagError::InvalidInput("empty document")`.

- [ ] **Step 1: Write the failing tests:**
  - `identical_text_is_stored_once` (two adds → one document, two `document_adds`, `Existing`);
  - `re_add_with_new_group_adds_one_edge`;
  - `doc_key_change_creates_replacement` (`Replaced`; old document excluded from `active_documents`; `replaces` assertion exists);
  - `notes_without_key_never_replace` (two different notes, both active);
  - `sections_become_nodes_with_has_part_edges` (fixture `# A\n## B\ntext` → document `has_part` A, A `has_part` B, B `has_part` chunk);
  - `rechunk_keeps_ids_of_unchanged_chunks` (change the profile overlap; chunks of untouched sections keep ids);
  - `empty_note_is_rejected`;
  - `compiled_origin_is_stored`.
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --features code --test documents`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): content-addressed documents with sections and doc_key versioning`.

### Task 7: Step status, next work, escalation, report

**Files:**
- Create: `graphrag-core/src/status.rs`
- Modify: `graphrag-core/src/mr.rs:576-709` (`pipeline_status` delegates to the new report)
- Test: `graphrag-core/tests/status.rs`

**Interfaces:**
- Produces:
  - `pub enum SubjectKind { Document, Chunk, Graph, Community }`
  - `pub enum Step { Chunk, Embed, Extract, Communities, Summary }`
  - `pub enum StepState { Current, Failed, Blocked }`
  - `pub fn Database::record_step(&self, kind: SubjectKind, id: i64, step: Step, fp: &str, state: StepState, error: Option<&str>) -> Result<(), GraphRagError>`
  - `pub fn Database::record_transient_failure(&self, kind, id, step, error: &str, next_try_at: &str) -> Result<u32 /* attempts */, GraphRagError>`
  - `pub fn Database::escalate_chunks(&self, chunk_ids: &[i64]) -> Result<usize, GraphRagError>`, which sets `priority=2` only where some step isn't current. It's one UPDATE.
  - `pub struct WorkItem { pub kind: SubjectKind, pub id: i64, pub step: Step, pub expected_fp: String, pub priority: u8 }`
  - `pub trait ExpectedFingerprints { fn expected(&self, db: &Database, kind: SubjectKind, id: i64, step: Step) -> Result<Option<String>, GraphRagError>; }`. The worker implements this from config. Core supplies `StaticConfig { embedder_id, dim, extraction: ExtractionConfig, community: CommunityConfig }` as the default implementation.
  - `pub fn Database::next_work(&self, cfg: &dyn ExpectedFingerprints, now: &str, limit: usize) -> Result<Vec<WorkItem>, GraphRagError>`:
    - chunk-level items (embed, extract) where the stored fp differs from the expected fp, or there's no row, ordered by `priority DESC`, then `chunks.id ASC`;
    - skips `next_try_at > now` and `Failed` with an unchanged fp;
    - returns `Graph/Communities` only when no chunk-level item is ready.
  - `pub struct StatusReport { per_group: BTreeMap<String, StepCounts>, total: StepCounts, backlog_by_priority: [usize; 3], failed_samples: Vec<(i64, Step, String)>, unmapped_relations: usize, rejected_entities: usize, type_conflicts: usize, pending_summaries: usize }`, with `pub struct StepCounts { current, stale, missing, failed: BTreeMap<Step, usize> }`.
  - `pub fn Database::status_report(&self, cfg: &dyn ExpectedFingerprints) -> Result<StatusReport, GraphRagError>`.
  - `pub fn StatusReport::is_complete(&self) -> bool` (zero missing and zero stale).

- [ ] **Step 1: Write the failing tests:**
  - `missing_status_is_work`;
  - `matching_fp_is_not_work`;
  - `changed_config_fp_makes_stale`;
  - `escalated_chunk_comes_first`;
  - `escalate_ignores_complete_chunks`;
  - `backoff_hides_item_until_next_try`;
  - `failed_is_not_retried_until_fp_changes`;
  - `communities_wait_for_chunk_work`;
  - `failed_chunks_do_not_block_communities`;
  - `report_counts_per_group` (two groups, one stale chunk in one);
  - `complete_means_no_missing_or_stale` (one failed item → `is_complete()` still true);
  - `compiled_documents_are_not_extracted` (a compiled document's chunks get embed work but no extract work, and don't count as missing extraction).
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --test status`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the suite, plus `cargo test --release -p graphrag-core --test mr_contract`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): fingerprinted step status and next-work query`.

## Phase B — Extraction and weights

### Task 8: Vocabulary and normalization

**Files:**
- Modify: `graphrag-core/src/synonyms.rs`, `graphrag-core/src/entity_types.rs:62`, `graphrag-core/src/db.rs` (`tx_get_or_create_entity` folding)
- Test: `graphrag-core/tests/vocabulary.rs`

**Interfaces:**
- Produces:
  - `pub enum RelationMapping { Canonical(&'static str), Unmapped(String) }`
  - `pub fn normalize_relation(raw: &str) -> RelationMapping`: lowercases, converts spaces and hyphens to `_`, then does a synonym lookup. Canonical names map to themselves.
  - `pub fn normalize_entity_type(raw: &str) -> &'static str`, which falls back to `"other"`.
  - `pub fn normalize_entity_name(raw: &str) -> String`: trim, collapse internal whitespace, case-fold for the key while keeping the display case of the first seen.
- Vocabulary changes:
  - add `("about", "subject")`, `("topic_of", "subject")` and the canonical `subject`;
  - split `type_of` (synonyms.rs:83-86): `is_a`, `instance_of` → `isa`; `kind_of`, `subclass_of`, `subtype_of` → `genls`. `type_of` itself maps to `isa`;
  - add the canonicals `disjoint_with`, `links_to` and `same_as` (reserved; no enforcement yet);
  - add the canonical type `group`, plus `section` and `document` kinds;
  - remove the `("collection", "dataset")` synonym. `collection` becomes a canonical type in the Cyc sense (concept class).

- [ ] **Step 1: Write the failing tests:**
  - `synonyms_map_to_one_canonical` (`utilizes`/`leverages`/`uses` → `uses`);
  - `subject_is_canonical`;
  - `group_is_its_own_type` and `collection_is_not_dataset`;
  - `isa_and_genls_are_distinct` (`instance_of` → `isa`, `kind_of` → `genls`);
  - `reserved_relations_are_canonical` (`disjoint_with`, `links_to`, `same_as`);
  - `unknown_relation_is_unmapped`;
  - `spelling_variants_merge` (Review Focus 4: `GraphRAG`, `graphrag `, `GRAPHRAG` → the same key; `get_or_create_entity` returns one id for all three).
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --test vocabulary`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the suite and the full core suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): Dublin Core vocabulary additions and name normalization`.

### Task 9: Extraction attempts, supersession and GC

**Files:**
- Create: `graphrag-core/src/attempts.rs`
- Test: `graphrag-core/tests/attempts.rs`

**Interfaces:**
- Consumes: `extract_fp` (Task 2), `normalize_*` (Task 8), `record_step` (Task 7).
- Produces:
  - `pub struct ExtractedEntity { pub name: String, pub entity_type: String, pub description: String }`
  - `pub struct ExtractedRelation { pub source: String, pub target: String, pub relation: String, pub description: String, pub strength: Option<u8> }`
  - `pub enum AttemptOutcome { Ok { entities: Vec<ExtractedEntity>, relations: Vec<ExtractedRelation> }, Empty }`
  - `pub struct AttemptRecord<'a> { pub chunk_id: i64, pub attempt_fp: &'a str, pub model: &'a str, pub profile: &'a str, pub prompt_version: &'a str, pub schema_version: &'a str, pub raw_response: &'a str, pub outcome: AttemptOutcome, pub rejected_entities: usize }`
  - `pub fn Database::commit_attempt(&self, rec: &AttemptRecord) -> Result<CommitReport, GraphRagError>`. One transaction does all of this:
    - refuses if `attempt_fp` doesn't match the chunk's current content (recompute);
    - an identical replay (same fp and same `response_sha256`) is a no-op;
    - deletes this chunk's previous current attempt with its `attempt_*` rows (supersession, no retention);
    - upserts entities and relations through normalization, and creates missing relation endpoints as `other` (Review Focus 5);
    - writes the `attempt_*` instance rows and derived `mentions` links from the chunk's section (or document) to each entity;
    - GCs entities and relations with zero current support and no assertions;
    - records `step_status(chunk, extract) = Current(attempt_fp)`.
  - `pub struct CommitReport { pub replay: bool, pub superseded: Option<String>, pub entities: usize, pub relations: usize, pub unmapped_relations: usize, pub implicit_endpoints: usize, pub gc_entities: usize, pub gc_relations: usize }`
  - `pub fn Database::relation_support(&self, relation_id: i64) -> Result<usize, GraphRagError>`, the count of distinct evidence units (the section of the chunk, or its document when it has no sections) that have a current attempt linking the relation. Each `attempt_relations` row records `raw_relation` (the model's name, before normalization) and its evidence unit.
  - `pub fn Database::canonical_entity_type(&self, entity_id: i64) -> Result<String, GraphRagError>`, the majority over current instances with the newest winning ties.
- The old `persist_extraction`, `extractions`, `extraction_entities` and `extraction_relations` are dropped from the write path. Keep the tables only until Task 19 deletes them.

- [ ] **Step 1: Write the failing tests:**
  - `two_sections_one_relation_support_two`;
  - `overlapping_chunks_in_one_section_support_one`;
  - `raw_relation_kept_beside_canonical` (`utilizes` is stored raw and counted toward `uses`);
  - `synonym_relations_share_support`;
  - `identical_replay_is_noop`;
  - `superseding_removes_old_support`;
  - `zero_support_relation_is_gc_ed`;
  - `asserted_edge_survives_gc` (an entity with an assertion isn't deleted);
  - `empty_outcome_is_terminal` (status current, no rows);
  - `implicit_endpoint_created_as_other` (Review Focus 5);
  - `stale_attempt_fp_is_rejected` (the chunk content changed after the fp was computed → error, nothing written);
  - `majority_type_wins` (two attempts say Software, one says Person → Software).
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --test attempts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): extraction attempts with supersession and evidence support`.

### Task 10: OpenAI-compatible chat client

**Files:**
- Create: `graphrag-llm/src/openai.rs`
- Modify: `graphrag-llm/src/lib.rs` (export), `graphrag-llm/Cargo.toml` (dev-dep none; use `std::net::TcpListener` for the mock)
- Test: `graphrag-llm/tests/openai.rs`

**Interfaces:**
- Produces:
  - `pub struct OpenAiChatClient { base_url: String, model: String, timeout: Duration, enable_thinking: bool }`
  - `pub fn OpenAiChatClient::from_env() -> Self`, reading the Global Constraints env vars.
  - `pub fn with_thinking(self, on: bool) -> Self`
  - `pub fn served_models(&self) -> Result<Vec<String>, LlmError>`, from `GET {base}/models`, reading `data[].id`.
  - `impl ChatClient`. `complete_with_format(prompt, Some(schema))` sends `response_format: {type: "json_schema", json_schema: {name: "graph", schema}}`, `chat_template_kwargs: {enable_thinking}`, `temperature: 0`, `max_tokens`. It returns `choices[0].message.content`.
  - `pub enum LlmError { Transport(String) /* connect, timeout, 5xx */, Http(u16, String) /* 4xx */, Malformed(String) }`. `ChatClient`'s `Result<String, String>` is kept by encoding the variant as a prefix (`"transport: …"`, `"http 400: …"`, `"malformed: …"`). `pub fn LlmError::classify(msg: &str) -> LlmErrorKind` decodes it for the worker.

- [ ] **Step 1: Write the failing tests** against a one-shot `TcpListener` mock that records the request body:
  - `sends_json_schema_and_thinking_flags`;
  - `returns_message_content`;
  - `connection_refused_is_transport`;
  - `http_500_is_transport`;
  - `http_400_is_http`;
  - `missing_choices_is_malformed`;
  - `served_models_reads_ids`.
- [ ] **Step 2: Run** `cargo test --release -p graphrag-llm --test openai`. Expected: FAIL.
- [ ] **Step 3: Implement** with `ureq` (already a dependency).
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(llm): OpenAI-compatible chat client for llama.cpp`.

### Task 11: `paper-json-v1` extraction profile

**Files:**
- Create: `graphrag-llm/src/paper_json.rs`
- Modify: `graphrag-llm/src/lib.rs` (`strategy_for_model` default for `qwen3.8` → this profile)
- Test: `graphrag-llm/tests/paper_json.rs`

**Interfaces:**
- Consumes: `ExtractedEntity`/`ExtractedRelation`/`AttemptOutcome` (Task 9; `graphrag-llm` gains a `graphrag-core` dependency without default features).
- Produces:
  - `pub const PROFILE: &str = "paper-json"`
  - `pub const PROMPT_VERSION: &str = "paper-json-v1.1"`
  - `pub const SCHEMA_VERSION: &str = "graph-v1"`
  - `pub fn build_prompt(chunk: &str, heading_path: &[String]) -> ChatPrompt`, with `max_tokens` 12000.
  - `pub fn response_schema() -> serde_json::Value`, the spike's schema (`scripts/extraction-spike/run.py` `PAPER_SCHEMA`).
  - `pub fn parse(raw: &str) -> Result<(AttemptOutcome, usize /* rejected fragments */), String>`. Malformed JSON gives `Err`. Valid JSON with both lists empty gives `Empty`.
  - `pub fn is_fragment(name: &str) -> bool`: more than 6 words, or terminal `.`/`!`/`?`, or longer than 60 chars, or empty, or numeric-only.
- The prompt starts from the spike's `PAPER_SYSTEM` and adds the judges' fixes, verbatim intent from the spec's "Extraction profile":
  - list the canonical entity types (`entity_types.rs`) and relations (`synonyms.rs::canonical_relations()`);
  - "tools, version-control systems, libraries and AI models are Software, never Person";
  - "ignore illustrative example names (e.g. alice, bob, foo) and bibliographic citations";
  - "prefer a specific relation; use associated_with only when nothing else fits";
  - "use isa when something is an instance of a kind, genls when one kind is a more specific kind of another".

- [ ] **Step 1: Write the failing tests:**
  - `prompt_lists_canonical_vocabulary`;
  - `prompt_contains_software_and_example_rules`;
  - `prompt_distinguishes_isa_from_genls`;
  - `parses_spike_arm_c_output` (fixture: copy one arm-C JSON from a spike run into `graphrag-llm/tests/fixtures/arm_c_1212.json`);
  - `empty_lists_are_empty_outcome`;
  - `broken_json_is_err`;
  - `fragments_rejected_and_counted` (`"use JWT tokens for session management"` is rejected);
  - `relation_to_undeclared_entity_is_kept` (Review Focus 5: the parser keeps the relation; Task 9 creates the endpoint).
- [ ] **Step 2: Run** `cargo test --release -p graphrag-llm --test paper_json`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(llm): paper-json-v1.1 extraction profile`.

### Task 12: Weighted projection for Leiden

**Files:**
- Create: `graphrag-core/src/weights.rs`
- Test: `graphrag-core/tests/weights.rs`

**Interfaces:**
- Consumes: `relation_support` (Task 9), assertions (Task 5), `CommunityConfig` (Task 2), `leiden::CommunityGraph`.
- Produces:
  - `pub struct Projection { pub graph: CommunityGraph, pub edges: Vec<(i64, i64, f64)>, pub isolates: Vec<i64> }`
  - `pub fn Database::project_graph(&self, store: &str, cfg: &CommunityConfig) -> Result<Projection, GraphRagError>`:
    - The weight between two nodes is the sum of the support of the extracted relations between them, plus `asserted_weight` × the active non-structural assertions, plus `mention_weight` × the derived mentions, plus `structure_weight` × the structural `has_part` assertions. Edges are undirected for Leiden.
    - Node set: `kind='entity'` always. Document, section and group nodes are included only if the relevant weight is > 0.
    - Every node id is passed through `resolve_entity` first, so `same_as` merges contract to one node.
    - Retracted assertions are excluded.

- [ ] **Step 1: Write the failing tests:**
  - `weight_equals_support`;
  - `forcing_unit_weights_fails` (the roadmap M3 mutant: a test-only `cfg` hook `force_unit_weights=true` must make `weight_equals_support` fail; assert the projected weight is 3.0, not 1.0, for a 3-attempt relation);
  - `zero_weights_exclude_structural_nodes`;
  - `asserted_weight_reaches_adjacency` (spy: an edge with weight `asserted_weight` appears between doc and group);
  - `same_as_contracts_nodes` (two merged entities project as one node carrying both nodes' edges);
  - `retracted_assertion_not_projected`;
  - `isolates_included`.
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --test weights`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): evidence-weighted Leiden projection`.

## Phase C — Communities and summaries

### Task 13: Community runs

**Files:**
- Create: `graphrag-core/src/community_runs.rs`
- Modify: `graphrag-cli/src/main.rs:523-567` (`enrich` Leiden path calls this; remove the hardcoded `1.0` at `:552`), `graphrag-nushell/src/commands/{communities,store_communities}.rs` (same)
- Test: `graphrag-core/tests/community_runs.rs`

**Interfaces:**
- Consumes: `project_graph` (Task 12), `graph_fp` (Task 2), `leiden_hierarchical`.
- Produces:
  - `pub struct RunReport { pub run_id: i64, pub reused: bool, pub communities: usize, pub levels: i32, pub modularity: f64 }`
  - `pub fn Database::build_community_run(&self, store: &str, cfg: &CommunityConfig) -> Result<RunReport, GraphRagError>`:
    - if the current run's `graph_fp` equals the new fp, return `reused: true`;
    - otherwise run `leiden_hierarchical(None, 1e-4, cfg.min_size_to_split, Some(cfg.max_levels))` with a fixed seed;
    - in one transaction, insert the run, communities (with `run_id` and a per-community `fp` = sorted member ids + internal edge weights) and memberships, mark the run `current`, and delete the previous run's communities;
    - record `step_status(graph, 0, communities) = Current(graph_fp)`.
  - `pub fn Database::current_run(&self, store: &str) -> Result<Option<CommunityRunRecord>, GraphRagError>`
- `build_community_hierarchy` (mr.rs:771) and its tests move to a `#[cfg(test)]` reference or are deleted. *Builder's call*, but no production path may call it.

- [ ] **Step 1: Write the failing tests:**
  - `same_graph_reuses_run`;
  - `changed_weight_creates_new_run_and_replaces_old` (exactly one `current` run; old communities gone);
  - `run_records_config_and_seed`;
  - `levels_capped_at_four`;
  - `interrupted_build_leaves_old_run_current` (inject an error before commit → the old run is still current).
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --test community_runs`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the suite, plus `cargo test --release -p graphrag-cli`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): fingerprinted community runs replace ad hoc clustering`.

### Task 14: Summaries

**Files:**
- Create: `graphrag-core/src/summaries.rs`
- Modify: `graphrag-core/src/mr.rs:409-480` (`plan_summarize`/`apply_summarize` delegate here)
- Test: `graphrag-core/tests/summaries.rs`

**Interfaces:**
- Consumes: `current_run` (Task 13), `SummaryInput`/`summary_fp` (Task 2), `groups_of` (Task 5).
- Produces:
  - `pub const SUMMARY_PROMPT_VERSION: &str = "summary-v1"`
  - `pub enum SummaryPlan { Skip { community_id: i64, reason: &'static str /* "skipped_small" | "reused" */ }, Call { community_id: i64, fp: String, prompt: ChatPrompt } }`. Core builds `ChatPrompt` text: `graphrag-llm`'s `ChatPrompt` moves to `graphrag-core::prompt` and is re-exported by `graphrag-llm`.
  - `pub fn Database::plan_summaries(&self, store: &str, model: &str, limit: usize) -> Result<Vec<SummaryPlan>, GraphRagError>`:
    - bottom-up: a parent is planned only when all its children are current or skipped;
    - fewer than 3 member entities → `Skip skipped_small`;
    - an existing `summaries.community_fp` row → `Skip reused`.
  - `pub fn Database::apply_summary(&self, community_id: i64, fp: &str, text: &str, model: &str) -> Result<(), SummaryGateError>`. Gates: non-empty, more than 40 chars, names at least one member entity (case-insensitive substring). On pass it writes `summaries` and `communities.summary`.
  - The prompt follows the 2026-08-06 contract and the spec's addition: it names the evidence groups when there's more than one.

- [ ] **Step 1: Write the failing tests:**
  - `small_communities_skipped`;
  - `parent_waits_for_children`;
  - `equal_fp_reused_across_runs` (rebuild with the same members → `Skip reused`, no call);
  - `gate_rejects_short_and_ungrounded`;
  - `prompt_names_groups_when_mixed` (a community with members from `papers` and `conversations` → the prompt contains both names).
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --test summaries`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the suite, plus `cargo test --release -p graphrag-core --test mr_contract`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): fingerprinted bottom-up community summaries`.

## Phase D — The worker and surfaces

### Task 15: `graphrag-worker` crate

**Files:**
- Create: `graphrag-worker/Cargo.toml`, `graphrag-worker/src/lib.rs`, `src/lock.rs`, `src/spawn.rs`, `src/steps.rs`
- Modify: workspace `Cargo.toml` (members and default-members)
- Test: `graphrag-worker/tests/worker.rs`

**Interfaces:**
- Consumes: Tasks 6–14, `OpenAiChatClient` (Task 10), `paper_json` (Task 11), the `graphrag-core` embedder.
- Produces:
  - `pub struct WorkerConfig { pub data_dir: PathBuf, pub model: String, pub concurrency: usize, pub max_items: Option<usize>, pub group: Option<String>, pub retry_failed: bool, pub accept_reextract: bool, pub outage_threshold: u32 /* 5 */, pub community: CommunityConfig }`
  - `pub enum ExitStatus { Done, AlreadyRunning, Usage, Unavailable, Outage }` with `pub fn code(&self) -> i32` returning 0, 0, 64, 69 and 75. `AlreadyRunning` prints `already running`.
  - `pub trait Embedder { fn id(&self) -> &str; fn dim(&self) -> usize; fn embed(&self, text: &str) -> Result<Vec<f32>, GraphRagError>; }`, defined in `graphrag-core/src/embedder.rs`. Implement it for the two existing embedder structs (`embedder.rs:212`, `:336`) with ids `"fastembed:nomic-embed-text-v1.5"` and `"remote:<model>"`.
  - The worker updates `worker_state` on each run start and finish and on each LLM success or failure.
  - `pub fn run(cfg: &WorkerConfig, llm: &dyn ChatClient, embedder: &dyn Embedder) -> ExitStatus`. The loop:
    1. Acquire the `flock` on `data_dir/worker.lock`, or return `AlreadyRunning`.
    2. Model pin: `served_models()` must contain `cfg.model`, or return `Unavailable`.
    3. Re-extract guard: if `status_report` shows more than 10% of chunks stale on extract, the run requires `accept_reextract`, or returns `Usage` with the count printed.
    4. Loop `next_work`. Up to `concurrency` LLM calls run in scoped threads; all DB writes go on the calling thread.
    5. Dispatch per step: embed → `set_chunk_embedding` + record; extract → prompt → parse → `commit_attempt` (malformed → retry up to 3, then `Failed`); communities → `build_community_run`; summary → `plan_summaries`/`apply_summary`.
    6. Transport errors → `record_transient_failure` with backoff `min(2^attempts, 3600)` s. `outage_threshold` consecutive failures return `Outage`.
    7. When nothing is ready, return `Done`.
  - `pub fn pause_flag(data_dir: &Path) -> PathBuf` (`worker.pause`). While it exists, `run` returns `Done` immediately.
  - `pub fn spawn_detached(data_dir: &Path) -> Result<bool /* spawned */, String>`: exec `$GRAPHRAG_BIN`, then `graphrag` on PATH, then `~/.cargo/bin/graphrag` with `work`, detached (`setsid`, null stdio). It returns `false` without spawning when the lock is held.
- The fake `ChatClient` and `Embedder` live in `graphrag-worker/tests/support.rs`.

- [ ] **Step 1: Write the failing tests** (fakes only):
  - `drains_seeded_graph_to_complete` (2 documents with sections → `status_report().is_complete()`, communities current, summaries current or skipped);
  - `second_run_does_zero_work` (the fake records the call count, which is 0 on the second run);
  - `kill_and_resume_matches_uninterrupted` (`max_items: Some(3)` then a full run equals one uninterrupted run, compared by entity/relation/support snapshot);
  - `transport_failures_back_off_then_outage`;
  - `malformed_three_times_is_failed`;
  - `model_not_served_exits_69_writes_nothing`;
  - `reextract_guard_requires_flag` (change `PROMPT_VERSION` via config → `Usage`; with the flag → proceeds);
  - `second_worker_exits_already_running`;
  - `escalated_chunk_processed_first`;
  - `pause_flag_stops_worker`.
- [ ] **Step 2: Run** `cargo test --release -p graphrag-worker`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(worker): self-driving ingestion worker with lock, pin and backoff`.

### Task 16: CLI surfaces and the launchd backstop

**Files:**
- Modify: `graphrag-cli/src/main.rs`, `graphrag-cli/Cargo.toml` (depend on `graphrag-worker`)
- Create: `scripts/launchd/com.ndn.graphrag-work.plist`, `scripts/launchd/README.md`
- Test: `graphrag-cli/tests/cli_work.rs`, update `cli_note.rs`, `mr_cli.rs`

**Interfaces:**
- Consumes: `add_document` (Task 6), `escalate_chunks`/`status_report` (Task 7), `graphrag_worker::{run, spawn_detached}` (Task 15).
- Produces these CLI changes:
  - `graphrag note <text> [--group G]... [--source S] [--doc-key K]`: `add_document` with `groups` defaulting to `["conversations"]`, embed inline, `spawn_detached`.
  - `graphrag ask <q> [--in G]... [--top N]`: scopes to `members_of_group`. After printing results it calls `escalate_chunks(hit_ids)` and, if any were escalated, `spawn_detached`. Each hit prints its groups and heading path.
  - Graph expansion lane in `ask`: for the fused top-N hits, collect their chunks' entities through `mentions`. Then add up to N/2 chunks from any group that mention those entities, ranked by the summed support of the shared entities, and fuse them as a third RRF list. This is how a note reaches the paper.
  - `graphrag work [--in G] [--once] [--max-items N] [--retry-failed] [--accept-reextract] [--pause|--resume]`: maps to `WorkerConfig` and exits with `ExitStatus as i32`.
  - `graphrag status [--json]`: prints `StatusReport` (per group: current/stale/missing/failed per step, backlog by priority, worker running, last LLM success or failure, pending summaries).
  - `--store S` stays accepted on every command as a hidden alias for `--in S`, printing `warning: --store is deprecated; use --in`.
  - `enrich` becomes an alias for `work` that prints a deprecation warning.
  - `backfill-embeddings` only embeds missing or stale chunks.
- The plist runs `~/.cargo/bin/graphrag work` every 900 s with `StandardOutPath`/`StandardErrorPath` under `~/Library/Logs/graphrag/`. Installing it is Task 20.

- [ ] **Step 1: Write the failing tests** (with `GRAPHRAG_DATA_DIR` pointing at a tempdir and `GRAPHRAG_BIN` pointing at a stub that records argv):
  - `note_defaults_to_conversations_group`;
  - `note_with_two_groups`;
  - `note_spawns_worker` (the stub was invoked with `work`);
  - `ask_in_scopes_results`;
  - `ask_escalates_incomplete_hits`;
  - `store_flag_warns_and_aliases`;
  - `unscoped_ask_reaches_paper_through_shared_entity` (a note in `conversations` and a paper chunk in `papers` share the entity `Leiden`; `ask` about the note's wording returns the paper chunk via the expansion lane; with `--in conversations` it doesn't);
  - `status_json_has_per_group_counts`;
  - `work_exit_code_passthrough` (unreachable LLM URL → exit 69).
- [ ] **Step 2: Run** `cargo test --release -p graphrag-cli`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(cli): work/status commands, group-scoped note and ask, launchd backstop`.

### Task 17: MCP surfaces

**Files:**
- Modify: `graphrag-mcp/src/main.rs` (tools `add_document` :811, `add_code` :886, `process_chunk` :991, `finalize_chunk` :1256, `summarize_community` :1434, `recall`, `search`, and the `handle_list_tools` schemas :397-611)
- Test: `graphrag-mcp/tests/mcp_tools.rs` (create; drive the server over stdio JSON-RPC as the existing manual tests do)

**Interfaces:**
- Consumes: Tasks 6, 7, 15.
- Produces:
  - `graphrag_add_document {content, source?, doc_key?, groups?: [string]}` and `graphrag_add_code` call `add_document` and `spawn_detached`. They return immediately with the document id and chunk count. No `sampling/createMessage` call remains.
  - `graphrag_recall`/`graphrag_search` accept `in?: [string]` and escalate hits.
  - `graphrag_summarize_community` returns the stored summary and its status and never samples.
  - A new `graphrag_status` tool returns `StatusReport` JSON.
  - `store` stays accepted as a deprecated alias for `in`.
  - The lexical cache is invalidated on every add.
- Remove `PendingOperation` (:184) and the reflection loop; keep `entity_extraction_prompt` only if Nushell uses it.

- [ ] **Step 1: Write the failing tests:**
  - `add_document_does_not_sample` (the server never emits `sampling/createMessage`; the response arrives within 2 s);
  - `add_document_creates_sections`;
  - `recall_in_papers_scopes`;
  - `status_tool_reports_groups`;
  - `store_alias_accepted`.
- [ ] **Step 2: Run** `cargo test --release -p graphrag-mcp`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the suite, then `cargo build --release -p graphrag-mcp`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(mcp): self-driving adds, group-scoped recall, no agent sampling`.

### Task 18: Nushell and Python bindings

**Files:**
- Modify: `graphrag-nushell/src/commands/*.rs`, `graphrag-py/src/lib.rs`
- Test: `cargo build --release -p nu_plugin_graphrag`; `maturin build -m graphrag-py/Cargo.toml --release`; existing Python smoke if present

**Interfaces:**
- Consumes: Tasks 6, 7, 9, 13.
- Produces:
  - Nushell `add` calls `add_document`: a record's `groups` list is used, and `store` maps to the group. An `entities` list commits an attempt with profile `nushell-input`.
  - `communities`/`store_communities` call `build_community_run`.
  - Python exposes `add_document`, `status_report`, `build_community_run`.
  - Neither spawns the worker; the backstop covers them.

- [ ] **Step 1: Build both.** Expected: compile errors against the removed APIs (that's the failing state).
- [ ] **Step 2: Port the call sites.**
- [ ] **Step 3: Build both again.** Expected: success. Run `nu -c 'plugin add target/release/nu_plugin_graphrag; plugin use graphrag; graphrag list'` against a temp data dir; expected output: the group list.
- [ ] **Step 4: Commit** `refactor(bindings): nushell and python on the one-graph API`.

## Phase E — Migration and rollout

### Task 19: `graphrag migrate v2`

**Files:**
- Create: `graphrag-core/src/migrate_v2.rs`
- Modify: `graphrag-cli/src/main.rs` (`migrate v2 --plan | --apply [--repo-root DIR]...`)
- Test: `graphrag-core/tests/migrate_v2.rs` with a fixture DB built in the test that mirrors the live source shapes

**Interfaces:**
- Consumes: Tasks 5, 6.
- Produces:
  - `pub struct MigrationPlan { pub per_group: BTreeMap<String, GroupPlan>, pub deletions: DeletionCounts, pub unresolved: Vec<String> }`, with `pub struct GroupPlan { from_files: usize, from_chunk_groups: usize, single_chunk: usize }`.
  - `pub fn plan_v2(db: &Database, repo_roots: &[PathBuf]) -> Result<MigrationPlan, GraphRagError>`
  - `pub fn apply_v2_migration(db: &Database, plan: &MigrationPlan, repo_roots: &[PathBuf], paper_path: &Path) -> Result<MigrationReport, GraphRagError>`. It refuses unless `plan.unresolved.is_empty()` and the plan recomputed inside the transaction equals the passed plan.
- Steps, in the spec's order (Migration 3–7):
  - delete entities, relations, old extraction tables, entity embeddings and merges, communities;
  - create the single store `memory` (*the spec leaves the name to the builder; this plan fixes it*);
  - map store → group per the spec's table;
  - resolve sources:
    - mica-docs `path.md#Heading` → `<root>/path.md` (the root is `--repo-root ~/development/mica`);
    - operation `project:doc#N` → `~/development/<project>/docs/<doc>.md`, then `~/development/<project>/<doc>.md`; if neither exists, group the chunks by `project:doc` in `#N` order;
    - `lesson:*`, `brain`, `prime-session:*`, `let-go/*`, 9front-kb → one document per chunk;
  - drop every paper chunk (`graphrag-paper`, `graphrag-paper-dense`, operation `graphrag-rs:microsoft-graphrag-paper#*`) and add `paper_path` as one document in `papers`;
  - re-attach chunk embeddings by `content_fp` where the text is unchanged;
  - drop the `stores` rows other than `memory`.

- [ ] **Step 1: Write the failing tests:**
  - `plan_counts_match_fixture`;
  - `mica_sections_become_section_nodes` (a 2-file fixture with 5 `#Heading` chunks → 2 documents, 5 sections);
  - `operation_doc_without_file_is_reassembled_in_order`;
  - `paper_copies_collapse_to_one_document`;
  - `lessons_stay_single_documents`;
  - `unresolved_label_blocks_apply`;
  - `embeddings_reattached_for_unchanged_text`;
  - `apply_is_atomic` (inject a failure → the DB is unchanged).
- [ ] **Step 2: Run** `cargo test --release -p graphrag-core --test migrate_v2`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the suite, then the whole workspace: `cargo test --release --workspace`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(migrate): merge stores into one graph with groups and sections`.

### Task 20: Re-spike, rehearsal, live run, clustering experiment

This task is operational: no new product code. **Live steps need Norman's explicit go-ahead at the marked gates.**

**Files:**
- Modify: `scripts/extraction-spike/run.py` (add arms `C_prod`, `V11_prod` and `V11_small`: production chunker (2,000/200), arm-C prompt vs `paper-json-v1.1`, plus `paper-json-v1.1` at 1,000/100; merge pieces per original chunk in `pack.py`)
- Create: `docs/2026-10-2x-ingestion-rollout-log.md` (counts, timings, judge scores, both clustering runs)

- [ ] **Step 1: Re-spike.** Run the harness per `scripts/extraction-spike/README.md` with both judges (Codex `-m gpt-6-sol`). Gate:
  - `V11_prod` ≥ `C_prod` on both judges;
  - no `jj` or `Opus` typed Person;
  - no citation or example-handle entities;
  - `isa`/`genls` used correctly in the sampled outputs;
  - pick the chunk size: if `V11_small` beats `V11_prod` on both judges by more than 1 point, switch `ChunkerProfile::default_markdown()` to 1,000/100 before the rehearsal (and note the doubled call count).

  If the gate fails, fix the prompt, bump `PROMPT_VERSION`, and repeat. Record the scores in the rollout log.
- [ ] **Step 2: Rehearsal on a copy:**
  - `sqlite3 <live> ".backup <scratch>/rehearsal.db"`;
  - `GRAPHRAG_DATA_DIR=<scratch> graphrag migrate v2 --plan --repo-root ~/development/mica`. Expected: zero unresolved, and per-group counts that match the spec's table;
  - `--apply`, then `graphrag work --accept-reextract` until exit 0;
  - `graphrag status` shows complete;
  - record wall time, attempt count, failed count, community count, singleton share, and cross-group communities;
  - homonym probe: for about 20 hand-picked generic names (worker, status, store, database, index, queue, …), list each resolved entity's groups and top relations, and count wrong cross-project merges.
- [ ] **Step 3: Show Norman the rehearsal numbers. GATE: wait for approval before touching live.**
- [ ] **Step 4: Live:**
  - fresh backup to `backups/<date>-pre-v2/`;
  - `cargo install --path graphrag-cli --force` (release);
  - rebuild `target/release/graphrag-mcp`;
  - migrate `--plan`, then `--apply`;
  - install the launchd plist (`launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ndn.graphrag-work.plist`);
  - `graphrag work --accept-reextract`;
  - verify with `graphrag status` and the atlas (`scratchpad/viz/export.py` re-run): varied weights, no singleton-dominated communities, and at least one community spanning `papers` and `conversations`.
- [ ] **Step 5: Clustering run 2:**
  - `graphrag work` with `community.asserted_weight=1, mention_weight=1, structure_weight=1` (config flag or env, *builder's call*);
  - compare the size distribution, modularity, cross-group communities, and the two judges on a blake2b-sampled set of 12 summaries per run;
  - record the results and show Norman, who picks the default.
- [ ] **Step 6:** Tell Norman to restart Claude Code (the MCP server binary changed). Update the CLAUDE.md graphrag block (stores → groups, `--in`, `graphrag status`) with his OK. Commit the rollout log.
