# One memory graph: fingerprinted, incremental, self-driving ingestion

**Date:** 2026-10-19
**Status:** Draft for review (revision 5: one store, groups and sections as graph nodes; research decisions applied; generalization, salience and erosion added)
**Supersedes:** the orchestration parts of `docs/2026-06-20-enrichment-engine-design.md` (SP1). Builds on `docs/2026-08-11-paper-workflow-roadmap.md` (M2/M3 principles) and `docs/paper-contract-v1.md` (fingerprint encoding).

## Why

graphrag is the shared recall layer for Norman and for Claude. The atlas built on 2026-10-16 showed that most stores only looked ingested:

- Chunks existed without vectors.
- Notes were never extracted.
- Every edge weighed 1.0.
- Old community runs were never cleaned up, and communities were mostly 1–2-member singletons.
- Nothing summarized communities by itself.

Every step exists somewhere, but nothing checks that all of them ran, in order, for everything, and nothing runs them when they haven't. "Ingested" has meant "a chunk row exists".

The data is also split into separate stores that can't see each other:

- A note about GraphRAG in `conversations` has no connection to the GraphRAG paper in `graphrag-paper`.
- The paper was stored twice (`graphrag-paper` and `graphrag-paper-dense`, 46 identical chunks) only to compare two extraction runs.
- Notes have no way to belong to more than one grouping.

This design makes two changes:

1. **Ingestion means the whole pipeline.** Each unit of work records which step it has reached and the fingerprint of the inputs that step saw. A background worker runs whatever is missing or stale, in dependency order, until the graph is current.
2. **One graph.** Everything lives in one store. Groupings like "papers" or "conversations" are nodes in that graph, and membership is an edge. A document can belong to any number of groups, and anything can reference anything.

## Goals

1. **Complete ingestion pipeline:** document → chunk → embed → extract → add with provenance → weight edges by evidence → weighted communities → community summaries.
2. **One graph with groups as nodes.** Documents, groups and extracted entities are all nodes. Group membership, aboutness and versions are edges named with the Dublin Core / Schema.org vocabulary the repo already uses. Recall can cover everything or be scoped to a group.
3. **Status as data.** Every subject (document, chunk, the graph, each community) records, per step, the input fingerprint it was built from. A step is current only if its stored fingerprint equals the fingerprint of its current inputs. Anything else is missing or stale.
4. **Incremental.** A run does only missing or stale work. A second run on a current graph does nothing. Editing a note invalidates only the chunks whose text changed.
5. **Stored by content.** Identical text is stored once. Adding it again only attaches the new group and records the add. Comparing extraction approaches never copies data: each approach is a separate extraction attempt on the same chunk.
6. **Self-driving.** graphrag runs the steps itself against llama.cpp (`http://ndn.lan:8080/v1`, OpenAI-compatible). It never hands the work back to the calling agent.
7. **Retrieval sets priority.** A chunk returned by recall that still has incomplete steps moves to the front of the work.
8. **Visible.** `graphrag status` shows, per group and per step, how many are current, stale, missing and failed, plus the backlog and whether a worker is running.
9. **Crash-safe.** Killing the worker or taking llama.cpp offline loses nothing. The next run continues where the last one stopped and repairs nothing by hand.

## Non-goals

- Paper global search (roadmap M1/M5), citation-resolvable structured community reports (M4), and the evaluation harness (M6). Summaries here are good prose with recorded inputs, not M4 reports.
- Embedding-assisted entity resolution. Entities merge on normalized exact names.
- Datalog/rules and the live browsing server. Asserted edges here are the first piece of the fact store, but general fact assertion through an API, plus audit and redaction, is the next project.
- Provider selection or retry policy inside `graphrag-core`. Roadmap rule 1 stands: core owns plans, fingerprints and persistence; the worker owns model calls.
- Online incremental community repair. Communities are rebuilt for the whole graph when it changes (cheap at this size), and only summaries whose inputs changed are regenerated.

## Anti-goals

These would count as failures even if everything else worked:

- A graph that reports "done" while any step is missing for any chunk.
- Two sources of truth for "what's left to do", such as a separate queue that drifts from the fingerprints.
- Weights that count duplicate rows or replays instead of distinct accepted extractions.
- Retrieval that blocks or slows down while it waits for extraction.
- Garbage in the graph from text that isn't a note, such as saved LLM extraction output.
- The same text stored twice.

## Decisions Norman made

- **Status is per-chunk metadata with fingerprints.** Invalidation is by fingerprint mismatch.
- **It runs incrementally.**
- **Recall escalates.** Chunks touched by retrieval get their next steps first.
- **Adding content triggers a background worker** that runs to completion.
- **The work list lives in graphrag's own database**, not laneq. Laneq stays for handoffs between sessions.
- **Clean slate.** Old relations, entities and communities without provenance are deleted, with no retention.
- **Edge weight follows the paper:** how many distinct accepted extractions found the relationship.
- **Notes must be chunked.** A 442K-character chunk is a bug.
- **Use llama.cpp on ndn.lan, not Ollama.** The extraction prompt was chosen by a blind spike (see "Extraction profile").
- **Junk deleted** (done; see "Data cleanup").
- **The paper is re-imported, into the "papers" group only.** Notes that discuss it link to it through shared entities.
- **One graph, groups as nodes.** A document can have many group edges. "If that's not what we've designed, we need a single data store."
- **Identical content can't be stored twice** unless it adds metadata.
- **Whether group edges affect clustering is an experiment:** "they should be, potentially. It's hard to tell without trying it."
- **Use the Dublin Core / Schema.org relation vocabulary** already in `synonyms.rs` and `entity_types.rs`.

## The graph model

### Nodes

| Kind | What | Where it lives |
|---|---|---|
| Document | A note, paper, file or doc page; the unit that gets added | `documents` (content-addressed) |
| Section | A heading-delimited part of a document (markdown headings, nested by level). Notes without headings have none | `sections`, also an entity of type `section` |
| Chunk | A token-bounded span inside one section (or inside a heading-less document); the unit of embedding and extraction | `chunks` |
| Group | A named grouping: `papers`, `conversations`, `operation`, `mica-docs`, `9front-kb`, or any new one | an entity of type `group` |
| Entity | Something extracted from text: a person, software, concept, … | `entities` |

Documents and groups take part in the graph as entities, so every edge connects two entity rows. A document's entity row is linked 1:1 to its `documents` row and has type `document`. *Builder's call:* whether documents become entity rows or the edge table becomes polymorphic. Either way, edges between any two node kinds must work and be queryable the same way.

### Edges

There are three kinds, depending on where an edge comes from.

| Kind | Example | Source of truth | Weight | Lifecycle |
|---|---|---|---|---|
| **Extracted** | `tokio -[used_by]-> moor` | extraction attempts (provenance) | support = number of distinct current attempts that found it | replaced on re-extraction; deleted when support reaches 0 |
| **Asserted** | `paper.md -[part_of]-> papers`; `note 1802 -[subject]-> let-go` | an `assertions` record: who (actor-prefixed: `human:norman`, `agent:claude`, `agent:migration`, `agent:cli`, `agent:mcp`, matching OKF v0.2 provenance), when, why (optional), and an optional **context** (nullable; reserved for Cyc-style microtheories and context-scoped visibility) | 1 per active assertion | stays until retracted; retraction sets `retracted_at` and keeps the record (the lace decision, 2026-08-16) |
| **Structural** | `paper.md -[has_part]-> "3 Methods" -[has_part]-> "3.1 GraphRAG Workflow" -[has_part]-> chunk 12` | parsed from the document when it's chunked | 1 | recomputed when the document is re-chunked |
| **Derived** | `section -[mentions]-> entity` (and through it, the document) | computed from extraction: an entity was extracted from one of the document's chunks | number of the document's chunks that mention it | recomputed with extraction |

### Vocabulary

Use the canonical vocabulary in `graphrag-core/src/synonyms.rs` and `entity_types.rs` (Dublin Core and Schema.org), with these additions:

- **Group membership:** `part_of` (Dublin Core isPartOf). Groups can be `part_of` other groups. Recall scoped to a group includes its subgroups.
- **Aboutness:** add `subject` (Dublin Core subject) to the canonical relations. A document is about an entity or project.
- **Versions:**
  - A changed document is a new document node that `replaces` the old one (Dublin Core replaces/isReplacedBy).
  - The old document's chunks drop out of active retrieval and extraction, but the node and edge remain.
  - `version_of` is for editions of external works.
- **Type fix:** add a canonical `group` entity type for groupings. `collection` is reserved for the Cyc sense (a concept class whose instances share a nature), so the `collection → dataset` synonym is removed and `collection` is not used for groups.
- **`isa` and `genls` are distinct** (Cyc; see the wiki's `concepts/cyc-ontology-mechanics`). `isa` is an individual being an instance of a collection ("graphrag-rs isa software project"). `genls` is one collection generalizing another ("Rust crate genls software library"). The old `type_of` canonical, with synonyms `is_a`, `instance_of` and `kind_of`, is split: `is_a`/`instance_of` → `isa`; `kind_of`/`subclass_of`/`subtype_of` → `genls`. The extraction prompt asks the model to choose explicitly.
- **Reserved now, enforced later:** `disjoint_with` (two collections share no instances; later a merge guard) and `links_to` (an untyped link from a markdown or OKF import, so imports don't invent meaning).
- **Extraction uses the vocabulary:**
  - The extraction prompt lists the canonical relations and asks the model to use them where they fit.
  - Responses are normalized through the synonym table.
  - A relation that maps to nothing is kept with its raw name and counted in `status` as unmapped, so the vocabulary can grow from evidence.
  - Support counts merge on the canonical relation, so `uses`, `utilizes` and `leverages` support one edge.

### Recall

- `graphrag ask` and MCP `recall` search everything by default. `--in <group>` (repeatable) restricts results to chunks whose document is `part_of` that group, directly or through subgroups.
- Graph expansion during recall follows extracted, asserted and derived edges across groups. That's how a note about Leiden reaches the paper's Leiden material.
- Results show each hit's groups.

## Pipeline

```
document added (note / file / add_document)          → content-addressed; assert part_of edges
  └─ chunk        per document    fp = sha(document text, chunker kind + profile)
       └─ embed   per chunk       fp = sha(chunk text, embedder id, dim)
       └─ extract per chunk       fp = ExtractionAttemptIdentityV1(store, chunk id, chunk text, model, profile, prompt ver, schema ver, glean rounds)
            └─ graph               extracted edges (support) + asserted edges + derived mentions
                 └─ generalize    whole graph     classes merged, genls/genlPreds, lifted class edges (G² admitted), salience and dormancy
                 └─ communities   whole graph     fp = sha(weighted edge list incl. isolates, community config, seed)
                      └─ summary  per community   fp = sha(members + descriptions, edges + weights + descriptions, child summary fps, model, prompt ver)
```

Embedding and extraction are independent of each other. Each depends only on the chunk.

All fingerprints use the M0 encoding: `sha256:<hex>` from `token::fingerprint_parts`, with a domain tag per step. The bare-hex `response_sha256` and the xxh3 chunk fingerprints are not reused as step identities. The builder may migrate them to the same encoding.

### Documents (step 0) and chunking (step 1)

- **`documents`:** `id, content, content_fp UNIQUE, doc_key?, origin, source, title?, created_at`.
- **`origin`** is `source` (a note, a paper, a doc page: something that states things first-hand) or `compiled` (a wiki page or other synthesis of other sources). Compiled documents are chunked, embedded and searchable, but **not extracted**: their claims would double-count the sources they summarize. They connect to their sources through asserted `references`/`links_to` edges.
- **Document identity across edits** (*Claude's call*):
  - A document has an optional caller-supplied `doc_key`: a file path for files, a page id for imported docs.
  - Adding new text under an existing `doc_key` creates a new document that `replaces` the previous one with that key.
  - Notes have no `doc_key` and are append-only, matching brain: a correction is a new note, and an explicit `replaces` assertion is how one note retires another.
  - Without a key, nothing is auto-versioned. Adding text whose `content_fp` already exists creates no document. It records an add event (`document_adds`: document, source, surface, time) and asserts any new group edges. Adding the same text with the same groups is a no-op apart from the add event.
- **Chunks** gain `document_id`, `start`, `end` (byte offsets from `chunker::chunk_spans` or the code chunker) and `content_fp`.
- **Sections are nodes** (Norman, 2026-10-19):
  - Markdown documents are parsed into a section tree from their headings: document `has_part` top-level sections, a section `has_part` its subsections, and a section `has_part` its chunks.
  - Each section records its heading text, level, ordinal, heading path (`approval-workflow.md › Load the Workflow`), byte span, and `content_fp`.
  - A document with no headings (most notes) has no sections; its chunks hang off the document directly.
  - **Chunk boundaries never cross a section.** A section that fits in the chunk size is one chunk. A longer one is split with overlap inside that section only. Small adjacent sections are never merged.
  - Text before the first heading forms an untitled leading section.
  - Section identity across re-chunking: `(document, heading path, ordinal among same-path siblings)`. A section whose text is unchanged keeps its id and its chunks.
  - Recall results show the chunk's heading path. Derived `mentions` edges attach to the section, so the graph shows which part of a document discusses an entity.
- **Re-chunking:**
  - A document whose text didn't change is never re-chunked.
  - If the chunker profile changes, chunks whose `content_fp` still exists in the new split keep their id and all downstream state. New chunks are created. Chunks that disappeared are deleted along with their extractions and provenance links.
- **Every write surface goes through this path:** CLI `note` (which today stores a whole note as one chunk), MCP `add_document`/`add_code`, Nushell `add` and Python.
- **Chunk size** (*Claude's call*): 2,000 characters with 200 overlap, markdown-aware. That's about 500 tokens, close to the paper's 600/100. A hard cap guarantees that no chunk sent to the LLM exceeds that size. `add_code` uses `code_chunker`, and the chunker kind is part of the fingerprint.
- **`graphrag note`** takes optional `--group <name>` (repeatable) and defaults to `conversations`. A group named for the first time is created as a `group` node.

### Embed (step 2)

- `chunk_embeddings` gains `fp`. One embedder serves the whole store: every current store is 768-dimension Nomic.
- `backfill-embeddings` becomes "embed what's missing or stale", not "re-embed everything".
- The write path may still embed inline so new notes are searchable immediately. The fingerprint makes doing it twice harmless.

### Extract (step 3)

- Uses `paper::ExtractionAttemptIdentityV1` (already in M0, not yet wired) as the step fingerprint and attempt identity. The M0 type is unchanged, so it includes `chunk_id`. Chunk ids are stable for unchanged text (see re-chunking).
- `extractions` becomes attempts keyed by that identity: `attempt_fp PK, chunk_id, model, profile, prompt_version, schema_version, state (current|superseded), outcome (ok|empty), response_sha256, raw_response, unmapped_relations, rejected_entities, created_at`.
- Outcomes:
  - **Transport failure** (timeout, 5xx, connection refused) is retryable. It writes no attempt row; it only updates the chunk's step status (`attempts`, `last_error`, `next_try_at`, exponential backoff).
  - **Malformed output** is retried up to N times. After that the step is `failed` with the error shown in `status`, and it is never silently treated as empty.
  - **A valid empty extraction** is terminal: outcome `empty`.
- **Supersession:**
  - When a chunk's extraction fingerprint changes (its text, the model or the prompt version changed), the new attempt becomes current in the same transaction that supersedes the old one.
  - Superseded attempts stop counting toward weights. They and their links are deleted, since Norman chose no retention for extraction history.
  - To compare extraction approaches, the builder may keep a superseded attempt under an explicit `--keep-attempts` experiment flag. It still never counts toward weights.
- **Element instances (roadmap M2):** each relationship an attempt found is recorded in `extraction_relations` (attempt, relation, description, strength), and each entity in `extraction_entities` (attempt, entity, description). Canonical `entities` and `relations` remain deduplicated current objects.
- **Hygiene before writing:**
  - Reject entity names that are sentence-shaped (very long, terminal punctuation, clause shape; as in the 2026-08-06 evidence doc).
  - Normalize relations and types through the synonym tables.
  - Count rejects and unmapped relations on the attempt.

### Weights (step 4, derived)

- `support(extracted relation)` = the number of distinct **evidence units** with a current attempt that links it. An evidence unit is a section, or the document when it has no sections (*Claude's call*). One relation found in two overlapping chunks of the same section counts once: the LLM doesn't report where in a chunk it found a relation, so overlap can't be de-duplicated by position. Counting by section keeps repetition across sections and documents, which is the paper's signal, and removes the overlap double-count.
- The Leiden edge weight between two nodes is the sum of:
  - the support of the extracted relations between them;
  - `asserted_weight` × the active asserted edges between them;
  - `mention_weight` × the derived mention edges between them;
  - `structure_weight` × the structural `has_part` edges between them.
- `asserted_weight`, `mention_weight` and `structure_weight` are community-config values, part of the community fingerprint (see "Clustering experiment").
- *Builder's call:* whether to also scale by extracted strength. The default is no, matching the paper's use of instance counts.
- Extracted relations and entities with zero current support and no asserted edges are deleted in the same transaction that removes their last support.
- No stored weight column is required. If one is added as a cache, it must be recomputed in the same transaction as attempt changes. A test must show that forcing all weights to 1.0 fails (the roadmap M3 mutant).
- Entity type (*Claude's call*): each instance keeps the type its attempt gave. The canonical entity's type is the most common type across current attempts, with ties going to the most recent. `status` lists entities whose instances disagree, which is how the `jj`-as-Person class of error shows up.
- Relation identity: directed `(head, tail, canonical_relation)`. Entities are matched on case-folded, trimmed names within the single store. Canonical descriptions are the distinct instance descriptions, newest first, packed to a fixed budget. LLM-aggregated descriptions (roadmap M2) are deferred.

### Communities (step 5, whole graph)

- One path for every surface: weighted hierarchical Leiden over the current graph, including isolates. It replaces both the CLI `enrich --levels` path and the unused `build_community_hierarchy`.
  - *Builder's call:* `leiden_hierarchical`'s downward subdivision with a fixed seed.
  - Subdivision stops when Leiden stops splitting, capped at 4 levels.
- **Node set:** extracted entities always. Document, section and group nodes are included only when one of `asserted_weight`, `mention_weight` or `structure_weight` is > 0.
- A **community run** records `run_id, graph_fp, config (incl. asserted_weight, mention_weight), seed, created_at, modularity`. Communities belong to a run. The current run is replaced in one transaction (no left-over runs).
- **When it runs:** when `graph_fp` differs from the current run's, and there's no ready chunk-level work. During a large backlog, the builder may rebuild at most once every N minutes so that escalated chunks show up in communities without a rebuild after every extraction. Failed and backing-off chunks don't count as ready work.
- Membership never comes from the LLM.

#### Clustering experiment

Norman wants to know whether group and document edges should shape clusters. Since the weights are config inputs to the run fingerprint, testing that needs no extra mechanism.

- **Run 1:** `asserted_weight = 0`, `mention_weight = 0`, `structure_weight = 0`. Text-only clusters; this is the default until the experiment decides.
- **Run 2:** all three moderate (*builder's call*, for example 1 each). If the two runs differ a lot, a third run isolating `structure_weight` shows whether sections or groups drive the difference.
- **Compare:**
  - community size distribution and singleton share;
  - modularity;
  - how many communities span more than one group;
  - the two blind judges' ratings of a deterministic sample of summaries from each run.
- Norman picks the default from the results.

### Summaries (step 6, per community)

- Bottom-up. A leaf summary's inputs are:
  - member entities (names, types, descriptions);
  - the edges between them with weights and descriptions;
  - a few grounding chunk excerpts;
  - the groups those excerpts come from.

  A parent summary's inputs are its children's summaries.
- Prompt contract from the 2026-08-06 prototype:
  - 2–4 sentences;
  - state the theme and name member entities;
  - ground only in the provided material;
  - prefer a shorter honest summary when the evidence is thin;
  - plain prose;
  - plus: name which groups the evidence comes from when there's more than one (for example "mostly from papers, with three notes").
- Every community with at least 3 member entities, at every level, is summarized. Smaller ones are marked `skipped_small`, which counts as complete.
- Mechanical gates: the summary is non-empty, names at least one member entity, and is over 40 characters. A summary that fails a gate is retried, then marked `failed`.
- Summary rows record `fp, model, prompt_version, created_at`. A summary is reused across runs when a community with an equal fingerprint existed before, so a stable seed avoids re-summarizing unchanged communities.

## Status and work selection

- New table `step_status(subject_kind, subject_id, step, fp, state, attempts, last_error, next_try_at, priority, updated_at)`:
  - `subject_kind`: `document`, `chunk`, `graph` or `community`;
  - `state`: `current`, `failed`, or `blocked` (an upstream step for the same subject isn't current);
  - missing means no row, stale means `fp` doesn't match.
- "What to do next" is a query, not a queue. The worker computes each subject's expected input fingerprint in code, from the configuration, and compares it with `step_status.fp`. The order is `priority DESC`, then age. No stored "expected" column can drift.
- Priority levels:
  - 2 = touched by recall;
  - 1 = newly added;
  - 0 = backlog.
- The graph is **complete** when there are zero missing or stale steps. `failed` is reported separately and doesn't block communities. Failed items are retried automatically when their fingerprint changes, or with `graphrag work --retry-failed`.
- `graphrag status [--json]` reports, per group and in total, per step:
  - current, stale, missing, and failed (with the latest error);
  - unmapped relations and rejected entities;
  - the backlog by priority;
  - whether a worker is running;
  - the last llama.cpp success or failure;
  - the summary call count pending.

  It extends the existing `pipeline_status`, and chunk embedding becomes a phase.

## The worker

- `graphrag work [--in GROUP] [--once] [--max-items N] [--retry-failed] [--accept-reextract]` drains ready work in priority order, then exits.
- The worker loop is a library function in a new `graphrag-worker` crate (it depends on core plus graphrag-llm, so core stays provider-independent). `graphrag work` is its CLI entry point.
- Single instance via an advisory lock file in the data directory (`flock`). A second invocation exits right away with status "already running".
- **Triggers:**
  - A write (CLI `note`, MCP add) records status and spawns `graphrag work` detached if no worker holds the lock. Spawners try `$GRAPHRAG_BIN`, then `graphrag` on PATH, then `~/.cargo/bin/graphrag`.
  - Nushell and Python don't spawn; they rely on the backstop.
  - Recall (CLI `ask`, MCP `recall`/`search`) raises the priority of returned chunks with incomplete steps and spawns a worker the same way. That write is a single cheap UPDATE and never delays results.
  - A launchd agent runs `graphrag work` every 15 minutes as a backstop.
- **Model pinning:**
  - `GRAPHRAG_LLM_MODEL` defaults to `qwen3.8-27b`.
  - At startup the worker checks that `/v1/models` serves it. If it doesn't, the worker exits 69 (UNAVAILABLE) and reports the mismatch.
  - Changing the configured model or prompt version is the explicit act that makes extraction stale. `graphrag work` requires `--accept-reextract` when more than 10% of the chunks would be superseded, and `status` shows that count first.
- **LLM down:** after M consecutive transport failures the worker exits 75 (TEMPFAIL) and records the outage. The next trigger or the backstop retries.
- **Concurrency:**
  - llama.cpp reports 4 slots with a 131K context. Up to 4 LLM calls run at once (`GRAPHRAG_LLM_CONCURRENCY`).
  - One thread writes to SQLite.
  - A pause mechanism (flag file or unloading the launchd agent) keeps the worker out of the way of large Cargo builds on the 16 GB host.
- **SQLite:** enable `journal_mode=WAL` and a busy timeout of several seconds on every connection. Neither is set today.

## LLM client

- New `OpenAiChatClient` in `graphrag-llm`:
  - `POST {base}/chat/completions`;
  - `response_format: json_schema` when the profile has a schema;
  - `chat_template_kwargs.enable_thinking` per profile;
  - a timeout;
  - no internal retry (the worker owns retry).
- Config: `GRAPHRAG_LLM_BASE_URL` (default `http://ndn.lan:8080/v1`), `GRAPHRAG_LLM_MODEL` (default `qwen3.8-27b`), `GRAPHRAG_LLM_CONCURRENCY` (default 4).
- `OllamaChatClient` stays for local use and isn't the default.
- MCP sampling extraction is removed. `add_document`/`add_code` store, chunk and trigger the worker; they no longer call back to the agent. `graphrag_summarize_community`'s agent round-trip goes away the same way.

## Extraction profile

**Decision:** paper-shaped JSON with thinking on (`paper-json-v1`, based on arm C of the spike below).

- Entities: `name`, `type`, `description`, `kind` (the class it is an instance of, as a short singular common-noun phrase, chosen from the class list shown in the prompt when one fits), `is_class` (the term itself names a category), `broader` (the next more general class of `kind`). Relationships: `source`, `target`, `relation`, `description`, `strength 1-10`.
- Enforced with llama.cpp `json_schema`.
- `enable_thinking: true`.
- Single block, no sandwich.
- No gleaning in v1. The profile records `glean_max_rounds=0`.
- The prompt and schema are committed in `graphrag-llm`, with these fixes for the defects the judges found:
  - the canonical entity types and relation list (see "Vocabulary");
  - guidance so tools, VCSs and models are Software rather than Person;
  - ignore illustrative example names and bibliographic citations;
  - prefer specific relations over `associated_with`.
- **The corrected prompt is re-spiked before the live run** (*Claude's call*), with `scripts/extraction-spike/` (committed).
  - Inputs: the same 12 source chunks re-cut with the production chunker (2,000/200). Each judge packet is the original chunk with the extractions of all its pieces merged, so quality and timing are measured at production size.
  - Arms: arm C (old prompt) versus the corrected prompt, both at production size, plus the corrected prompt at a smaller chunk size (about 1,000 characters). The GraphRAG paper's Fig. 3 shows smaller chunks surfacing more entities, and chunk size is part of every fingerprint, so this is the last cheap moment to choose it.
  - Judges: a fresh Claude subagent and Codex `-m gpt-6-sol`. Codex is available; its configured default model isn't.
  - Gate: the corrected prompt scores at least as well as arm C on both judges, `jj` and `Opus` aren't typed Person, and no citation or example-handle entities appear.

The triple-line strategy (`strategy_for_model`'s default) is retired for this model. That default came from the 2026-08-03 spike on Qwen3.6 and nemotron3:33b over Ollama, and doesn't transfer to qwen3.8-27b.

### Extraction profile spike (2026-10-19)

**Setup:**
- Throwaway, read-only on the DB.
- 12 chunks, 3 each from operation, conversations, mica-docs and graphrag-paper. They were sampled deterministically by blake2b and limited to 300–6,000 characters.
- qwen3.8-27b on llama.cpp, temperature 0, 4 requests at a time.
- Two blind judges, a fresh Claude Opus subagent and Codex `gpt-6-sol`. Each scored shuffled X/Y/Z labels on grounding, recall, entity quality and usefulness, 0–5 each.

| Arm | Parsed | Wall/chunk (4 concurrent) | Out tokens | Entities | Rels | Claude /20 | Codex /20 |
|---|---|---|---|---|---|---|---|
| A: triple lines, no thinking (current prompt) | 11/12 | 8.1 s | 221 | 13.5 | 12.5 | 10.8 | 10.3 |
| B: paper JSON, no thinking | 12/12 | 12.2 s | 475 | 5.5 | 4.1 | 15.2 | 15.0 |
| C: paper JSON, thinking | 12/12 | 24.5 s | 1,183 | 8.8 | 6.4 | 16.1 | 15.5 |

**Findings:**
- Both judges rank C > B >> A. The gap from triples to described JSON (+4.5 to +5) is large and consistent. The gap from B to C (+0.5 to +0.9, mostly recall) is within what 12 samples can resolve.
- C was chosen as the best measured. At about 6 s per chunk with 4 running at once, re-extracting the corpus takes roughly 3 hours once.
- A's failure modes:
  - generic nouns and clause-like entities ("use JWT tokens for session management");
  - hub-and-spoke invented edges, such as a 45-entity star around one commit;
  - blank or "0" entities;
  - one unparseable output.
- Shared defects:
  - `jj` (a VCS) and `Opus` (a model) typed as Person;
  - example handles (`alice`) and citations treated as entities;
  - C sometimes uses vague `related_to` hub edges.
- One chunk (190) correctly produced an empty extraction in all arms. Valid-empty has to stay a terminal outcome.

The spike harness and judge prompt are committed under `scripts/extraction-spike/` (README included), so the re-spike and future model changes can repeat it.

## Migration (one store, clean slate)

All steps run on a copy first (rehearsal), then on live after a fresh backup.

1. Back up the DB and index files.
2. **Schema migration:** add `documents`, `document_adds`, the new chunk columns, attempts, `assertions`, `step_status`, community runs, and summary metadata. Enable WAL.
3. **Delete** all entities, relations, extraction rows, provenance links, entity embeddings and merges, and communities (Norman: no retention).
4. **Delete non-note chunks:** already done on live (see "Data cleanup"); the counted check is kept for rehearsal copies.
5. **Merge into one store.** Create the single store (name: *builder's call*, for example `memory`). For each old store, create a group node:

   | Old store | Group |
   |---|---|
   | `conversations` | `conversations` |
   | `operation` | `operation` |
   | `mica-docs` | `mica-docs` |
   | `9front-kb` | `9front-kb` |
   | `graphrag-paper` and `graphrag-paper-dense` | `papers` |
6. **Documents from existing chunks** (*Claude's call*; checked against the live DB on 2026-10-19):

   | Group | Live sources | Migration unit |
   |---|---|---|
   | `conversations` | 379 notes with `source='brain'`, 2 `prime-session:…`, 3 `let-go/…` | one document per chunk (each is a whole note) |
   | `mica-docs` | 407 `path.md#Heading` sections from 54 files; 53 of the 54 exist under `~/development/mica` | re-import each found file as one document with `doc_key` = repo-relative path; its headings become section nodes, which recovers today's 407 sections as nodes instead of losing them. The missing file's chunks become one document, with one section per old `#Heading` label |
   | `operation` | `lesson:…` records (one chunk each), repo docs labelled `project:doc#N` (`N` is an old chunker index), 1 `operational:…` | lessons: one document per chunk. Repo docs: re-import `~/development/<project>/docs/<doc>.md` (or the repo root `<doc>.md`) when it exists, as one document with `doc_key` = that path; otherwise group the chunks sharing a `project:doc` label, in `#N` order, into one document |
   | `9front-kb` | 3 chunks from a dry run | one document per chunk |
   | `papers` | `graphrag-paper` and `-dense` chunks, plus `operation`'s `graphrag-rs:microsoft-graphrag-paper#N` chunks | dropped; replaced by the re-import in step 7 |

   - Each document gets an asserted `part_of` edge (asserter `agent:migration`) to its old store's group.
   - Content-addressing collapses any identical text.
   - Before running, the migration prints per group: documents re-imported from files, documents assembled from chunk groups, single-chunk documents, unresolved labels. The rehearsal must show zero unresolved labels.
7. **The paper:**
   - Remove the old paper chunks.
   - Re-import `docs/microsoft-graphrag-paper.md` as one document, `part_of` `papers`, chunked under the new profile.
8. **Run the worker:** embed what's missing or stale, extract everything (priority 0), then build communities (run 1 of the clustering experiment) and summaries.
9. **Verify with `status`:**
   - zero missing or stale steps;
   - the atlas shows varied weights;
   - communities are no longer mostly singletons;
   - at least one community spans `papers` and `conversations` through shared entities;
   - **homonym probe:** about 20 hand-picked generic names (worker, status, store, database, …) are checked for wrong cross-project merges, and the count goes in the rollout log.
10. **Clustering run 2 and comparison;** Norman picks the default.
11. **Surface switch:**
    - `--in <group>` is the scope everywhere (*Claude's call*). `--store`/`store` stays accepted for one release as a deprecated alias for `--in`, with a warning.
    - Callers to update in the same change: the CLI (`note`, `ask`, `status`, `enrich`/`work`, `plan`/`apply`, `backfill-embeddings`, `export`), the MCP tools that take `store` (recall, search, add_document, add_code, entities, relations, communities, query_global, find/merge entities, embed_entities), every Nushell command, graphrag-py, and the brain zsh wrapper (it passes no store, so it only needs the default).
    - The brain wrapper keeps calling `graphrag note`, which now defaults to `--group conversations`.
    - The MCP server must be restarted after migration.

## Testing

All without a live LLM unless noted. Release builds.

- **Full pipeline:** a fake `ChatClient` with canned responses. The pipeline on a seeded graph reaches all-current, and a second run does zero work.
- **Sections:**
  - a markdown fixture yields the expected section tree and heading paths;
  - no chunk crosses a section boundary, and small sections aren't merged;
  - editing one section re-chunks and re-extracts only that section's chunks;
  - heading-less notes get no sections.
- **Content-addressing:** adding identical text twice yields one document and two add events. Adding it with a new group adds one edge. The text is never stored twice.
- **Fingerprint invalidation:**
  - changing one paragraph re-extracts only the affected chunks;
  - changing the prompt version re-extracts everything (and requires `--accept-reextract`);
  - a model mismatch exits 69 with nothing written.
- **Kill-and-resume:** abort the worker mid-batch and rerun. The final state equals an uninterrupted run, and no attempt is duplicated or half-applied.
- **Outcomes:** transport failure stays pending with backoff; malformed output retries, then becomes `failed`; valid empty becomes `empty`.
- **Weights:**
  - two chunks stating one relation give support 2;
  - synonyms (`utilizes`/`uses`) merge support;
  - re-applying an attempt changes nothing;
  - superseding removes its support;
  - the mutant forcing weight to 1.0 fails a test;
  - `asserted_weight`/`mention_weight` reach the Leiden adjacency (a spy test).
- **Assertions:** asserting, retracting and re-asserting `part_of` behaves as expected. A retracted edge leaves recall scope and clustering but keeps its record.
- **Group-scoped recall:** `--in papers` returns only papers chunks, including subgroups. Unscoped recall expands across groups through shared entities.
- **Escalation:** a recall hit on an unextracted chunk becomes the next item processed.
- **Locking:** of two concurrent `graphrag work` processes, one exits immediately.
- **LLM client:** `OpenAiChatClient` against a mock HTTP server, including the json_schema and thinking flags.
- **Live rehearsal** on a DB copy against llama.cpp before touching live.
- Paper M2/M3 gated capabilities stay gated unless their binding tests are actually met; this design delivers parts of both.

## Generalization, salience and erosion (revision 5, 2026-10-20)

Norman's direction (2026-10-19/20):
- a normalization pass in which named entities **and relations** work their way up to higher-level concepts;
- new classes and generalizations go live automatically;
- specific terms slowly erode when unused, leaving the more generic ones;
- the graph must not become over-connected generically, since link maintenance can be quadratic or worse.

Research basis: the wiki's `concepts/memory-decay-and-consolidation`, `concepts/graph-backbone-and-summarization` and `ideas/memory-graph-construction-research`. A class-level spike on 2026-10-19 showed:
- qwen3.8-27b names sensible classes (jj → version control system → software tool; llm-proxy → proxy service);
- but it produced 152 distinct classes for 200 entities, so classes must be canonicalized;
- upper ontologies (OpenCyc, WordNet) anchor only the upper classes. The middle of the hierarchy comes from the corpus.

### Classes and generalization hierarchies

- **Every extracted entity gets an `isa` edge** to the class node named by its `kind`, unless `is_class` is set, in which case the entity *is* a class node. `broader` gives a `genls` edge between classes. These are extracted edges, with support counted like any other.
- **Classes are canonicalized as they arrive.**
  - The extraction prompt shows the current class list (the top N classes by member count, plus the classes of entities already in the chunk's document).
  - A new `kind` within a small embedding distance of an existing class, or matching it after normalization, maps to that class.
  - Otherwise it becomes a new class immediately (live, per Norman). `status` reports new classes per day.
  - *Builder's call:* the distance threshold, tuned in the re-spike.
- **Relations get a generalization hierarchy** (Cyc's genlPreds): a table `relation_genls(relation, broader)` seeded from the canonical vocabulary (`pushes_to` → `exchanges_data_with`, `chosen_over` → `compared_with`, `utilizes` → `uses`).
  - Unmapped raw relations get a proposed broader relation from the LLM in the generalize step, live immediately.
  - The table is data, not code, so it grows from evidence.
- **Names are reduced to base forms** before matching (`llms` → `llm`, `knowledge graphs` → `knowledge graph`, on the head noun).

### Lifted class-level edges

- For each instance relation `a -[r]-> b`, the generalize step adds evidence to a candidate class edge `class(a) -[r']-> class(b)` for `r'` = r and each of r's generalizations, at each class level up to the erosion floor (below).
- **Support of a lifted edge** = the number of distinct instance pairs under it (AMIE's distinct-pair support), never summed instance support.
- **Admission test (beats chance):** build a 2×2 table of distinct pairs for (r', C1, C2) against the relation's totals. Admit if:
  - G² ≥ 10.83 (p < 0.001);
  - observed > expected;
  - at least 3 distinct pairs.

  Admitted lifted edges are **derived** edges, with `derived_from` links to the instance relations that support them. Non-admitted candidates keep their counts but aren't edges. This is what stops `software -[uses]-> software`. The thresholds are starting values, measured in the rehearsal.
- Lifted edges and `isa`/`genls` edges are **excluded from default clustering**. They get a `class_weight` in `CommunityConfig`, default 0, and join the clustering experiment.
- Never materialized: class co-membership, and transitive `genls`/`part_of`/`relation_genls` closures. Those are computed at query time.

### Salience and erosion

- **Use log (schema now):** `uses(item_kind, item_id, at, kind, source_id, weight)`. `item_kind` is entity, relation or class.
  - A **mention** is a new distinct evidence unit (section, or document) that states the item, weight 1.
  - A **recall** is the item appearing in a recall result, weight 1.0, counted at most once per item per day (Norman: things searched for repeatedly should have their salience boosted; the daily cap keeps one busy session from inflating a term).
  - An **assertion** counts as one mention.
  - Re-extracting an unchanged source is **not** a use.
- **Salience** = ACT-R base-level activation: `B = ln Σ wⱼ · aⱼ^−0.5`, with `aⱼ` the age of use *j* in days. Computed on read, and cached by the worker in a salience column. Example values: one use a year ago gives about −2.95; ten uses this month give about +0.95.
- **Dormancy:** an entity or extracted relation becomes *dormant* only when **both** hold:
  1. `B < τ`;
  2. its gist survives above it, meaning every one of its active relations is covered by an admitted lifted class edge (through its `isa` class).

  A unique specific, such as a one-off gotcha whose details don't generalize, never goes dormant however old it is. Erosion only removes detail that's already summarized one level up.
  - Dormant items drop out of recall's graph expansion, clustering and summaries.
  - They are **never deleted**, and the next use revives them.
  - Their evidence stays counted in the lifted class edges above them, which is how specifics erode while the gist remains.
  - τ starts at −3.0, so a thing mentioned once stays fully active for about a year (e^(−6) ≈ 1/403 days). It is calibrated in the rehearsal.
- **Raw notes are never groomed (Norman, 2026-10-20).** A document's text, which is the raw note, is never made dormant, folded or deleted by any automated process: the worker, salience, generalization, folding or any future grooming pass. Its sections and chunks are views derived from that text. They're never made dormant or folded, and while the document exists it always has a current set of them, embedded and keyword-indexed, so vector and keyword recall always see it. The one automated change allowed is rebuilding those views when the chunker profile changes (re-chunking replaces chunks; the text is untouched).
  - **Folding or deleting a document is explicit only:** a deliberate command by Norman or Claude, recorded as an assertion with the actor (`human:norman` / `agent:claude`), when, and why.
  - The commands (`graphrag forget`, `graphrag fold`) belong to the later audit/redaction project. This design only guarantees that no automatic path exists.
- **Erosion floor (basic level, not top).** Specifics erode toward the nearest `genls` ancestor whose information content `IC(c) = −log p(c)` (p = the share of entity instances under c) is above a corpus-calibrated floor and that has at least 3 members. The top classes (`software`, `thing`) have IC near 0 and are never the erosion target. Lifting stops at that floor too, so there are no edges between near-root classes.
- **Folding (reserved, later):** an `edge_status` of `active | dormant | implied` is reserved now, for entities and relations only (never documents). A later batch pass may fold a dormant specific node into its class when the Navlakha MDL merge score is positive and the neighborhood error is at most ε. Folded edges are marked `implied`, never deleted.
- **Backbone (later):** a per-edge disparity-filter α, computed at query time from support weights, can thin the active graph for recall expansion. It needs varied weights; uniform 1.0 keeps nothing.

### Growth check

The rehearsal measures active edges (non-dormant extracted, plus admitted lifted) after migrating half the corpus and after all of it. Growth should be near-linear in documents, not quadratic. It also hand-labels about 20 homonyms and about 20 fold/no-fold cases.

## Research decisions (revision 4, 2026-10-19)

From a research pass over the GraphRAG paper, recent LLM knowledge-graph construction papers (AutoSchemaKG, EDC, KGGen, a 2025 survey), OpenCyc, and OKF v0.2. Filed in Norman's wiki: `ideas/memory-graph-construction-research.md` and the pages it links. Norman approved folding these in.

1. `isa` and `genls` are split (see "Vocabulary"), and the extraction prompt chooses between them.
2. Groups are `group`, not `collection` (see "Vocabulary").
3. Assertions carry an optional context (see "Edges").
4. **Entity merges and same-as are assertions.** Merging A into B asserts `A -[same_as]-> B` (asserter recorded). Reads and the Leiden projection resolve `same_as` chains to one representative. Retracting the assertion un-merges without data loss. The existing `merge_entities` row rewrite is replaced.
5. **Each extraction instance keeps the raw relation name and description** next to the canonical relation, so a later definition-based canonicalizer (EDC-style) can run without re-extraction.
6. `disjoint_with` and `links_to` are reserved vocabulary (see "Vocabulary").
7. Asserters are actor-prefixed (see "Edges").
8. Documents carry `origin` (`source` or `compiled`); compiled ones aren't extracted (see "Documents").
9. The re-spike compares two chunk sizes, and the rehearsal runs a homonym probe (see "Extraction profile" and "Migration").
10. Support counts distinct evidence units (sections), not chunks (see "Weights").

Deferred, but the schema doesn't block them:
- Cyc default/monotonic assertion strength;
- closure rules (subgroups, `genls` chains) computed as queries;
- EDC-style canonicalization;
- an upper-ontology concept layer, kept out of default clustering;
- wiki export (index and log as generated views, summaries as compiled artifacts);
- a hand-labeled evaluation set.

## Data cleanup (approved and done 2026-10-19)

All of the following were checked against the live DB on 2026-10-19 and deleted in one transaction that checked the exact counts (239 chunks), along with the `test` store row and one orphaned chunk embedding (chunk 526). Backup: `backups/2026-10-19-pre-conversation-cleanup/graphrag.db`.

1. **All 218 `session:` chunks in `conversations`** (MCP testing debris from 2026-01-14, created between 00:01 and 01:33):
   - 191 saved extraction responses, including the 442K chunk;
   - 27 test inputs or `Entities: ... Summary:` stubs.
2. **14 copies of the GraphRAG paper in `conversations`** (`source LIKE 'graphrag-paper%'`).
3. **Test data:** the `test` store (5 chunks), chunk 238 (`test-ingestion` Rust hello-world), and chunk 255 (`test: smoke`).

`conversations` now holds 384 chunks: 379 brain notes, 2 prime-session notes and 3 let-go notes.

## Left to the builder

- Table and column names; whether documents and groups are entity rows or edges are polymorphic; how much of the existing `extractions` table to reshape versus replace.
- The single store's name.
- Exact retry counts, backoff, and the outage threshold.
- Whether communities rebuild on a timer during a large backlog.
- The launchd plist and the pause mechanism.
- Fragment-entity heuristics, tuned against the spike outputs.
- The moderate weights for clustering run 2.
