# Ingestion workflow: fingerprinted, incremental, self-driving

**Date:** 2026-10-19
**Status:** Draft for review
**Supersedes:** the orchestration parts of `docs/2026-06-20-enrichment-engine-design.md` (SP1). Builds on `docs/2026-08-11-paper-workflow-roadmap.md` (M2/M3 principles) and `docs/paper-contract-v1.md` (fingerprint encoding).

## Why

graphrag is the shared recall layer for Norman and for Claude. The atlas built on 2026-10-16 showed that most stores only looked ingested:

- Chunks existed without vectors.
- Notes were never extracted.
- Every edge weighed 1.0.
- Old community runs were never cleaned up, and communities were mostly 1–2-member singletons.
- Nothing summarized communities by itself. The MCP path asked the calling agent to do the work.

Every step exists somewhere: MCP sampling, `enrich --extract` against Ollama, and `plan`/`apply` JSONL. But nothing checks that all of them ran, in order, for everything, and nothing runs them when they haven't. "Ingested" has meant "a chunk row exists".

This design makes ingestion mean the whole pipeline. Each unit of work records which step it has reached and the fingerprint of the inputs that step saw. A background worker runs whatever is missing or stale, in dependency order, until the store is current.

## Goals

1. **Complete ingestion pipeline:** chunk → embed → extract → add triples with provenance → weight edges by evidence → weighted communities → community summaries.
2. **Status as data.** Every chunk and every store-level artifact records, per step, the input fingerprint it was built from. A step is current only if its stored fingerprint equals the fingerprint of its current inputs. Anything else is missing or stale.
3. **Incremental.** A run does only missing or stale work. A second run on a current store does nothing. Editing or re-adding a note invalidates only the chunks whose text changed.
4. **Self-driving.** graphrag runs the steps itself against llama.cpp (`http://ndn.lan:8080/v1`, OpenAI-compatible). It never hands the work back to the calling agent.
5. **Retrieval sets priority.** A chunk returned by recall that still has incomplete steps moves to the front of the work.
6. **Visible.** `graphrag status` shows, per store and per step, how many are current, stale, missing and failed, plus the backlog and whether a worker is running.
7. **Crash-safe.** Killing the worker or taking llama.cpp offline loses nothing. The next run continues where the last one stopped and repairs nothing by hand.

## Non-goals

- Paper global search (roadmap M1/M5), citation-resolvable structured community reports (M4), and the evaluation harness (M6). Summaries here are good prose with recorded inputs, not M4 reports.
- Embedding-assisted entity resolution. Entities merge on normalized exact names, as today.
- Datalog/rules, assert/retract of facts, and the live browsing server. These are later projects in Norman's order, and they build on the provenance this design produces.
- Provider selection or retry policy inside `graphrag-core`. Roadmap rule 1 stands: core owns plans, fingerprints and persistence; the worker owns model calls.
- Online incremental community repair. Communities are rebuilt for a whole store when its graph changes (cheap at this size), and only summaries whose inputs changed are regenerated.

## Anti-goals

These would count as failures even if everything else worked:

- A store that reports "done" while any step is missing for any chunk.
- Two sources of truth for "what's left to do", such as a separate queue that drifts from the fingerprints.
- Weights that count duplicate rows or replays instead of distinct accepted extractions.
- Retrieval that blocks or slows down while it waits for extraction.
- Garbage in the graph from text that isn't a note, such as saved LLM extraction output.

## Decisions Norman made

- **Status is per-chunk metadata with fingerprints.** Invalidation is by fingerprint mismatch.
- **It runs incrementally.**
- **Recall escalates.** Chunks touched by retrieval get their next steps first.
- **Adding to a store triggers a background worker** that runs to completion.
- **The work list lives in graphrag's own database**, not laneq. Laneq stays for handoffs between sessions.
- **Clean slate.** Old relations, entities and communities without provenance are deleted, with no retention. The 2026-10-18 backup is the only copy kept.
- **Edge weight follows the paper:** how many distinct accepted extractions found the relationship.
- **Notes must be chunked.** A 442K-character chunk is a bug.
- **Use llama.cpp on ndn.lan, not Ollama.** Extraction quality was checked with a blind spike before choosing a prompt (see "Extraction profile").

## Pipeline

```
document (note / file / add_document)
  └─ chunk        per document   fp = sha(document text, chunker profile)
       └─ embed   per chunk      fp = sha(chunk text, embedder id, dim)
       └─ extract per chunk      fp = ExtractionAttemptIdentityV1(store, chunk text, model, profile, prompt ver, schema ver, glean rounds)
            └─ (graph = entities + relations + provenance links; weight = count of distinct current attempts)
                 └─ communities  per store  fp = sha(sorted weighted edge list + isolates, algorithm config, seed)
                      └─ summary per community fp = sha(member names+descriptions, edges+weights+descriptions, child summary fps, model, prompt ver)
```

Embedding and extraction are independent of each other. Each depends only on the chunk.

All fingerprints use the M0 encoding: `sha256:<hex>` from `token::fingerprint_parts`, with a domain tag per step. The existing bare-hex `response_sha256` and the xxh3 chunk fingerprints are not reused as step identities. The builder may migrate them to the same encoding.

### Documents and chunking (step 1)

- New `documents` table: `id, store, source, title?, content, content_fp, created_at`. A note is one document.
- Chunks gain `document_id`, `start`, `end` (byte offsets into the document, from `chunker::chunk_spans`) and `content_fp`.
- Chunking is markdown-aware text-splitter with overlap. The chunker profile (size and overlap) is part of the chunk fingerprint.
- Re-chunking a document whose text didn't change is a no-op. If the text changed, chunks whose `content_fp` still exists in the new split keep their id and all downstream state. New chunks are created. Chunks that disappeared are deleted, and their extractions and provenance links go with them through cascade.
- **Every write surface goes through this path:** CLI `note` (which today stores a whole note as one chunk), MCP `add_document`/`add_code`, Nushell `add` and Python.
- Size profile (*Claude's call*): keep the existing default of 2,000 characters with 200 overlap, markdown-aware. That's about 500 tokens, close to the paper's 600/100, and every spike chunk fell in this range. A hard cap guarantees that no chunk sent to the LLM exceeds that size.

### Embed (step 2)

- `chunk_embeddings` gains `fp`. Missing or stale embeddings are computed by the worker with the configured embedder.
- `backfill-embeddings` becomes "embed what's missing or stale", not "re-embed everything".
- To keep recall immediate, the write path may still embed inline. The fingerprint makes doing it twice harmless.

### Extract (step 3)

- Uses `paper::ExtractionAttemptIdentityV1` (already in M0, not yet wired) as the step fingerprint and attempt identity.
- `extractions` becomes attempts keyed by that identity: `attempt_fp PK, chunk_id, model, profile, prompt_version, schema_version, state (current|superseded), outcome (ok|empty), response_sha256, raw_response, created_at`.
- Outcomes:
  - **Transport failure** (timeout, 5xx, connection refused) is retryable. It writes no attempt row; it only updates the chunk's step status with `attempts`, `last_error` and `next_try_at`, using exponential backoff.
  - **Malformed output** is retried up to N times. After that the step is `failed` with the error shown in `status`, and it is never silently treated as empty.
  - **A valid empty extraction** is terminal: outcome `empty`.
- When a chunk's extraction fingerprint changes because its text, the model or the prompt version changed, the new attempt becomes current in the same transaction that supersedes the old one. Superseded attempts' provenance links stop counting toward weights. Because Norman chose no retention, superseded attempts and their links are deleted rather than kept.
- Element instances from the paper (roadmap M2): each relationship found by an attempt is recorded with its description and strength (if the chosen profile emits them) in `extraction_relations` (attempt, relation, description, strength). Each entity found is recorded the same way in `extraction_entities`. Canonical `entities`/`relations` remain deduplicated current objects.
- Hygiene before writing: reject entity names that are sentence-shaped (heuristics as in the 2026-08-06 evidence doc: very long, terminal punctuation, clause shape) and count the rejects in the attempt record.

### Weights (step 4, derived)

- `support(relation) = count of distinct current attempts linking it` (through `extraction_relations`).
- Leiden edge weight between two entities is the sum of support over the relations between them. *Builder's call:* whether to also multiply by extracted strength; default is no, matching the paper's use of instance counts.
- Relations or entities with zero current support are deleted (garbage-collected) in the same transaction that removes their last support.
- No stored weight column is required. If one is added as a cache, it must be recomputed in the same transaction as attempt changes. A test must show that forcing all weights to 1.0 fails (the roadmap M3 mutant).

### Communities (step 5, per store)

- One path for every surface: weighted hierarchical Leiden over the store's current graph, including isolates. It replaces both the CLI `enrich --levels` path and the unused `build_community_hierarchy`. *Builder's call:* use `leiden_hierarchical`'s downward subdivision, the closest to the roadmap's recursive per-parent Leiden, with a fixed seed.
- A **community run** records `run_id, store, graph_fp, config, seed, created_at, modularity`. Communities belong to a run. A new run replaces the old one in one transaction (no more left-over runs).
- When it runs: a store's communities are rebuilt when its `graph_fp` differs from the current run's, **and** that store has no ready chunk-level work. While a big backlog drains, the builder may add a cap (for example, rebuild at most once every N minutes) so that touched chunks show up in communities without rebuilding after every extraction.
- Membership never comes from the LLM.

### Summaries (step 6, per community)

- Bottom-up. A leaf summary's inputs are member entities (names, types, descriptions), the relations between them with weights and descriptions, and a few grounding chunk excerpts. A parent summary's inputs are its children's summaries.
- Prompt contract from the 2026-08-06 prototype:
  - 2–4 sentences;
  - state the theme and name member entities;
  - ground only in the provided material;
  - prefer a shorter honest summary when the evidence is thin;
  - plain prose.
- Mechanical gates: the summary is non-empty, names at least one member entity, and is over 40 characters. A summary that fails a gate is retried, then marked `failed`.
- Summary rows record `fp, model, prompt_version, created_at`. After a community rebuild, a summary is reused when an equal-fingerprint community existed in the previous run (same members and edges), so a stable seed avoids re-summarizing unchanged communities.

## Status and work selection

- New table `step_status(subject_kind, subject_id, step, fp, state, attempts, last_error, next_try_at, priority, updated_at)`:
  - `subject_kind` is `document`, `chunk`, `store` or `community`;
  - `state` is `current`, `failed` or `blocked`;
  - missing means no row, stale means `fp` doesn't match.
- "What to do next" is a query, not a queue. It computes each subject's current input fingerprint for each step, compares it with `step_status.fp`, and orders by `priority DESC`, then age. Since fingerprints depend on configuration (model, prompt version), the worker computes expected fingerprints in code. There's no stored "expected" column that could drift.
- Priority levels:
  - 2 = touched by recall;
  - 1 = newly added;
  - 0 = backlog.
- `graphrag status [--json]` reports, per store and per step: current, stale, missing, failed (with the latest error), the backlog by priority, whether a worker is running, and when llama.cpp last succeeded or failed. It extends the existing `pipeline_status`. Chunk embedding becomes a phase, which it is not today.

## The worker

- `graphrag work [--store S] [--once] [--max-items N]` drains ready work in priority order, then exits.
- Single instance via an advisory lock file in the data directory (`flock`). A second invocation exits right away with status "already running".
- **Triggers:**
  - Any write (CLI `note`, MCP add, Nushell add) records status and spawns `graphrag work` detached if no worker holds the lock. Nushell may skip spawning and rely on the backstop.
  - Recall (CLI `ask`, MCP `recall`/`search`) raises the priority of returned chunks that have incomplete steps and spawns a worker the same way. That write is a single cheap UPDATE and never delays the results.
  - A launchd agent runs `graphrag work` every 15 minutes as a backstop (llama.cpp was down, the machine slept, a spawn failed).
- **LLM down:** after M consecutive transport failures the worker stops with exit code 75 (TEMPFAIL) and records the outage in status. The next trigger or the backstop retries.
- **Concurrency:** llama.cpp reports 4 slots with a 131K context. The worker runs up to 4 LLM calls at once (configurable) and writes to SQLite from a single thread. Cargo builds and the worker shouldn't compete on the 16 GB host; the builder documents how to pause it (`graphrag work --pause` flag file, or unloading the launchd agent).
- **SQLite concurrency:** the database has no WAL or `busy_timeout` today. With a long-running writer and the MCP reader/writer at the same time, enable `journal_mode=WAL` and a busy timeout of several seconds on every connection.

## LLM client

- New `OpenAiChatClient` in `graphrag-llm`: `POST {base}/chat/completions`, `response_format: json_schema` when the profile has a schema, `chat_template_kwargs.enable_thinking` per profile, timeout, no internal retry (the worker owns retry).
- Config: `GRAPHRAG_LLM_BASE_URL` (default `http://ndn.lan:8080/v1`), `GRAPHRAG_LLM_MODEL` (default: the first id from `/v1/models`, recorded in every attempt), `GRAPHRAG_LLM_CONCURRENCY` (default 4).
- `OllamaChatClient` stays for local use and isn't the default.
- MCP sampling extraction is removed. `add_document`/`add_code` store and chunk, then trigger the worker; they no longer call back to the agent. `graphrag_summarize_community`'s agent round-trip goes away the same way.

## Extraction profile

**Decision:** paper-shaped JSON with thinking on (`paper-json-v1`, arm C below).
- Entities: `name`, `type`, `description`. Relationships: `source`, `target`, `relation`, `description`, `strength 1-10`.
- Enforced with llama.cpp `json_schema`.
- `enable_thinking: true`.
- Single block, no sandwich: the 2026-08-03 spike found the closing block neutral-to-harmful for extraction.
- No gleaning in v1. The profile records `glean_max_rounds=0`, so adding gleaning later is just a fingerprint change that re-extracts.

The triple-line strategy (`strategy_for_model`'s default) is retired for this model. That default came from the 2026-08-03 spike on Qwen3.6 and nemotron3:33b over Ollama. Neither result transfers to qwen3.8-27b.

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
- C was chosen because it's the best measured, and the extra time is affordable at this size: about 6 s per chunk at 4 concurrent, so roughly 3 hours for the whole corpus once.
- A's failure modes, as the judges described them:
  - generic nouns and clause-like entities ("use JWT tokens for session management");
  - hub-and-spoke invented edges, such as a 45-entity star around one commit;
  - blank or "0" entities;
  - one unparseable output.
- Shared defects to fix in the prompt or the type list:
  - every arm typed `jj` (a VCS) and `Opus` (a model) as Person;
  - every arm treated illustrative example handles (`alice`) and citations as real entities;
  - C sometimes uses vague `related_to` hub edges.
- One chunk (190) correctly produced an empty extraction in all arms. Valid-empty has to stay a terminal outcome.

Spike artifacts, which aren't kept in the repo: `scratchpad/spike/` (`run.py`, `pack.py`, `score.py`, packets, both judges' JSON).

## Migration (clean slate)

All steps run on a copy first (rehearsal), then on live after a fresh backup.

1. Back up the DB and index files (as on 2026-10-18).
2. Schema migration: add `documents`, the new chunk columns, attempts, `step_status`, community runs, and summary metadata. Enable WAL.
3. **Delete** all entities, relations, extraction rows, provenance links, entity embeddings and merges, and communities (Norman: no retention).
4. **Delete non-note chunks** as listed under "Data cleanup", with count assertions.
5. Turn every remaining chunk into a document, then re-chunk it under the new profile. Chunks that were already small map 1:1.
6. Let the worker run: embed what's missing or stale, extract everything (priority 0), then communities and summaries.
7. Verify with `status`: zero missing, stale or failed. The atlas shows varied weights, and communities are no longer mostly singletons.

## Testing

All without a live LLM unless noted. Release builds.

- Fake `ChatClient` with canned responses. The full pipeline on a seeded store reaches all-current, and a second run does zero work.
- Fingerprint invalidation: changing one paragraph of a note re-extracts only the affected chunks. Changing the prompt version re-extracts everything. Changing the model changes the attempt fingerprint.
- Kill-and-resume: abort the worker mid-batch and rerun. The final state equals an uninterrupted run, and no attempt is duplicated or half-applied.
- Transport failure stays pending with backoff. Malformed output retries, then becomes `failed`. Valid empty becomes `empty`.
- Weights: two chunks stating one relation give support 2. Re-applying the same attempt doesn't change support. Superseding an attempt removes its support. The mutant forcing weight to 1.0 fails a test.
- Escalation: a recall hit on an unextracted chunk makes it the next item processed.
- Locking: two concurrent `graphrag work` processes, and one exits immediately.
- `OpenAiChatClient` against a mock HTTP server, including the json_schema and thinking flags.
- Live rehearsal on a DB copy against llama.cpp before touching live.
- Paper M2/M3 gated capabilities stay gated unless their binding tests are actually met. This design delivers parts of both.

## Data cleanup (approved and done 2026-10-19)

All of the following were checked against the live DB on 2026-10-19.

1. **Every `session:` chunk in `conversations` (218 chunks) is development debris from MCP testing on 2026-01-14, created between 00:01 and 01:33:**
   - 191 are saved extraction responses: they contain `<think>`, `"head":`, or start with `Extract from the following text`. They total about 1.17M characters and include the 442K chunk.
   - The other 27 are test inputs ("Another test 1", "Stream test 3", "Test message 2 about Rust and nushell", "Say hi") or generated `Entities: ... Summary:` stubs.
   - **Selector:** `store='conversations' AND source LIKE 'session:%'`.
2. **The GraphRAG paper was saved into `conversations` as well.** 14 chunks there duplicate the `graphrag-paper` store. **Selector:** `store='conversations' AND source LIKE 'graphrag-paper%'`.
3. **Other test data:**
   - the `test` store (5 chunks);
   - chunk 238 (`source='test-ingestion'`, a Rust hello-world);
   - chunk 255 (`brain`, content `test: smoke`).

Norman approved deleting all three groups. They were deleted from the live DB on 2026-10-19, ahead of the migration, in one transaction that checked these exact counts (218, 14, 5 + 2 = 239 chunks), along with the `test` store row and one orphaned chunk embedding (chunk 526). Backup: `backups/2026-10-19-pre-conversation-cleanup/graphrag.db`. Migration step 4 is therefore a no-op on live, but it keeps the counted check for rehearsal copies. Afterwards `conversations` keeps 384 chunks: the 379 real `brain` notes, 2 `prime-session` notes and 3 `let-go/*` notes.

## Calls made after the builder check (Claude's calls; Norman can override)

1. **The extraction fingerprint keeps `chunk_id`** (the M0 type is unchanged). Chunk ids are stable. Re-chunking keeps the existing id of any chunk whose `content_fp` is unchanged, so unchanged text keeps its id and its extraction. During migration, chunks that don't need splitting keep their id. Only oversized chunks get new ids, and those are re-extracted anyway in the clean slate.
2. **The model is pinned, not discovered.**
   - `GRAPHRAG_LLM_MODEL` defaults to `qwen3.8-27b`.
   - At startup the worker checks that `/v1/models` serves that id. If it doesn't, the worker exits with status 69 (UNAVAILABLE) and reports the mismatch. It does not re-extract against whatever happens to be loaded.
   - Changing the configured model or prompt version is the explicit act that makes extraction stale. `status` then shows how many chunks a change would re-extract before any work starts, and `graphrag work` asks for `--accept-reextract` when more than 10% of a store would be superseded.
3. **Failure semantics.**
   - States: `current`, `failed` (retries exhausted; the error is kept), `blocked` (an upstream step for the same subject isn't current; for example extraction waits on chunking).
   - A store is **complete** when it has zero missing or stale. `failed` is reported separately and doesn't block communities: failed or backing-off chunks don't count as ready work for the community gate.
   - Failed items are retried automatically when their fingerprint changes (new prompt or model), or on demand with `graphrag work --retry-failed`.
4. **The prompt lives in the repo.**
   - The spike's arm-C prompt and schema are committed as the `paper-json-v1` profile in `graphrag-llm`.
   - It gets the fixes the judges found: guidance on Software and model/tool types (so `jj` and `Opus` aren't typed as Person), ignore illustrative example names and citations, and avoid generic `related_to` edges.
   - Because those fixes weren't measured, **the corrected prompt is re-spiked** on the same 12 chunks with the same two judges before the live run. Gate: it scores at least as well as arm C on both judges, and the shared defects are gone.
5. **Summaries:**
   - Every community with at least 3 member entities, at every level, is summarized. Smaller communities are marked `skipped_small`, which is not an error and counts as complete.
   - Hierarchy depth: recursive subdivision until Leiden stops splitting, capped at 4 levels.
   - `status` reports the summary call count before a drain starts.
6. **Relation identity:**
   - Directed `(head entity, tail entity, canonical_relation)` within a store. Entities are matched on case-folded, trimmed names, as `tx_get_or_create_entity` does today.
   - Descriptions are kept per instance (per attempt). A canonical node or edge description is the distinct instance descriptions, newest first, packed to a fixed budget.
   - LLM-aggregated descriptions (roadmap M2) are deferred. Summary fingerprints hash the packed descriptions actually used.
7. **Code stays on the code chunker.** `add_code` uses `code_chunker` (the chunker kind is part of the chunk fingerprint). The live DB has no code stores beyond the test chunk being deleted, so this changes nothing in the migration.
8. **Worker location:**
   - The worker loop is a library function in a new `graphrag-worker` module, which depends on core plus graphrag-llm, so core stays provider-independent. `graphrag work` is its CLI entry point.
   - Spawners exec `$GRAPHRAG_BIN`, then `graphrag` on PATH, then `~/.cargo/bin/graphrag`. If none is found, they rely on the launchd backstop.
   - MCP and CLI spawn. Nushell and Python don't; they rely on the backstop.

## Left to the builder

- Table and column names; how much of the existing `extractions` table to reshape versus replace.
- Exact retry counts, backoff, and the outage threshold.
- Whether communities rebuild on a timer during a large backlog.
- The launchd plist details and the pause mechanism.
- Fragment-entity heuristics, tuned against the spike outputs.
