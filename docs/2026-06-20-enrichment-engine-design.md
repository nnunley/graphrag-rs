# Enrichment Engine — Design (SP1)

**Date:** 2026-06-20
**Status:** Approved design, ready for implementation plan
**Repo:** graphrag-rs
**Related:** `docs/2026-06-01-memory-api-plan.md` (durable work queue — deferred to SP1b);
fleet multi-agent substrate architecture note (agent-sandbox `docs/plans/2026-06-20-fleet-multi-agent-substrate-architecture.md`)

## Problem

`graphrag-rs` already implements every GraphRAG *algorithm* — chunking (text + tree-sitter
code), embeddings, LLM entity/relation extraction, hierarchical Leiden, HNSW, BM25, RRF hybrid
fusion, graph expansion, SQLite persistence, plus a CLI and a 14-tool MCP server. What is
missing is **orchestration**: the pipeline that drives a large batch of small extraction and
summarization LLM calls to turn a populated store into a fully enriched, queryable graph.

Concretely, three gaps:

1. **No automatic community summarization.** `update_community_summary` (db) and the
   `graphrag_summarize_community` MCP tool exist, but nothing drives summarization in a loop,
   so the global-search layer has **no content**. (MCP sampling is scaffolded but not wired.)
2. **LLM access is Ollama-only.** There is a clean one-method `ChatClient` trait
   (`complete(&ChatPrompt) -> Result<String, String>`) but only `OllamaChatClient` implements it.
3. **Extraction is sequential, per-chunk.** No concurrency, no provider-agnostic dispatch.

## Goal

Given a store already populated with chunks + embeddings (via `note` / `add_document` /
`add_code`), produce a fully enriched graph — entities/relations + Leiden communities +
**community summaries** — driven by **concurrent cheap-model calls through an OpenAI-compatible
endpoint** (the fleet's `llm-proxy`), **idempotent on re-run**.

Success: `graphrag enrich --extract --summarize --provider-url <proxy> --model <cheap>` on a
populated store yields entities + relations + hierarchical communities + non-empty community
summaries; `graphrag_query_global` / `ask` return summary-backed context; a re-run is a fast
no-op for completed work; a mid-run failure is recovered by re-running.

## Scope

**In scope**
- `OpenAiChatClient` — new `ChatClient` impl (OpenAI `/v1/chat/completions` wire format).
- Concurrent batch entity/relation extraction (replaces the sequential loop).
- **Automatic community summarization** stage.
- Idempotent skip + `--force`.
- `enrich` CLI flags.
- Sandwich-structured prompts for both extraction and summarization (small-model accuracy).

**Deferred (not this spec)**
- Durable, resumable work queue → SP1b (the `docs/2026-06-01-memory-api-plan.md` plan).
- MCP sampling wiring (the CLI path does not need it).
- Incremental/delta indexing (full re-run per store; idempotent skip suffices).
- Retrieval-as-context-layer integration (SP2), KV/prompt caching (SP3).
- Native Anthropic/multi-provider clients (the proxy brokers providers; the client speaks one
  wire format).

## Dispatch topology (decided)

One **OpenAI-compatible `ChatClient`** with a configurable `base_url` + `model`, pointed at the
fleet's `llm-proxy`, which brokers credentials and routes to a cheap model. **No raw provider
keys live in graphrag-rs** (the proxy injects them). For local dev, `base_url` can point
directly at Ollama / OpenAI / ollama-cloud. Concurrency is graphrag-rs's own bounded tokio
semaphore — **not** container/fleet dispatch (a container per sub-second call is the wrong tool).

## Components

### 1. `graphrag-llm`: `OpenAiChatClient`
Implements the existing `ChatClient` trait. Config: `base_url`, `model`, optional `api_key`
(normally empty — proxy injects), request `timeout`, `max_retries`. reqwest-based. Emits
OpenAI chat-completions requests; where the model supports it, uses JSON-schema-constrained
output (`response_format`); otherwise instruction-only + parse-and-retry. `OllamaChatClient`
remains for local use.

### 2. `graphrag-core`: `enrich` orchestrator (new module)
Three stages over a store, each idempotent:

- **Extract.** Load chunks that still need extraction (see Idempotency) → run up to N concurrent
  `complete()` calls (bounded semaphore) → parse `EntityTriple`s → upsert entities/relations.
  Prompt uses the **sandwich** structure (below) + the existing reflection/continuation passes
  (max 2). `temperature 0`.
- **Communities.** Run the existing hierarchical Leiden (already wired into `enrich`).
- **Summarize.** Walk communities **leaf → root** (so a parent's prompt can include child
  summaries) → for each community lacking a summary, build a sandwich-structured summary prompt
  from its entities/relations/child-summaries → concurrent `complete()` → persist via
  `update_community_summary`.

### 3. `graphrag-cli`: extend `enrich`
New flags: `--summarize`, `--provider-url <url>`, `--model <name>`, `--concurrency <N>`,
`--force`. Existing `--extract` and Leiden flags are retained.

## Prompt design — "sandwich prompting" (small-model accuracy)

Lead and close with the **same** instruction+schema block, content in the middle, so a small
model does not drift off-task after a long content span (primacy + recency reinforcement):

```
[INSTRUCTION + SCHEMA BLOCK]   ← task, canonical entity/relation types, JSON schema, 1–2 few-shot examples
[CHUNK CONTENT]                ← the text to extract from
[INSTRUCTION + SCHEMA BLOCK]   ← same block restated ("now emit ONLY JSON matching the schema above")
```

This is implemented as an **isolated, unit-tested prompt-builder** so the ordering is tunable
and verifiable independently. The same sandwich structure is reused for community
summarization (format/instructions before *and* after the entity/relation/child-summary
payload). Canonical types come from the existing `canonical_entity_types` / `canonical_relations`.

## Data flow

```
populated store (chunks + embeddings)
  → extract   (chunks → ChatClient[concurrent] → entities/relations in SQLite)
  → Leiden    (graph → hierarchical communities)
  → summarize (communities[leaf→root] → ChatClient[concurrent] → summaries in SQLite)
  → enriched store  (ask / graphrag_query_global return summary-backed context)
```

## Idempotency

- **Extraction:** add a nullable `extracted_at` timestamp column to `chunks` (schema change is
  acceptable — we are the only consumers). A chunk with `extracted_at` set is skipped unless
  `--force` (which clears/ignores it). Set `extracted_at` only after the chunk's
  entities/relations are committed.
- **Summarization:** a community is "done" when its `summary` is non-empty (already in schema);
  `--force` re-summarizes.

## Concurrency & error handling

No durable queue, so the model is **tolerate + report + resume-by-rerun**:

- Bounded tokio semaphore, size = `--concurrency` (conservative default, e.g. 8; expected to be
  tuned per endpoint/model since small-model endpoints throttle and sandwich prompts add tokens).
- Per-call `timeout` + bounded retry (e.g. 2×, backoff) on 429 / 5xx / timeout.
- Persistent per-item failure → log, count, **skip that item and continue** (never abort the run).
- Malformed LLM JSON → existing reflection/continuation passes; still bad → skip + count.
- End-of-run summary line, e.g. `extracted 980/1000 (18 tempfail, 2 dataerr); summarized 47/47`.

### Exit codes (sysexits.h convention)

| Exit | Meaning | Re-run helps? |
|---|---|---|
| `0` | Complete — all chunks extracted, all communities summarized | n/a |
| `64` (USAGE) | Bad flags / no store / unreadable config | no — fix invocation |
| `69` (UNAVAILABLE) | Proxy/provider unreachable or auth rejected (401/403) | no — fix endpoint/creds |
| `75` (TEMPFAIL) | Partial: transient failures (429/5xx/timeout) after retries | **yes — re-run resumes** |
| `65` (DATAERR) | Pervasive unparseable model output / schema mismatch | maybe — change model/prompt |
| `70` (SOFTWARE) | Unexpected internal error | depends |

## Testing (no live LLM)

- **Unit:** `OpenAiChatClient` request/response shape against a mock HTTP server; entity-JSON
  parsing; the sandwich prompt-builder ordering; idempotency skip logic.
- **Integration:** a **fake `ChatClient`** with canned extractions/summaries → run full `enrich`
  on a seeded store → assert entities/relations/communities/summaries populated; re-run →
  assert no duplicate work; inject a fake failure → assert the run continues, reports, and a
  re-run completes the remainder with the correct exit code.
- **Concurrency:** `--concurrency > 1` against the fake → race-free, correct results.

## Risks / open items

- Cheap models vary in JSON-schema-constrained-output support; the instruction-only +
  parse-retry fallback covers models without it. Validate the default model early.
- Default `--concurrency` is a guess; tuning is expected (and is the reason it is a flag).
- `extracted_at` is the only schema change; a migration/`ALTER TABLE` is needed for existing DBs.
