# SP1 evidence: RLM orchestration prototype + SLM prompting findings

**Date:** 2026-08-06
**Status:** Evidence note (uncommitted draft for review)
**Relates to:** `docs/2026-06-20-enrichment-engine-design.md` (SP1)
**Sources:** live prototype run against the production store (backup taken:
`graphrag.db.bak-2026-08-06`); measured findings transcribed from
`~/development/engineering-notebook/src/oneshot-summarize.ts` and `summarize.ts`.

## 1. Prototype run: community summarization via agent orchestration

Executed 2026-08-06 against `~/Library/Application Support/graphrag/graphrag.db`
(stores: conversations=517 chunks, 9front-kb, test; 88 entities, 108 relations,
63 communities of which 1 had a summary).

Pipeline (the shape SP1 should implement in Rust):

1. **Pack extraction** (SQL): per unsummarized community — member entities
   (name/type/properties), all incident relations rendered as
   `head -[rel]-> tail`, up to 6 grounding chunk excerpts (800 chars each).
   All 62 packs had chunk evidence.
2. **Concurrent dispatch**: 2 workers x 31 packs. Prompt contract: 2-4 sentence
   summary; state theme + name member entities; ground ONLY in provided
   relations/chunks; never invent; "shorter honest summary" on thin evidence;
   plain prose, no markdown. Output: JSON map community_id -> summary with
   worker-side keyset self-validation.
3. **Mechanical verification gates** (cheap, all passed):
   - keyset exactness per batch (62/62 covered, no extras)
   - all summaries non-empty
   - grounding audit on a deterministic sample: item selected iff
     `blake2b("community:"+id) mod 31 < 5` (7/62 selected); pass = summary
     names >=1 member entity and length > 40 chars. 7/7 passed.
4. **Write**: parameterized `UPDATE communities SET summary=? WHERE id=? AND
   (summary IS NULL OR summary='')`. Result: 1 -> 63/63 summarized.
5. **End-to-end falsifier**: `graphrag_query_global` over MCP stdio went from
   no content to returning 60 community contexts for map-reduce on a real
   cross-community question.

Findings for the Rust implementation:

- The prompt contract above is sufficient for grounded output on packs as thin
  as 2 entities + 1 relation. Workers honestly flagged thin/artifact packs
  rather than padding (18 + 16 flagged across batches).
- Validation must be mechanical and pipeline-owned (keyset, non-empty,
  member-grounding, deterministic sampling), not prompt-hoped.
- **Upstream defect surfaced**: extraction emits sentence fragments as entities
  ("More work is needed", "performance generalizes to datasets...") from the
  graphrag-paper ingest, producing degenerate 2-member communities.
  Summarization cannot repair extraction; fix belongs in `strategy_for_model`
  extraction filtering (WIP at c2e9bf5). 39/62 communities had only 2 members —
  partly a symptom of this.
- Batch economics: 31 packs/worker completed comfortably in one worker context;
  membership stayed structural (Leiden) — the LLM never decides membership.

## 2. Transferred SLM findings (engineering-notebook, measured)

Provenance: 4-arm controlled spikes, N=10 samples, dual judge (Claude haiku-4-5
primary + Codex gpt-5.5), 20-point scale, eviction-corrected wall times.
Models: Qwen3.6, nemotron3:33b on local Ollama.

**Adopt for SP1 cheap-model calls:**

- **Sandwich anchoring** (full instructions BEFORE and AFTER the content):
  eliminated continuation-drift (model "picking up the conversation" instead of
  summarizing) across 4K-128K transcripts on Qwen3.6 + JSON-schema; on
  nemotron3:33b + block format the effect shrank but stayed positive
  (15.6 -> 15.1 without it). One extra prompt block; keep it.
- **`think:false` on Ollama**: 13-22x speedup (1.6s vs 28s @4K; 2.6s vs 40s
  @128K), no observed quality loss. Caveat: Ollama then drops `required`-field
  enforcement in JSON schema -> mirror required fields in the prompt body and
  parse tolerantly. Belongs in the Ollama strategy, not call sites.
  (Thinking-on bought +0.6 accuracy but -0.5 skip judgment and 2-3x wall.)
- **Deterministic pre-slicing beats LLM filtering**: given multi-day input,
  nemotron conflates dates. Slice upstream so the model only sees in-scope
  content. graphrag analog: feed exactly the community's chunks (the pack
  model); never ask the SLM to filter or decide membership.
- **Few-shot from the user's own corpus** (+0.9): examples must come from real
  stored entries, not canned text — the model adopts house style.
- **Introspection-derived rules** (+0.8): when SLM and strong-judge disagree,
  ask the SLM why; in 3/5 cases it articulated the missing rule
  ("documentation-only work is a skip"), which then closed the gap encoded
  explicitly.

**Negative results (equally load-bearing — do not re-add):**

- Reasoning preambles ("first think through..."): no effect.
- Scope rules citing paths: became hallucination anchors, regressed 15.6 -> 14.4.
- Anti-skip-happiness caveats: no measurable effect.
- Lesson: minimal prompts; every rule must earn its place via a controlled spike.

**Reference quality ladder** (daily-summary task, nemotron3:33b local):
v1 sandwich+JSON 14.8 -> +few-shot 15.7 -> +date-slicing 15.4 (-36% wall) ->
v2 (+doc-skip rule) 15.6 best-skip; Claude haiku reference 17.3 at ~3.5x wall
and API cost. An SLM at ~90% of haiku quality for zero marginal cost is the
SP1 economic case, provided the eval harness exists to keep it honest.

## 3. Implications for the SP1 design

1. `ChatClient` strategy layer should own: sandwich assembly, think:false +
   tolerant parsing (Ollama), and per-model prompt format — consistent with the
   `strategy_for_model` direction already in c2e9bf5.
2. The enrichment driver should ship with the mechanical gates from section 1;
   summaries that fail grounding get re-queued, not hand-fixed.
3. Build the spike/eval harness (4-arm, N=10, dual judge, deterministic
   blake2-mod sampling) as part of SP1, not after: model choice and every
   prompt rule should be spike-validated the way engineering-notebook did.
4. Extraction quality gates before community detection: reject fragment
   entities (heuristics: sentence-like length, verb-phrase shape, terminal
   punctuation) so Leiden stops minting degenerate communities.

## 4. Open items

- Community membership: design intent exists (memory-api notes' global lane —
  "community updates, summaries, cross-note synthesis" — governed by the
  urgency meter; enrichment design keeps membership structural via Leiden),
  but the incremental-membership tool is NOT implemented yet: nothing today
  assigns newly ingested notes/entities to existing communities or decides
  when a community is stale enough to re-run Leiden and re-summarize. This is
  the next tool to build after the summarization loop. Note: this evidence doc
  now supplies the measured backing (engineering-notebook spikes) for the
  sandwich-prompting section already in the SP1 design.
- `/tmp/rlm-spike/FINDINGS.md` (referenced from oneshot-summarize.ts) has
  expired with tmp-rot; rescue any surviving spike artifacts into version
  control.
- Decide fingerprint standard for cross-language audit-sample interop
  (Python blake2b today; xxh3 only if Rust side standardizes on it).
- Re-run `enrich --extract` after extraction filtering lands, then re-summarize
  affected communities (summaries carry no staleness marker yet — consider a
  `summarized_at` / input-fingerprint column).

- Byte-1 probe findings (2026-08-06, authoring-time discovery prototype):
  - `graphrag ask` passes raw text to leit's lexical query parser, which errors
    on operator-like tokens (`--stat`, bare `:`): "leit search: query error:
    failed to parse query". Fix candidate: escape or term-mode fallback in ask.
  - `find_merge_candidates` silently returns "no candidates" when
    `entity_embeddings` is empty (it was: 0 rows until graphrag_embed_entities
    ran — 78 entities, 0 errors). Empty-prerequisite should be an error, not a
    quiet no-op (silent false-negative class).
  - Merge-candidate quality at threshold 0.85: four exact case-fold dup pairs at
    1.000 (case-normalize in extraction would prevent these), judgment band
    0.87-0.97, related-not-identical below ~0.85. Sane defaults: review at
    >=0.85, auto-merge only >=0.99.
  - CLI `vec` badge prints cosine DISTANCE (usearch MetricKind::Cos, lower =
    closer) with no unit hint — consider labeling `dist=` to prevent
    similarity misreads (one occurred during prototyping).
