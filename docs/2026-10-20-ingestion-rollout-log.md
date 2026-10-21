# Ingestion rollout log

Plan: `docs/2026-10-19-ingestion-workflow-plan.md`, Task 20. Everything here
ran on `.backup` copies of the live DB; the live DB is untouched. Only
aggregate numbers are recorded here (the repo is public; the eval queries
quote private notes and live next to the DB, not in git).

## 2026-10-20

### Re-spike (Step 1)

Twelve source chunks (3 each from operation, conversations, mica-docs,
graphrag-paper), each re-cut by the production chunker; pieces extracted with
qwen3.8-27b on llama.cpp and merged per source chunk. Blind judges: a fresh
Claude Opus subagent and Codex `gpt-6-sol`; 0–5 on grounding, recall, entity
quality and usefulness (max 20).

| Prompt | Claude: C_prod / prod / small | Codex: C_prod / prod / small | Mechanical gate |
|---|---|---|---|
| v1.1 | not judged | not judged | fail: Opus typed Person; `#alice`/`#bob` from a code sample |
| v1.2 | 14.3 / 12.9 / 13.4 | 14.6 / 13.6 / 13.2 | pass |
| v1.3 | 14.2 / 13.2 / 13.5 | 14.2 / 12.9 / 13.6 | pass |

`C_prod` is the 2026-10-19 arm-C prompt at production size. It still types
`jj` and `Opus` as Person and keeps example handles, and it has no class
fields (`kind`, `is_class`, `broader`). The shipped prompt (v1.3) fixes those
but scores about one point lower on both judges, mostly recall and
usefulness. Distinct classes per 100 entities: v1.3 prod 60.5, small 46.7.

**Gate: failed** (the shipped prompt must score at least arm C on both judges).
Two prompt iterations didn't close the gap. The choice goes to Norman before
the rehearsal extraction runs.

### Migration rehearsal (Step 2, local part)

`graphrag migrate v2 --plan` on a copy: 0 unresolved.

| Group | From files | Reassembled | Single |
|---|---|---|---|
| conversations | 0 | 0 | 386 |
| mica-docs | 53 | 0 | 0 |
| operation | 21 | 3 | 159 |
| 9front-kb | 0 | 0 | 3 |
| papers | 1 | 0 | 0 |

Deleted 4,435 entities, 4,836 relations, 1,043 communities and 760
extraction checkpoints; 1,556 chunks replaced; 124 paper copies dropped.
`--apply` took 2.8 s: 626 documents, 1,947 chunks (longest 1,999 characters),
1,193 sections, 536 embeddings re-attached, 848 old documents removed.
Embedding the other 1,411 chunks took 31 min of CPU.

### Retrieval eval (Task 19A)

132 frozen queries. 13 of them have anchors that exist only in v1 chunk text
(12 identifier, 1 paraphrase): those mica/operation docs changed on disk since
the old ingest, and the migration re-imports the current files. The table
scores the 119 queries whose anchors exist in both corpora; recall@10 / MRR@10.

| Arm | ALL | known_item | paraphrase | identifier | cross_group |
|---|---|---|---|---|---|
| v1 baseline (old binary, per-store, merged) | 0.718 / 0.653 | 0.900 / 0.710 | 0.581 / 0.464 | 0.750 / 0.563 | 0.525 / 0.956 |
| v2 after migration, before extraction (all lanes) | 0.773 / 0.707 | 0.950 / 0.831 | 0.742 / 0.513 | 0.750 / 0.576 | 0.500 / 0.942 |
| v2, vector lane only | 0.752 / 0.675 | 0.925 / 0.803 | 0.806 / 0.638 | 0.643 / 0.438 | 0.475 / 0.806 |
| v2, lexical lane only | 0.626 / 0.631 | 0.900 / 0.751 | 0.516 / 0.420 | 0.429 / 0.429 | 0.525 / 1.000 |

- Before any extraction, v2 already beats the baseline overall. The gain comes from known-item and paraphrase queries; identifier recall is unchanged.
- The lexical lane alone is weak on identifiers (0.43). Tokenization of `::`, paths and flags is the likely cause, to check. Fusion makes up for it.
- Cross-group recall can't improve until extraction feeds the graph lane.
- Latency isn't compared: the arms ran under different machine load.
- A `.v2` query set, generated from the migrated corpus, is needed before comparing extraction arms.
