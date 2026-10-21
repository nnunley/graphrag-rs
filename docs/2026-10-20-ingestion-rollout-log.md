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

132 frozen queries; recall@10 / MRR@10, p50 latency.

| Arm | ALL | known_item | paraphrase | identifier | cross_group | p50 |
|---|---|---|---|---|---|---|
| v1 baseline (old binary, per-store, merged) | 0.723 / 0.611 | 0.900 / 0.710 | 0.562 / 0.449 | 0.775 / 0.468 | 0.525 / 0.956 | 4.8 s |
| v2 after migration, before extraction (all lanes) | 0.697 / 0.637 | 0.950 / 0.831 | 0.719 / 0.497 | 0.525 / 0.403 | 0.500 / 0.942 | 0.96 s |
| v2, vector lane only | 0.678 / 0.608 | 0.925 / 0.803 | 0.781 / 0.618 | 0.450 / 0.307 | 0.475 / 0.806 | 0.53 s |
| v2, lexical lane only | 0.564 / 0.569 | 0.900 / 0.751 | 0.500 / 0.407 | 0.300 / 0.300 | 0.525 / 1.000 | 0.63 s |

Section-bounded chunking improves known-item and paraphrase recall and
MRR, and recall is about 5× faster. Identifier recall drops from 0.775 to
0.525: BM25 over smaller chunks, fused with a vector lane that is weak on
identifiers. Cross-group recall is unchanged without extraction (the graph
lane needs entities). Both are re-measured after the rehearsal extraction.
