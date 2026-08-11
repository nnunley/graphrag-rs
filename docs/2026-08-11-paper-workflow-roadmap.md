# Roadmap: complete the paper GraphRAG workflow

**Scope:** Microsoft Research, *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, arXiv:2404.16130v2, especially §3.1 and Appendices A, C, and E.  
**Primary source:** [repository copy](microsoft-graphrag-paper.md) and [arXiv HTML](https://arxiv.org/html/2404.16130v2).  
**Code baseline:** `graphrag-rs` commit `b721fcf`.  
**Target:** a provider-independent, notebook-drivable implementation of the paper's offline index and online global-query workflow, with enough evaluation to state precisely where it does and does not reproduce the paper's findings.

This roadmap does **not** redefine current hybrid chunk retrieval as paper global search. The existing `graphrag ask` command performs vector/BM25 retrieval over source chunks. Paper global search exhaustively maps over every community report at one selected hierarchy cut (the paper C-level view), scores partial answers, and reduces the useful answers once.

## Paper contract

```text
OFFLINE INDEX
source documents
  -> token-bounded overlapping chunks
  -> entity / relationship instances (+ descriptions, strength; claims optional)
  -> reconciled entities + aggregated descriptions + evidence-weighted edges
  -> hierarchical Leiden partitions
  -> reference-bearing report for every community at every materialized level

ONLINE GLOBAL QUERY
query + one selected community level
  -> seeded shuffle of every report at that level
  -> fixed-token report batches
  -> parallel partial answers + helpfulness scores
  -> discard score 0, rank remaining answers, token-pack
  -> one reference-preserving global answer
```

A paper-complete implementation must also compare this path with vector semantic search (SS) and identical map-reduce over shuffled source text (TS). Local Search, DRIFT, dynamic mixed-level retrieval, and incremental indexing are valuable later Microsoft GraphRAG features, but are not prerequisites for this paper milestone.

## Current state at `b721fcf`

| Paper stage | State | Evidence and gap |
|---|---|---|
| Documents -> chunks | Partial | Text/Markdown/code chunkers produce spans, but persisted chunks do not durably retain document/span identity. MCP document ingestion chunks, while CLI `note` stores a whole input as one chunk and CLI `chunk` only renders chunks. Sizing is character/byte-oriented rather than an injected token-count contract; the paper's 600-token/100-token-overlap setting cannot yet be reproduced exactly. |
| Chunks -> element instances | Partial | `plan_extract`/`apply_extract` produce typed triples with exact nonce provenance. Entity descriptions, relationship descriptions/strength, claims, domain prompt profiles, and general plan/apply gleaning are absent. MCP alone has a reflection loop. In the core/CLI plan-apply path, malformed extraction currently parses as a terminal zero-yield checkpoint, making parser failure indistinguishable from a valid empty result. |
| Instances -> graph | Partial | Folded exact-name entity reuse, deduplicated global relations, response hashes, and exact extraction links exist. The database does not retain paper-shaped immutable element instances, aggregate their descriptions, or derive edge weight from repeated relationship evidence. |
| Graph -> communities | Partial | Two paths disagree: CLI `enrich --levels` calls downward hierarchical Leiden (defaulting to one flat level), while tested `build_community_hierarchy` aggregates upward to a synthetic root and has no production caller. Both project stored relations at weight `1.0`. The project must select one paper-compatible path, wire every surface to it, consume evidence weights, and enforce the recursive hierarchical-Leiden/MECE contract. |
| Communities -> reports | Partial | Bottom-up `plan_summarize`/`apply_summarize` exists. Leaf prompts contain entity names only; higher prompts contain child summary strings. There is no edge/claim evidence, degree prioritization, token packing, child-report substitution, structured report, or grounded record references. |
| Reports -> global answer | Missing | MCP `query_global` is only a delegation scaffold: it returns communities from all levels (optionally truncating database order) and asks the caller to orchestrate work. There is no single-level exhaustive selection, report shuffle/batching, scored map answers, zero filtering, ranked reduce, or final query trace. `graphrag ask` is unrelated raw-chunk hybrid retrieval. |
| Paper evaluation | Missing | No shared-budget C0-Cn/TS/SS harness, global-question set, blind pairwise judging, claim analysis, cost report, or paper-claim gate exists. |

The live corpus being searchable does not close these gaps: the recent RLM dogfood validated ingestion, exact extraction provenance, and raw-chunk hybrid retrieval, not community-report global search.

## Architectural rules

1. **Keep Rust provider-independent.** Core owns plans, validation, token-packing rules, fingerprints, and persistence. Python/notebooks or CLI adapters own model calls, concurrency, retry, and provider choice.
2. **Make every expensive result reproducible.** Persist model, prompt/schema version, input fingerprint, token counts, and exact source IDs. Reject stale or cross-input results before writes.
3. **Preserve instances before aggregation.** Canonical entities and edges are current graph objects; paper element instances are immutable evidence. Aggregation must never erase which extraction/chunk supplied a description or relationship.
4. **Use evidence counts as weights.** Physical duplicate relation rows are not weights. Derive graph weight from distinct accepted relationship instances/provenance links.
5. **Make token budgets explicit.** Introduce a tokenizer/counting contract. Do not silently substitute bytes or characters where the paper requires token-window packing.
6. **Keep global query exhaustive.** One run scans every report at one selected hierarchy cut. Top-k report retrieval is a different mode and must not be called paper global search.
7. **Seed randomness and retain traces.** Paper shuffling remains, but every plan records its seed, ordered report IDs, batch boundaries, helpfulness scores, and reduce inputs.
8. **Test the negative path.** Each phase needs positive, negative, property, migration, mutation, and real-corpus checks before it can feed the next phase.

## Delivery sequence

### M0 — Freeze the executable paper contract

**Purpose:** prevent a plausible-looking implementation from quietly changing the method.

Deliverables:

- Add a small, checked-in corpus fixture with repeated entities/relationships, one isolated entity, at least two hierarchy levels, one oversized community, and two global questions.
- Define only the minimal versioned boundaries needed to avoid rework: token counts, extraction-attempt identity, report-context rows, map/reduce results, and query traces. Defer full instance/report storage schemas to M2/M4.
- Define a `TokenCounter`/tokenizer boundary usable from Rust and Python; record tokenizer identity in plans.
- Define one canonical core + notebook/JSONL ingestion path for the paper milestone. Cross-surface convergence is production hardening, not an M0 blocker.
- Add a selectable, recorded paper benchmark profile: 600-token chunks, 100-token overlap, and 8k-token report/map/reduce call windows. Keep these values configurable rather than architectural constants.
- Add an executable end-to-end conformance test covering source chunks -> reports -> global answer with fake deterministic executors. Keep unimplemented stage assertions ignored/feature-gated, then enable them milestone by milestone so default CI remains green.
- Record the paper's non-negotiable invariants in staged conformance tests: exact source attribution, relation evidence weights, MECE cuts, report coverage, exhaustive single-cut scan, score-zero filtering, and one reduce.

Exit gate:

- The fixture, benchmark profile, and minimal contracts are reviewed.
- The green conformance skeleton identifies explicitly unimplemented stage capabilities; each later milestone removes its corresponding gate/ignore.
- Mutation cases for top-k substitution, dropped communities, stale fingerprints, lost weights, and extra reduce calls are encoded next to the capability they will activate; no default test pretends an unimplemented stage works.

### M1 — Ship a thin global-query spine over current summaries

**Purpose:** implement the paper's defining missing online path early, before improving report quality.

Deliverables:

- `plan_global_query(store, query, level, token_budget, seed)` selects every summarized community at exactly one level, seeded-shuffles reports, and creates token-bounded map units.
- Map results contain query nonce, batch/report IDs, answer, helpfulness `0..=100`, model, response hash, and report-ID references.
- Reduce planning drops score `0`, sorts descending by helpfulness with deterministic tie-breaking, packs within budget, and emits one final-answer unit.
- For legacy summaries, the query plan captures an ephemeral `sha256(community_id, level, summary)` snapshot and validates it at apply time. This prevents mid-query drift but does not claim graph-input freshness or grounded lineage; hierarchy-run/report fingerprints arrive in M3-M4.
- Apply validates query/report snapshot hashes and returns a traceable final result; query answers remain ephemeral unless explicitly saved as a capsule.
- Core plus one notebook/JSONL adapter execute plan -> external model map -> reduce without adding a provider dependency to core.
- Define an explicit compatibility adapter from current free-text summaries to the report-context wire type. M1 is a non-grounded legacy-input slice and must not harden an incompatible report schema.

Exit gate:

- Every report at the chosen cut occurs in exactly one map batch.
- Same seed/input yields the same plan and exact coverage. On a fixture with at least three reports, selected known seeds exercise different permutations without changing coverage.
- Invalid scores, duplicate/missing map batches, stale reports, unsupported references, and over-budget reduce inputs fail loudly.
- A real query completes against existing summaries, clearly labelled **thin global query**, not paper-complete output.

### M2 — Retain paper-shaped element instances and aggregate them

**Purpose:** give communities the descriptive, weighted evidence the paper summarizes.

Deliverables:

- Replace the current one-checkpoint-per-chunk identity with versioned extraction attempts. Fingerprint `(store, chunk ID/content, model, extraction profile, prompt/schema version, glean configuration)` and track accepted/superseded state so a paper-profile backfill can coexist with legacy extraction safely.
- Add immutable entity instances and relationship instances linked to an accepted extraction attempt and source chunk. Store extracted names/types, descriptions, relationship strength, model, prompt/schema version, and response hash.
- Add optional claim/covariate records behind the same provenance contract; claims are not on the critical path unless the selected benchmark profile enables them.
- Extend extraction profiles for paper-shaped output and domain-specific entity types/few-shots.
- Make extraction outcomes typed: provider/transport failure and malformed/schema-invalid output remain retryable and write no terminal checkpoint; an explicit valid-empty result is terminal and resolvable.
- Generalize iterative gleaning as a plan/apply protocol: all passes share one extraction attempt and commit atomically only when finalized.
- Reconcile instances using the paper-compatible normalized exact-name baseline. Keep embedding-assisted merge as an explicit later resolution pass, not hidden extraction behavior.
- Plan/apply aggregate descriptions for canonical nodes and edges, retaining contributing instance IDs and fingerprints.
- Derive relation support count and projected weight from distinct relationship instances belonging to accepted attempts. Replay and superseded attempts must not increase either.

Exit gate:

- Two chunks describing one entity create two instances, one canonical node, a traceable aggregate description, and the expected evidence count.
- Repeated relationships from distinct accepted attempts increase support count/projected weight without creating duplicate physical edges; replay and superseded attempts change nothing.
- Multi-pass gleaning is atomic, idempotent, and bounded. Recall change is measured under the paper benchmark profile rather than required to improve for every model/fixture.
- A deliberate malformed-output mutant stays pending while an explicit empty extraction becomes a resolvable zero-yield checkpoint.
- Migration preserves legacy rows without fabricating descriptions, strength, claims, or evidence.

### M3 — Make hierarchy construction evidence-weighted and invalidatable

**Purpose:** produce the recursive MECE hierarchy required for divide-and-conquer summarization.

Deliverables:

- Build the Leiden projection from canonical nodes and accepted-attempt evidence weights, including isolates.
- For the paper profile, recursively apply Leiden to each parent-induced subgraph until terminal. The current aggregate-upward quotient clustering is structurally a different algorithm; keep it only as an explicitly named non-paper mode or replace it. A virtual corpus root may aid navigation, but it is not a Leiden community or paper invariant.
- Wire the canonical core + notebook/JSONL path and document hierarchy level/cut semantics. Cross-surface CLI/MCP/Python convergence follows in production hardening.
- Persist hierarchy run identity: algorithm/config/seed, graph fingerprint, modularity, parent/child containment, and memberships. Rebuild/report invalidation must be transactional at the run boundary; query plans bind the run ID rather than bare community IDs.
- Define explicit level cuts for uneven recursive trees: when a branch terminates, project its terminal community into deeper cuts so every queryable C-level remains MECE.
- Recurse within each non-terminal parent until the configured Leiden stopping rule says the community cannot be partitioned further.
- Expose run, cut, terminal-leaf, weight, and coverage diagnostics through core and the canonical adapter.

Exit gate:

- Every queryable cut is mutually exclusive and collectively exhaustive over the run's nodes.
- Every child has at most one parent, is contained by that parent, and participates in no cycle or cross-run link. Multiple coarsest communities are valid.
- Every non-leaf is partitioned by a Leiden run over that parent's induced subgraph; every leaf is terminal under the recorded stopping rule. Reference/spy tests prove recursion receives only parent members. C0 -> Cn exposes progressively lower detail with terminal projection where needed.
- Projected numeric weights exactly equal distinct accepted relationship support counts. A spy/reference test and a mutant forcing weights to `1.0` prove the weighted adjacency reaches Leiden; a partition change is not required.
- Same graph/config/seed reproduces the run. A graph change creates a new run and prevents old reports from satisfying new query plans.

### M4 — Generate citation-resolvable community reports

**Purpose:** replace thematic name lists with the report index consumed by global search.

Deliverables:

- Add a versioned `community_reports` record rather than overloading free-text `communities.summary`: title, executive summary, detailed findings, references, model, prompt version, token counts, hierarchy-run ID, and input fingerprint. The paper prompt profile also emits impact rating/rationale; keep those profile fields optional in the portable core schema.
- Construct leaf context from node descriptions, edge descriptions/weights, and enabled claims. Prioritize edges by decreasing combined endpoint degree and retain record IDs for lineage.
- Implement deterministic token packing. Never silently truncate an element mid-record.
- For higher communities, use all element summaries when they fit; otherwise substitute child reports for their longer child element sets, ranked by element-summary token count, until the context fits.
- Parse and validate structured report output; references must resolve to included entity, relationship, claim, or child-report IDs.
- Materialize bottom-up when child substitution is required. A parent whose elements fit directly does not depend on child reports under the paper algorithm; always-bottom-up scheduling may remain as a documented deterministic simplification.

Exit gate:

- Every materialized community has one complete, current report before its cut is globally queryable.
- Oversized leaf and parent fixtures prove prioritization/substitution behavior and remain within the configured token budget.
- Unsupported references, malformed report output, stale fingerprints, and use of unavailable required child reports write nothing.
- Every cited finding has resolvable lineage to graph records and extraction chunks. Citation existence is not claimed as entailment; sampled/evaluator factuality is a separate measure.

### M5 — Upgrade the global-query spine to paper-complete reports

**Purpose:** converge the online path from M1 with the faithful index from M2-M4.

Deliverables:

- Feed structured citation-resolvable reports from one hierarchy cut into the exhaustive map planner.
- Preserve valid report references through partial answers and the final reduce; reject invented record IDs.
- Add configurable response type, per-call map/reduce windows, cut, seed, and prompt versions to the query trace.
- Stabilize core plus notebook/JSONL global-map/global-reduce interfaces. Product CLI/MCP/Python aliases belong to H1; keep `ask` documented as hybrid source-chunk retrieval.
- Export the complete query trace as JSON for notebook inspection and evaluation.

Exit gate:

- End-to-end fixture passes from source documents through a citation-resolvable global answer.
- The trace proves complete cut coverage, score filtering, rank order, token packing, and citation lineage.
- A controlled fixture requires predefined evidence from multiple map batches; SS under its declared single-context budget omits predefined aspects while global map-reduce includes them. Real-corpus dogfood remains qualitative rather than a proof gate.

### M6 — Build the paper evaluation harness

**Purpose:** distinguish workflow completion, custom workflow validation, and reproduction of the paper's empirical protocol.

Comparison policy:

- Hold model, answer prompts, response policy, and **per-call** context window constant where the paper does. Do not equalize total evidence tokens or total calls: C-levels and TS exhaustively map different corpus volumes, while SS fills one retrieved context. Record aggregate tokens/calls/latency/cost as outcomes.
- `C0..Cn`: global query over every report in each hierarchy cut.
- `TS`: identical shuffle/map/score/reduce over source chunks.
- `SS`: vector retrieval of source chunks packed into one available context.

Deliverables:

- Three labelled evaluation tiers:
  - **Workflow validation:** custom corpus and human-reviewed frozen global questions.
  - **Protocol replication/generalization:** implements corpus-description -> `K` personas -> `N` tasks/persona -> `M` questions/task with `K=N=M=5` (125 questions per dataset), published conditions/metrics, and five blind judgment replicates; disclose corpus/model/prompt deviations.
  - **Empirical reproduction:** additionally uses the paper's Behind the Tech transcript corpus and 2013-2023 news corpus at published scale, benchmark chunk/window profile, prompts, and GPT-4-turbo configuration, with every unavoidable deviation enumerated.
- A pre-run protocol recording tier, benchmark profile, answer/judge models, prompts, answer-position counterbalancing/randomization, exactly five blind judgment replicates per question for protocol/empirical runs, statistical method, and any stricter project acceptance threshold. Disclose any human filtering and freeze questions before comparison.
- Separate judgments for comprehensiveness, diversity, empowerment, and directness; retain every raw winner/tie judgment and answer ordering.
- Token/call/latency/cost accounting for index and query stages, including unequal aggregate context consumed by each condition.
- The v2 validation lane: Claimify-style factual-claim decomposition, within-answer duplicate removal, mean unique-claim count for comprehensiveness, and complete-linkage agglomerative claim clusters using distance `1 - ROUGE-L` across multiple declared thresholds for diversity.
- Statistical report with win/tie/loss rates, effect sizes/significance/uncertainty, and criterion-specific interpretation. Do not collapse all criteria into one “GraphRAG wins” result.

Exit gate:

- All conditions answer the same frozen questions under audited equal per-call policies; expected aggregate token/call differences are visible rather than normalized away.
- Results are reproducible from saved plans, seeds, model/prompt versions, question-generation artifacts, raw answers, and raw judgments.
- Selected-corpus runs are labelled workflow validation or protocol replication/generalization, not reproduction of the paper's empirical findings. An **empirical reproduction** claim requires the original corpora/configuration tier and the paper's pairwise direction/significance protocol.
- Any lower-bound or pre-registered margin is labelled an additional `graphrag-rs` product policy, not attributed to the paper.
- Interpretation explicitly permits SS to win directness, empowerment to remain mixed, and GraphRAG not to beat TS. Factuality/entailment is reported separately; the paper did not establish a fabrication advantage.

## Production hardening after the paper milestone

M0-M5 complete the paper algorithm; M6 determines what empirical claim the implementation supports. The following hardening milestones make it the default operational RLM capability but are not conditions for calling the paper workflow implemented.

### H1 — Unify product surfaces and lifecycle safety

Deliverables:

- Route CLI, MCP, Nushell, Python, and notebook ingestion through one bounded-overlap/document-identity contract; remove the CLI whole-document bypass. Embeddings remain a product/local-search concern, not a paper-index prerequisite.
- Route product global-query commands through the canonical single-cut planner; replace/deprecate MCP's all-level delegation scaffold and keep `ask` explicitly hybrid/local.
- Expose stable aliases for extraction, hierarchy, report, global-map, and global-reduce plans without adding provider selection to Rust core.
- Complete legacy migrations, accepted/superseded attempt transitions, transactional hierarchy/report invalidation, recovery tooling, and compatibility tests.
- Publish lifecycle status: snapshot/run IDs, freshness, model/prompt/profile fingerprints, pending counts, and stale artifacts.

Exit gate:

- Equivalent inputs through supported surfaces create the same document/chunk identities and paper-index plans.
- No supported automated extraction or global-query surface bypasses typed validation, attempt identity, hierarchy cuts, or trace capture.
- Interrupted migration/rebuild/query application resumes idempotently or rolls back without repair.

### H2 — Backfill and operate the live corpus

**Purpose:** turn paper completeness into a usable RLM memory capability.

Deliverables:

- Snapshot the live database; run migration and cost/cardinality estimates on a copy; enumerate destructive cleanup separately.
- Backfill accepted element instances/aggregates, weighted hierarchy runs, and reports with resumable plan/apply batches.
- Audit a stratified sample of extraction descriptions, weights, communities, reports, references, malformed/empty outcomes, and superseded attempts.
- Run global questions about the operational corpus and compare them with current hybrid `ask` results.
- Publish the exact corpus snapshot, hierarchy run, report profile, and query traces used.

Exit gate:

- All index phases are complete for one immutable corpus snapshot; no legacy row is presented as exact evidence.
- Global-query traces resolve every report/reference to that snapshot and run.
- Recovery from an interrupted batch is idempotent and requires no graph repair.
- Live promotion occurs only after copy-based rehearsal, backup verification, and explicit approval for any destructive cleanup or push.

## Proposed change-set order

Keep these as reviewable, sequential changes rather than one long-lived implementation branch:

1. Paper contract fixture, benchmark profile, token-counter boundary, minimal report/query wire types, and gated conformance skeleton.
2. Thin global map planner, legacy-summary snapshot adapter, and query trace.
3. Map apply, helpfulness validation, reduce planner/apply, and canonical notebook/JSONL adapter.
4. Versioned extraction-attempt/accepted/superseded migration.
5. Immutable element instances, typed empty/malformed outcomes, and optional claims.
6. Paper extraction profile plus atomic bounded multi-pass gleaning.
7. Aggregate-description jobs and accepted-evidence support weights.
8. Canonical recursive hierarchy runs, cuts, terminal projection, and weight/MECE diagnostics.
9. Leaf report context builder, prioritization, and token packer.
10. Higher-level substitution and structured report plan/apply.
11. Citation-resolvable global-query convergence and full conformance fixture.
12. Evaluation condition/trace/budget skeleton (this can start earlier once M0 types stabilize).
13. Frozen global questions, SS/TS/C-level runs, judgments, claim analysis, and cost/statistical report.
14. H1 product-surface/lifecycle convergence.
15. H2 copy-based live backfill, audit, and dogfood.

Each change demonstrates its defining test red locally, then lands with default CI green (enabling the next conformance capability as appropriate). Run release-mode workspace tests, fmt, clippy with warnings denied, at least one relevant property test, and a deliberate mutation of the defining invariant. Provider calls and large Cargo builds remain serialized on the 8-core/16-GB host.

## Critical path and parallelism

```text
M0 contract
  |----> M1 thin global query -------------------------|
  `----> M2 instances/weights -> M3 hierarchy -> M4 reports
                                                       |
                                                       v
                                      M5 citation convergence
                                                       |
                                      M6 evaluation protocol/runs
                                                       |
                                               H1 hardening -> H2 live
```

M1 and M2 can be designed independently after M0, but prefer sequential integration unless a measured schedule constraint justifies parallel worktrees. M3's recursive hierarchy/cut semantics can be tested earlier against an injected weighted graph; production projection waits for accepted-attempt weights from M2. M4 depends on stable hierarchy-run identity and aggregate descriptions. The evaluation condition/trace skeleton can start after M0, but expensive/provider runs wait until M5 passes deterministic fixtures and small-corpus dogfood.

## Explicitly deferred beyond paper completion

- Entity-centric Local Search, DRIFT, LazyGraphRAG, mixed-level or dynamic report retrieval.
- Online incremental hierarchy repair and just-in-time community reports.
- Provider selection/retry policy inside Rust.
- Automatic soft entity resolution as part of extraction.
- Claims as a mandatory runtime feature.
- A claim that GraphRAG always beats source-text map-reduce or improves factuality.

## Final definition of done

`graphrag-rs` is paper-workflow complete when:

1. A fresh corpus can be indexed into described, accepted-evidence-weighted graph elements, reproducible MECE hierarchy cuts, and one current citation-resolvable report for every community.
2. A global query exhaustively maps every report in one chosen cut, filters and ranks scored partials, reduces once within its per-call window, and returns a reference-resolvable trace bound to the corpus/hierarchy/report snapshot.
3. The same frozen questions run under C-level, TS, and SS conditions with audited equal per-call policies, visible aggregate token/call differences, and reproducible evaluation artifacts.
4. The project labels each result as workflow validation, protocol replication/generalization, or empirical reproduction. Selected-corpus results test generalization; only the original-corpus/configuration tier may claim reproduction of the paper's criterion-specific findings.

H1-H2 determine whether the completed workflow is safe and convenient as the default live RLM capability; they do not retroactively redefine paper algorithm completeness.
