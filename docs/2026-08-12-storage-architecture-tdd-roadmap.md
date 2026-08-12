# Storage architecture TDD roadmap — static composition, dual backends, and atomic consumer cutover

**Status:** approved architecture roadmap; no production implementation or release approval.  
**Planning baseline:** `paper-workflow-roadmap` at `222d469` (including extraction provenance `b721fcf`).  
**Decision source:** Norman's explicit decisions on 2026-08-12; frozen in `2026-08-12-storage-architecture-decisions.md`.  
**Prior inputs:** `remediation-plan-v2.md`, `remediation-plan-v3.md`, `architecture-review.md`, `execution-review.md`, `v2-architecture-review.md`, `v2-execution-review.md`, and the v2 codec/SQLite/counted-SMT contracts under the current Prime Agent session artifacts.

## Consulted lessons before acting

- Isolate mutating work in its own worktree and preserve unrelated work.
- Narrow implementation iterations without deleting the deeper roadmap.
- Inventory existing capabilities before changing architecture claims.
- Consequential storage, migration, and contract decisions remain user-gated.


## 0. Scope statement

This is an **enabling architecture slice**, not a replacement for the immutable-evidence roadmap and not evidence that any paper, provenance, integrity, anchoring, claim, typed-relation, or evaluation requirement has been implemented.

The slice proves one architectural claim: core use cases can be statically composed over narrow storage capabilities; SQLite and authoritative plain files can satisfy the same semantic contract; Leit-backed indexes remain disposable derivatives; only the PyO3 boundary erases a fully composed service dynamically; existing consumers can then cut over in one revertible commit without inventing evidence or changing semantics.

The order is deliberately narrow:

1. freeze behavior and close blocking decisions;
2. establish a static core seam;
3. deliver a SQLite walking skeleton;
4. attach a rebuildable Leit index;
5. dogfood that whole service through Python early;
6. add the authoritative plain-file implementation;
7. prove backend parity and recovery;
8. cut over all existing consumers atomically after every pre-cutover gate is green.

Narrowing the first implementation slice does **not** remove anything from the deep roadmap (§12).

## 1. Evidence inventory and constraints

Repository facts that the implementation baseline must re-check and record rather than assume:

- `graphrag-core/Cargo.toml` currently depends directly on `rusqlite` and absolute-path `leit_*` crates.
- `graphrag-core/src/db.rs` exposes a concrete SQLite `Database`; core `mr`, capsule, vector, and error paths also contain SQL coupling.
- CLI, MCP, PyO3, Nushell, tests, and examples call that concrete API; migration ownership cannot be an exit-criterion-only promise.
- `graphrag-core/src/lexical.rs` currently builds a Leit `InMemoryIndex`; serialized segment bytes are write-only because the checked-in Leit revision has no public reload/search path.
- `graphrag-py::Store` currently owns `Mutex<Database>` and already exposes `stats`, hybrid `search`, extract/summarize `plan`/`apply`, `communities`, `triples`, and `export`. Python is an existing dogfood surface (DL-002), not a hypothetical binding.
- Prior v1/v2 plans were NO-GO; draft v3 is paused. Contract documents are design inputs, not proof that behavior exists.
- The v3 registry has 44 requirements: 27 evidence/storage, 7 paper-workflow, and 10 explicitly deferred evaluation requirements. This architecture plan adds enabling requirements; it neither renumbers nor marks those 44 complete.

Before implementation, create a machine-readable symbol inventory by repository search. It must contain actual file/symbol/import/call-kind rows for core SQL, public `Database`, CLI, MCP, Python, Nushell, tests, and examples. Do not predict a row count; the validator instead requires `discovered_rows > 0`, a stored search-output digest, and an owner/status for every discovered row.

## 2. Architecture invariants (stable requirements)

These are outcome requirements. Exact trait names, crate placement, file layout, and whether a Leit-common interface is reused are intentionally not specified.

### Static core composition

- **ARC-001 — dependency direction.** New core use-case code imports no `rusqlite`, SQLite backend crate/module, PyO3, or backend file-layout module. A temporary waiver may cover enumerated legacy files only; it cannot expand.
- **ARC-002 — static dispatch.** Rust composition roots construct concrete service types whose storage capabilities are selected at compile time. Core code contains no `Box<dyn Storage*>`, `Arc<dyn Storage*>`, or equivalent per-capability runtime erasure.
- **ARC-003 — narrow capabilities.** Core asks for the smallest semantic capabilities required by a use case, rather than a god `Store` or a lowest-common-denominator SQL/filesystem API. Capability methods exchange owned domain values, domain IDs, explicit page/cursor values, and typed errors—not row IDs, SQL strings, paths, or borrowed backend records.
- **ARC-004 — semantic transaction boundary.** A multi-record domain command is one backend operation/unit of work with all-or-nothing visibility. Core never simulates atomicity by issuing unrelated public capability calls.
- **ARC-005 — no physical identity leakage.** Physical SQLite row IDs, file paths, directory order, inode/mtime, and Leit doc IDs do not enter domain/wire/export identity.
- **ARC-006 — explicit modes.** Create, read-write existing, read-only existing, and legacy inspection are distinguishable. Read-only/inspection never creates, migrates, checkpoints, journals, lock artifacts, or indexes.
- **ARC-007 — typed capability/readiness.** Unsupported, unavailable-on-legacy, corrupt, conflict, stale-derived-index, and not-ready are distinct. A persisted boolean never substitutes for recomputed readiness.

### Backend semantic parity

- **BEP-001 — authoritative equivalence.** For the same frozen input sequence, SQLite and plain-file backends return the same canonical domain observations, stable ordering, idempotence/conflict result, and reopen result. Physical representations may differ.
- **BEP-002 — caller supplied/domain identity.** Backends preserve frozen domain IDs and canonical bytes; they do not assign observable identity from row order or file placement.
- **BEP-003 — replay/conflict.** Same logical ID plus identical canonical bytes is `AlreadyPresent`/no-op; same ID plus different canonical bytes is `IdentityConflict`; neither changes the committed head.
- **BEP-004 — CAS.** Head/publication replacement checks the expected predecessor. A stale writer returns a typed conflict and cannot expose a mixed generation.
- **BEP-005 — deterministic reads.** Reopen, pagination, and list order are backend-independent and explicitly tied to a domain ordering key.
- **BEP-006 — recovery.** Every owning multi-write command has named failpoints. Reopen after each failpoint exposes either complete pre-state or complete post-state, never a hybrid.
- **BEP-007 — zero-write inspection.** Recursive hashes, mtimes, sidecars, and SQLite auxiliary files are unchanged by read-only and legacy-inspection scenarios.

### SQLite

- **SQL-001.** Schema creation starts with a monotonic, checksummed migration ledger; opening current state does not reinterpret earlier bytes.
- **SQL-002.** Structural invariants available to SQLite are constraints, not comments; cross-object invariants are revalidated immediately before atomic commit/CAS.
- **SQL-003.** Read-only uses an actual read-only/query-only path and never calls initialization or migration.
- **SQL-004.** SQLite-specific errors map once into the shared typed error contract without leaking SQL text as behavior.
- **SQL-005.** Transaction ownership and failpoints are local to the semantic command that claims atomicity.

### Authoritative plain files

- **FIL-001.** Plain files are authoritative domain state, not an export of a hidden SQLite database and not a cache whose loss requires SQLite recovery.
- **FIL-002.** A fresh process can reopen using documented ordinary file formats plus version markers; no Leit index is needed to recover authoritative state.
- **FIL-003.** Commit visibility, durability claim, locking/single-writer policy, temp-file cleanup, and corruption detection are explicit and tested after the layout decision.
- **FIL-004.** Read-only mode creates no directory, temp file, lock, manifest rewrite, or derived index.
- **FIL-005.** Unknown future versions and partial/corrupt commits fail typed and do not trigger “repair” writes on open.
- **FIL-006.** Removing all derived-index artifacts leaves authoritative reads and deterministic index rebuild possible.

### Leit-derived indexes

- **IDX-001.** Leit artifacts are rebuildable derivatives of a named authoritative snapshot/generation; they are never the only copy of a domain fact or evidence record.
- **IDX-002.** Index metadata binds backend-independent corpus/snapshot identity, index profile/version, analyzer/config commitment, and source inventory commitment.
- **IDX-003.** Missing, stale, future-version, truncated, or corrupt indexes produce typed state and can be rebuilt without mutating authoritative records.
- **IDX-004.** A successfully reopened/rebuilt index yields backend-parity result IDs and deterministic tie ordering for the fixed corpus. Floating scores are compared only under a frozen scorer/revision and documented tolerance.
- **IDX-005.** Authoritative commit succeeds or fails independently of index refresh. The index becomes ready only after complete build plus atomic publication of its own descriptor.
- **IDX-006.** Read-only queries may use an already valid index but may not build or refresh one implicitly.
- **IDX-007.** Leit-native doc IDs and segment paths remain adapter-local and map through an explicit domain-ID table/encoding.

### PyO3 whole-service erasure and dogfood

- **PY-001 — erasure boundary.** PyO3 may own one `Box<dyn PythonGraphRagService + Send>` (name illustrative), representing a fully constructed service. Dynamic erasure does not cross back into core and is not applied separately to storage/index/transaction capabilities.
- **PY-002 — static construction before erasure.** Each factory branch first constructs and type-checks a concrete whole service (for example, SQLite storage + selected concrete index strategy), then erases that whole value.
- **PY-003 — stable Python behavior.** Existing `Store` methods and typed results remain baselined. Before `AS-008`, backend selection exists only through an explicitly experimental/test-only proving seam; the public default does not change.
- **PY-004 — honest capability reporting.** Python reports backend, protocol/version, derived-index state, and readiness reasons; it never claims evidence integrity, anchoring, or paper parity from architecture conformance alone.
- **PY-005 — GIL/thread discipline.** Blocking Rust work releases the GIL where safe; the erased service’s actual Send/Sync/serialization contract is tested rather than papered over with an outer mutex.
- **PY-006 — reopen/replay.** Python can close/reopen a fixture, repeat the narrow journey, and observe stable IDs/no duplicate effects through each enabled backend.
- **PY-007 — early proof.** SQLite dogfood through the proving seam follows the first Leit walking skeleton; it does not partially migrate the public Python surface.

### Consumer cutover and operations

- **CON-001.** Every discovered concrete-`Database`/raw-SQL caller has one owner, replacement seam, native test command, and status (`legacy`, `cutover-ready`, `removed`).
- **CON-002.** All surfaces remain legacy until one atomic commit moves Python, CLI, MCP, Nushell, tests, and examples and evacuates core SQL.
- **CON-003.** Pre-cutover comparison covers canonical IDs/bytes, typed outcomes, stable ordering, projection/search result IDs, and readiness. Any mismatch blocks the cutover; there is no shadow runtime or automatic resolution.
- **CON-004.** Experimental Python proofs may land before the cutover only when they do not alter the public `Store`, default behavior, or persisted contract. They are test evidence, not a migrated surface.
- **CON-005.** Legacy bridge/migration fabricates no immutable evidence and does not label mutable historical rows as provenance.
- **CON-006.** Core SQL evacuation is complete only when the discovered inventory has no concrete legacy caller and dependency/source sentinels pass on the atomic cutover commit.
- **CON-007.** The exact cutover commit is independently revertible against its parent; data downgrade compatibility is not inferred and remains a separate live-migration decision.
- **OPS-001.** No live migration/backfill, default promotion, destructive cleanup, dependency/revision change, push, or empirical provider run follows from a green fixture suite.

## 3. Stable scenarios

Scenario IDs and meanings are stable; test files may move only with a manifest update preserving ownership and history.

| Scenario | Logical cases | Required observations |
|---|---:|---|
| `MANIFEST-ARCH-SLICE` | 12 | every architecture requirement/scenario has one owner, status, command, positive count, and current inventory/search digest |
| `PRE-CUTOVER-SENTINELS-V1` | 14 | architecture, dependency, parity, recovery, RO hash, index rebuild, Python proof, legacy API snapshot, every legacy caller owned and cutover-ready, clean fixtures, no live-path access |
| `ARCH-STATIC-V1` | 10 | compile two concrete fake compositions; deny dynamic storage erasure; deny forbidden deps/new SQL; owned values/errors; narrow capability bounds; semantic UoW; composition-root construction |
| `SQLITE-SEMANTICS-V1` | 16 | create/open modes, ledger/checksum, idempotence, conflict, CAS, deterministic pages, reopen, typed corruption/future schema, zero-write RO, command atomicity |
| `WALKING-SKELETON-V1` | 8 | create corpus, store two exact source revisions, reopen, resolve/list, commit one expected-head transition, build derived index, query fixed token, repeat as no-op |
| `FILE-SEMANTICS-V1` | 16 | same semantic cases as SQLite plus layout-version, partial commit, corrupt manifest/object, no hidden DB/index dependency, zero-write RO |
| `BACKEND-PARITY-V1` | 24 logical / 48 backend invocations | fixed operation trace on both backends; canonical observation transcript byte-equal after physical fields excluded |
| `LEIT-DERIVED-V1` | 12 logical / 24 backend invocations | build, bind metadata, reopen/rebuild, missing/stale/corrupt/future, no implicit RO rebuild, atomic descriptor, stable IDs/ties |
| `PY-WHOLE-SERVICE-V1` | 12 | one whole-service dyn boundary, SQLite factory/dogfood, method compatibility, capabilities/readiness, GIL/thread behavior, close/reopen/replay, typed errors |
| `PY-FILE-PARITY-V1` | 8 logical / 16 backend invocations | opt-in backend selection, same fixture/transcript/results, delete/rebuild index, rollback selector, no default change |
| `CONSUMER-CUTOVER-V1` | 16 | complete caller inventory, all-surface native compatibility, explicit construction, canonical transcript preservation, no legacy core SQL/callers, atomic commit/revert proof, no invented evidence |
| `RELEASE-SENTINELS-V1` | 14 | architecture, dependency, parity, recovery, RO hash, index deletion/rebuild, Python import, legacy API snapshot, inventory, clean fixtures, no live-path access |

The conformance runner must reject zero matching tests, reject a count different from the manifest, reject ignored/filtered cases unless explicitly declared, record command/exit/tree digest, and print both logical and concrete backend invocation counts.

## 4. Backend parity corpus

Create one versioned, license-safe corpus in the repository. It is a conformance fixture, not a paper/evaluation corpus and not an empirical result.

Required fixture classes:

1. empty corpus and an explicitly valid empty query result;
2. two distinct source byte strings with stable frozen test IDs and spans (including newline and non-ASCII UTF-8 boundaries);
3. duplicate identical insertion/replay;
4. same ID/different canonical bytes conflict;
5. two records whose physical insertion order differs from domain order;
6. expected-head success followed by stale-head conflict;
7. a multi-record semantic command used for all failpoint recovery cases;
8. pagination boundary (empty, exact page, continuation, end);
9. one token found only lexically, one multi-token OR query, one deterministic tie;
10. missing, stale, corrupt, truncated, and future-version derived-index fixtures;
11. read-only and legacy/future-format fixture trees with precomputed recursive hashes and mtimes;
12. canonical expected observation transcript containing only domain values, typed outcomes, readiness reasons, and snapshot/index commitments.

The corpus must not freeze unresolved production evidence IDs by accident. Before `BEP-002` is activated, DL-009 must approve either the exact persisted vertical-slice ID/codec subset or an explicitly non-durable internal spike. Test-only IDs must be tagged `conformance-fixture-v1` and cannot be rebranded as production `evidence-v1`.

Parity compares canonical observations, not physical byte equivalence between SQLite and files. SQLite pages, WALs, file manifests, temp names, and Leit segments are excluded from the semantic oracle but are inspected by backend-specific safety tests.

## 5. Red-first protocol

Every milestone uses this order:

1. freeze requirement/scenario ownership and exact expected count;
2. add the narrowest compiling API scaffold needed to let behavior tests execute (methods return typed `Unsupported`/`NotReady`; no backend behavior);
3. run `--list` and prove the count is positive and exact;
4. run the scenario and capture RED: named assertions fail for the intended missing behavior, not compilation/platform noise;
5. implement the minimum vertical behavior;
6. capture GREEN with the same command/count;
7. apply each reviewed mutant or failpoint; the owning scenario must go RED with the same matched count;
8. revert and prove pre/post tree digest equality;
9. run release sentinels and independent review against the exact tree/plan digest.

A compile failure may be useful design feedback but is not accepted as the required behavioral RED. A test that passes before implementation is either a baseline/sentinel or incorrectly owned. No `#[ignore]`, unconditional early return, backend-name skip, mock-only assertion, or “expected count 0” satisfies evidence.

The runner command shape is:

```bash
rtk python3 scripts/conformance/run.py \
  --scenario <ID> --expect-logical <N> --expect-tests <M> --phase red
```

`--phase red` succeeds as an evidence-capture command only when exactly `M > 0` tests execute and the manifest-declared named assertions fail; the stored underlying test exit remains non-zero. Green uses `--phase green` and requires all `M` pass. The runner itself uses the Python standard library; adding a third-party test harness remains dependency-gated.

## 6. Walking skeleton

The first production walking skeleton is deliberately smaller than the evidence journey:

`frozen conformance source values → statically composed core command → SQLite authoritative commit → close/reopen → Leit derivative build → fixed lexical query → whole-service PyO3 erasure → Python observation/replay`.

It proves architecture and operation wiring only. It does **not** prove raw-provider transcript custody, immutable acceptance/publication, paper relationship weights, authenticated maps, external anchoring, claims, or evaluation.

The skeleton is complete only when:

- `ARCH-STATIC-V1` has 10/10 green;
- `SQLITE-SEMANTICS-V1` has 16/16 green;
- `WALKING-SKELETON-V1` has 8/8 green;
- `LEIT-DERIVED-V1` has 12 logical/24 concrete green for the available backend lanes (SQLite plus a conformance reference lane until file backend lands);
- `PY-WHOLE-SERVICE-V1` has 12/12 green;
- deleting the Leit artifact preserves authoritative reads and a rebuild returns the same result IDs/order;
- Python replay is a no-op and reports architecture-only readiness honestly.

The conformance reference lane is not a production backend and cannot count toward the two-backend release gate.

## 7. Factored milestones

### `AS-000` — decision closure, baseline, and conformance harness

**Depends on:** none. **Edits authorized:** planning/fixture harness only after separate GO.  
**Deliverables:** exact baseline outputs; symbol inventory; requirement/scenario manifest; parity-corpus draft; decision-ledger additions in §11; no new production crate/schema.

RED/GREEN:

```bash
rtk python3 scripts/conformance/run.py --scenario MANIFEST-ARCH-SLICE --expect-logical 12 --expect-tests 12 --phase red
rtk python3 scripts/conformance/run.py --scenario MANIFEST-ARCH-SLICE --expect-logical 12 --expect-tests 12 --phase green
```

RED mutant: `MUT-MANIFEST-ORPHAN` removes one owner; all 12 execute and the ownership assertion fails.  
Gate: the five storage decisions frozen on 2026-08-12 are represented in the scenario manifest. DL-003's direction is fixed—narrow static capabilities in core, backend implementations outside core, and dynamic erasure only around the complete PyO3 service—while exact crate/module names remain implementation-reviewed rather than persisted contract.

### `AS-001` — static core capability seam

**Depends on:** `AS-000`.  
**Scope:** narrow capability bounds, typed errors/modes, generic whole service, test concrete fake/reference types, architecture deny rules. Do not add a production backend or persisted evidence schema.

```bash
rtk python3 scripts/conformance/run.py --scenario ARCH-STATIC-V1 --expect-logical 10 --expect-tests 10 --phase red
rtk python3 scripts/conformance/run.py --scenario ARCH-STATIC-V1 --expect-logical 10 --expect-tests 10 --phase green
```

Mutants (10 tests each; expected RED):

- `MUT-CORE-DYN-STORAGE`: replace one generic bound with `Box<dyn ...>`;
- `MUT-CORE-SQL-IMPORT`: add a forbidden SQL import to new core code;
- `MUT-ROWID-LEAK`: expose a backend integer/path as domain identity;
- `MUT-SPLIT-UOW`: split the semantic command across two public writes.

Exit: two independently compiled concrete test compositions behave identically; new core code has no backend dependency. Trait/module/crate names are not treated as persisted API until architecture review approves them.

### `AS-002` — SQLite minimum authoritative backend

**Depends on:** `AS-001`, approved persisted vertical-slice ID/codec decision.  
**Scope:** SQLite modes, checksummed ledger from schema 1, minimum frozen records/heads needed by the skeleton, semantic UoW, reopen, RO inspection. No legacy live migration and no full evidence schema pre-creation.

```bash
rtk python3 scripts/conformance/run.py --scenario SQLITE-SEMANTICS-V1 --expect-logical 16 --expect-tests 16 --phase red
rtk python3 scripts/conformance/run.py --scenario SQLITE-SEMANTICS-V1 --expect-logical 16 --expect-tests 16 --phase green
rtk python3 scripts/conformance/run.py --scenario WALKING-SKELETON-V1 --backend sqlite --expect-logical 8 --expect-tests 8 --phase red
rtk python3 scripts/conformance/run.py --scenario WALKING-SKELETON-V1 --backend sqlite --expect-logical 8 --expect-tests 8 --phase green
```

Mutants/failpoints:

- `MUT-SQL-RO-MIGRATE`, `MUT-SQL-ID-ROWID`, `MUT-SQL-REPLAY-DUP`, `MUT-SQL-CAS-OMIT` (16 matched tests each, named failure);
- failpoints `sqlite.after-open`, `sqlite.after-ledger`, `sqlite.after-records`, `sqlite.before-head-cas`, `sqlite.after-head-cas-before-commit` (8 walking-skeleton tests each; reopen pre/post only).

Exit: migration 1 is not created by ordinary RO open; schema constraints and command validation are both exercised; no live database was opened.

### `AS-003` — Leit derivative over SQLite skeleton

**Depends on:** `AS-002`, approved exact Leit revision/dependency action, and sufficient decision on the index seam (not necessarily a universal common-index abstraction).  
**Scope:** derive, bind, discard, and rebuild the fixed lexical index; map Leit doc IDs to domain IDs; no authoritative facts in the segment.

```bash
rtk python3 scripts/conformance/run.py --scenario LEIT-DERIVED-V1 --backend sqlite,reference --expect-logical 12 --expect-tests 24 --phase red
rtk python3 scripts/conformance/run.py --scenario LEIT-DERIVED-V1 --backend sqlite,reference --expect-logical 12 --expect-tests 24 --phase green
```

Mutants/failpoints (24 tests each):

- `MUT-INDEX-AUTHORITATIVE`: make a record readable only from Leit;
- `MUT-INDEX-OMIT-SNAPSHOT`: omit source generation from metadata;
- `MUT-INDEX-IMPLICIT-RO-REBUILD`: rebuild on RO query;
- `MUT-INDEX-DOCID-LEAK`: return native doc ID;
- failpoints `index.after-temp-build`, `index.before-descriptor-publish`, `index.after-descriptor-publish`.

Exit: deleting/corrupting the index changes readiness, not authoritative data; rebuilding restores fixed IDs/order. If current Leit cannot reopen searchable bytes, the honest behavior is `RebuildRequired`; this milestone must not claim persistence/reload support the dependency does not provide.

### `AS-004` — PyO3 whole-service proving seam (pre-cutover)

**Depends on:** `AS-003`.  
**Scope:** prove that Python can own one dynamically erased, fully composed service without migrating the existing public `Store` or changing its default. Exercise the new SQLite service through an explicitly experimental/test-only entry point; preserve and baseline every current method for the later atomic cutover.

```bash
rtk python3 scripts/conformance/run.py --scenario PY-WHOLE-SERVICE-V1 --expect-logical 12 --expect-tests 12 --phase red
rtk uv run --project graphrag-py python -m unittest discover -s graphrag-py/tests -p 'test_*.py' -v
rtk python3 scripts/conformance/run.py --scenario PY-WHOLE-SERVICE-V1 --expect-logical 12 --expect-tests 12 --phase green
```

The unittest command must report `Ran 12 tests` and `OK`; build/install prerequisites are recorded separately and may require explicit dependency approval.

Mutants (12 tests each):

- `MUT-PY-DYN-STORAGE`: independently erase a storage capability;
- `MUT-PY-DEFAULT-FILE`: silently change backend default;
- `MUT-PY-READINESS-BOOL`: persist/return an unsupported green boolean;
- `MUT-PY-REPLAY-DUP`: duplicate an observation on reopen/replay;
- `MUT-PY-HOLD-GIL`: hold the GIL across the declared blocking probe.

Exit: the existing Python compatibility fixture is unchanged; an end-to-end experimental SQLite/Leit transcript is captured; architecture readiness is honest. This is pre-cutover evidence, not a partial consumer migration.

### `AS-005` — authoritative plain-file backend

**Depends on:** `AS-001`, approved file-layout/durability/locking decision from §11, and the same frozen domain subset as SQLite. It may proceed after `AS-004`; it must not block early Python dogfood.  
**Scope:** implement the same capabilities using ordinary authoritative files; no SQLite and no Leit dependency for reopen.

```bash
rtk python3 scripts/conformance/run.py --scenario FILE-SEMANTICS-V1 --expect-logical 16 --expect-tests 16 --phase red
rtk python3 scripts/conformance/run.py --scenario FILE-SEMANTICS-V1 --expect-logical 16 --expect-tests 16 --phase green
rtk python3 scripts/conformance/run.py --scenario WALKING-SKELETON-V1 --backend file --expect-logical 8 --expect-tests 8 --phase red
rtk python3 scripts/conformance/run.py --scenario WALKING-SKELETON-V1 --backend file --expect-logical 8 --expect-tests 8 --phase green
```

Mutants/failpoints:

- `MUT-FILE-HIDDEN-SQLITE`, `MUT-FILE-PATH-ID`, `MUT-FILE-RO-LOCK`, `MUT-FILE-INDEX-SOURCE`, `MUT-FILE-FUTURE-REPAIR` (16 each);
- exact failpoint names are selected only after layout decision, but must cover every visibility transition: temp/object write, durability barrier if claimed, manifest/head preparation, atomic visibility step, cleanup. Each runs 8 skeleton tests and reopens to pre/post only.

Exit: authoritative journey reopens after all Leit artifacts are removed; zero-write hashes/mtimes hold; future/corrupt/partial states fail typed. Do not invent failpoint names before the commit protocol exists.

### `AS-006` — dual-backend parity and recovery sweep

**Depends on:** `AS-003`, `AS-005`.  
**Scope:** run one canonical operation trace against independently fresh SQLite and file fixtures, compare domain transcripts, exercise concurrency/CAS and every owning failpoint. Fixes belong in their owning backend milestone; this sweep adds no new transaction semantics.

```bash
rtk python3 scripts/conformance/run.py --scenario BACKEND-PARITY-V1 --backend sqlite,file --expect-logical 24 --expect-tests 48 --phase red
rtk python3 scripts/conformance/run.py --scenario BACKEND-PARITY-V1 --backend sqlite,file --expect-logical 24 --expect-tests 48 --phase green
rtk python3 scripts/conformance/run.py --scenario LEIT-DERIVED-V1 --backend sqlite,file --expect-logical 12 --expect-tests 24 --phase green
```

Mutants (48 parity tests unless noted): `MUT-PARITY-ORDER`, `MUT-PARITY-CONFLICT-AS-REPLAY`, `MUT-PARITY-PHYSICAL-ID`, `MUT-PARITY-STALE-CAS`; `MUT-INDEX-BACKEND-TIE` runs 24 index tests.

Exit: canonical transcripts are byte-equal; backend-specific artifacts differ only outside the oracle; failpoint evidence is linked to each owning command.

### `AS-007` — Python dual-backend pre-cutover proof

**Depends on:** `AS-004`, `AS-006`.  
**Scope:** extend the experimental/test-only Python proving seam with explicit SQLite and file factories, construct each concrete whole service, then erase it at the same PyO3 boundary. The existing public `Store` and its default remain untouched until `AS-008`.

```bash
rtk python3 scripts/conformance/run.py --scenario PY-FILE-PARITY-V1 --backend sqlite,file --expect-logical 8 --expect-tests 16 --phase red
rtk python3 scripts/conformance/run.py --scenario PY-FILE-PARITY-V1 --backend sqlite,file --expect-logical 8 --expect-tests 16 --phase green
```

Mutants: `MUT-PY-FACTORY-NESTED-DYN`, `MUT-PY-FILE-DEFAULT`, `MUT-PY-FILE-INDEX-AUTHORITY`, `MUT-PY-BACKEND-MISMATCH-IGNORE` (16 tests each).

Exit: the same Python journey yields canonical observations on both explicit experimental backends; existing public Python behavior remains unchanged. This proves the cutover target but does not migrate a surface.

### `AS-008` — atomic all-consumer cutover and core SQL evacuation

**Depends on:** `AS-004`, `AS-006`, `AS-007`, every consumer-specific prerequisite, and a green `PRE-CUTOVER-SENTINELS-V1` run.  
**Cutover unit:** one self-contained, independently revertible commit migrates Python, CLI, MCP, Nushell, tests, and examples; evacuates legacy core SQL; and updates the complete caller inventory. No surface lands early, no shadow mode is introduced, and no automatic mismatch resolution exists.

```bash
rtk python3 scripts/conformance/run.py --scenario CONSUMER-CUTOVER-V1 --expect-logical 16 --expect-tests 16 --phase red
rtk python3 scripts/conformance/run.py --scenario BACKEND-PARITY-V1 --backend sqlite,file --expect-logical 24 --expect-tests 48 --phase green
rtk python3 scripts/conformance/run.py --scenario PRE-CUTOVER-SENTINELS-V1 --expect-logical 14 --expect-tests 14 --phase green
# Apply the complete consumer cutover as one candidate commit, then:
rtk python3 scripts/conformance/run.py --scenario CONSUMER-CUTOVER-V1 --expect-logical 16 --expect-tests 16 --phase green
rtk python3 scripts/conformance/run.py --scenario RELEASE-SENTINELS-V1 --expect-logical 14 --expect-tests 14 --phase green
```

The 16 cutover tests cover public compatibility for each surface, explicit backend construction, canonical transcript preservation, inventory completeness, absence of concrete `Database` callers, and absence of forbidden SQL dependencies in core. Surface-native tests from the live caller inventory are mandatory in addition to the generic scenario.

Mutants: `MUT-CUTOVER-DROP-CALLER`, `MUT-CUTOVER-LEGACY-SQL`, `MUT-CUTOVER-PY-DEFAULT`, `MUT-CUTOVER-SURFACE-DRIFT`, `MUT-CUTOVER-FABRICATE-EVIDENCE`, and `MUT-CUTOVER-PARTIAL-COMMIT` (16 tests each).

Hard gate: `AS-006` parity and recovery, both Python backend proofs, all surface-native compatibility tests, and all 14 **pre-cutover** sentinels must be green on the exact parent commit. Pre-cutover sentinel 11 requires every concrete legacy caller to be owned and `cutover-ready`; it does not require those callers to be absent. On the candidate cutover commit, the 14 release sentinels run with the post-cutover expectation that no concrete legacy caller remains. Because DL-022 rejects a shadow window, this phase-separated evidence is the only discrepancy gate.

Rollback proof: before approval, apply and revert the exact cutover commit in a disposable worktree; the reverted tree digest and full legacy compatibility suite must match the parent. Rollback is VCS-level and does not claim that data written after deployment can be downgraded; any live-data migration remains separately approval-gated.

Exit: inventory has no concrete legacy caller or unowned row; all consumers compile and pass on the new service in the same commit; core SQL dependency deny is green; no production push, default promotion, or live migration is implied.

### `AS-009` — release-candidate sentinels

**Depends on:** all enabled architecture milestones; does not authorize release/push/promotion.  

```bash
rtk python3 scripts/conformance/run.py --scenario RELEASE-SENTINELS-V1 --expect-logical 14 --expect-tests 14 --phase green
rtk cargo test --workspace --all-targets
rtk cargo clippy --workspace --all-targets -- -D warnings
```

Optional Python/Nushell/platform prerequisites are reported separately; do not turn an unavailable optional toolchain into a false green or silently drop it from release scope.

## 8. Pre-cutover and release sentinels

`PRE-CUTOVER-SENTINELS-V1` runs on the exact parent of the atomic cutover. It has the same 14 categories below, except category 11 requires a current inventory in which every concrete legacy caller is owned, has native tests, and is `cutover-ready`.

`RELEASE-SENTINELS-V1` runs on the candidate cutover commit. Its 14 tests are:

1. scenario manifest has one owner/status/command and positive count for every active requirement;
2. no forbidden new core dependency/import or absolute local path dependency in a releasable lockfile;
3. no per-capability `dyn Storage/Index/Transaction` outside reviewed PyO3 whole-service erasure;
4. SQLite/file canonical parity transcript matches the reviewed fixture digest;
5. all owning backend failpoints have pre/post-only reopen evidence;
6. recursive RO hashes/mtimes/SQLite sidecars remain unchanged;
7. removal/corruption of every Leit artifact preserves authoritative reads and yields typed rebuild state;
8. rebuilt index returns frozen domain IDs/tie order for both backends;
9. PyO3 extension imports and reports backend/protocol/readiness; 12 existing/new compatibility tests pass;
10. Python explicit SQLite/file transcripts match and selector default has not changed without approval;
11. caller inventory search digest is current, no symbol row is unowned, and the atomic cutover leaves no concrete legacy caller;
12. every legacy public API snapshot/fixture remains green on the cutover commit or has a separately approved breaking-change record;
13. fixture tests touch no live/default user database, home data directory, provider endpoint, or external anchor;
14. repository tree, migration fixtures, and mutant revert digests are clean; quarantined/unrelated work is preserved.

A release report states exact counts and exits. “Workspace green” without scenario counts, mutants, failpoints, parity transcript, and optional-surface status is insufficient.

## 9. Worktree, review, and approval gates

### Read-only planning gate

This artifact authorizes no repository edit, lease, dependency action, build mutation, live data access, cleanup, commit, or push. The target worktree must be re-inspected at execution time; this draft does not assert it is still clean or exclusively owned.

### Before the first implementation edit

1. Norman approves the selected milestone and the corresponding decision-ledger entries.
2. Refresh base, worktree status, untracked/quarantined paths, and non-ancestry; preserve unknown work.
3. Claim the target worktree using the repository’s lease mechanism if present; otherwise create the reviewed external lease. One writer only.
4. Record base commit, plan digest, requirement-manifest digest, exact command versions, Rust/Python/Leit revisions, and baseline exits.
5. Obtain PAAD architecture/execution review for the milestone’s exact slice. Adjudicate findings and re-review before GO.
6. If subagents edit, each gets a separate branch/worktree/lease. Read-only reviewers may inspect the same tree.

### Separate approvals that a milestone green does not imply

- new crate topology or moving public types;
- any new dependency, absolute-path-to-git/version switch, or Leit revision bump;
- production file layout, locking model, durability claim, or format freeze;
- adopting/creating a Leit-common index abstraction;
- persisted canonical-ID/codec freeze or reinterpretation;
- live database/file migration, backfill, repair, or legacy bridge execution;
- default backend/index/consumer promotion;
- destructive cleanup or removal of legacy paths/data;
- external key custody, anchor/log operation, or trust promotion;
- provider calls, budgets, licensed corpus use, or empirical evaluation;
- commit/push/PR/merge/release.

### Per-milestone evidence gate

GO is bound to exact plan/tree digests and permits only the stated milestone. STOP on: count mismatch/zero matches, unexplained baseline red, dirty/unleased worktree, unresolved decision required by that milestone, mutant that stays green, failpoint hybrid state, RO write, parity mismatch, dependency drift, or review verdict NO-GO.

## 10. Exact non-goals of this architecture slice

1. No claim that the 44 v3 evidence/paper/evaluation requirements are implemented or proved.
2. No full immutable-evidence schema, raw-provider envelope, acceptance/publication generation, projection, counted SMT, proof, or anchor implementation.
3. No decision on DL-004 publication consistency, DL-007 paper-versus-semantic identity, DL-010 authenticated-map structure, or DL-016 relation inference. DL-022 is decided as an atomic cutover, but live legacy-data migration/backfill/removal remains separately gated.
4. No silent freeze/reinterpretation of canonical v1 IDs, codecs, enum tags, canonical bytes, or persisted schema.
5. DL-023 fixes hybrid external-file + managed-record authority and an atomic generation manifest, but does not yet choose exact paths, canonical encodings, visibility primitive, multi-process lock, or durability level.
6. DL-024 fixes common Leit-derived retrieval as the target and current concrete wrappers as the stage; it does not freeze the eventual universal interface, persisted segment API, vector index, or adapter names.
7. No runtime-swappable backend/plugin system inside Rust core; no independently boxed storage/index/transaction traits.
8. No promise that SQLite and files share physical format, byte-for-byte artifacts, identical performance, or identical internal query plans.
9. No use of Leit/HNSW/BM25 artifacts as authoritative evidence, facts, identities, or publication heads.
10. No automatic index build, migration, repair, or lock creation during read-only open/query.
11. No default backend switch before AS-008, no shadow mode, and no automatic mismatch resolution.
12. No live migration/backfill and no fabrication of evidence for legacy rows.
13. No empirical provider run, paper reproduction claim, quality/factuality superiority claim, or budget/license decision.
14. No final claims schema, claim extraction/ordering/invalidation policy, typed-relation constraint semantics, open/closed-world choice, materialized/query-time inference, vocabulary import policy, or external fact-set trust tier.
15. No descope of any item in §12.

## 11. Frozen architecture decisions

The authoritative rationale and consequences are in `2026-08-12-storage-architecture-decisions.md`.

- **DL-003 — static seam direction:** narrow capabilities and domain values live in `graphrag-core`; SQLite and file implementations live outside core; Rust composition uses static dispatch; only `graphrag-py` dynamically erases one fully composed service. Exact crate/module names remain implementation-reviewed.
- **DL-009 — first durable subset: B.** Freeze only versioned corpus/source/chunk/head/index-binding identifiers and codecs required by the walking skeleton. Later evidence types are additive versions; test-only IDs cannot be rebranded.
- **DL-023 — file authority: D.** External/source files remain in place; managed immutable evidence and metadata records plus an atomic generation manifest live beside them. `AS-005` must specify and test locking, durability, corruption, and visibility transitions before production code claims them.
- **DL-024 — Leit: A target, C staged.** Both authoritative backends target one common Leit-derived retrieval layer. Implement against today's concrete wrappers first; do not block on persisted executable segments, vector indexing, or Leit PR #18.
- **DL-025 — embedding custody: B.** Each authoritative backend retains exact vector bytes and model/profile identity. Leit owns disposable search structures, never the sole vector copy.
- **DL-022 — consumer migration: C.** Norman explicitly selected one big-bang cutover over the recommended shadow migration. `AS-006` becomes a hard pre-cutover gate and `AS-008` is one independently revertible all-consumer commit.

These decisions authorize planning detail only. Each implementation milestone still requires its own GO and evidence gate.

## 12. Preserved complete deep roadmap

The following remains additive after the architecture slice. Nothing below is deleted, marked complete, or folded into a vague “future work” bucket.

1. **Canonical domain/codec freeze:** transitively complete IDs, union tags, policy/config commitments, vectors, no physical IDs; stable additive migrations only.
2. **Immutable sources and provider evidence:** exact source revisions/spans; Rust-owned logical requests; exact provider-native request/response bytes and metadata; attempt/pass/delivery/outcome/receipt; malformed, failure, valid-empty, absent, rejected, accepted, and superseded remain distinct.
3. **Typed parsed instances and candidate decisions:** entity/relationship descriptions and properties survive receipt → parse → instance → reopen; duplicate tuple ordinals remain distinct; idempotence/conflict exact.
4. **Atomic active publication:** inactive immutable candidates; chosen DL-004 consistency model; lane replacement, projection/support/bundle, and corpus-head CAS with pre/post-only recovery.
5. **Paper-exact projection:** literal raw-name equality; directed endpoint-pair identity if DL-007 confirms coexistence; edge weight equals active accepted relationship-instance count; replay/rejected/replaced/strength mutants.
6. **Immutable descriptions and aggregate summarization:** entity, relationship, and claim-support descriptions stay distinct; aggregate records bind all support IDs, exact production aggregation request/transcript/result, snapshot, lineage, and invalidation.
7. **Claims as an owned stage:** immutable claim extraction schema, ordering, support references, aggregation/report inclusion, freshness/invalidation, and explicit claims-disabled profiles until proved. Claims removed from draft v3 are restored here per DL-013.
8. **Authenticated evidence and proofs:** selected DL-010 map/log structure; membership, exact-key nonmembership, map-scoped mutation, bounded enumeration, and composite fact→acceptance→parse→receipt→attempt→source lineage; cardinality is not semantic truth.
9. **External anchoring:** signed bundle/head/predecessor/policy tuple, external key/log/checkpoint inputs, rollback tests, custody, rotation, witness policy; early status remains exactly integrity-valid/authorization-unanchored.
10. **Legacy bridge and production migration:** fixture proof first; no fabricated evidence; pending/stale/inexact readiness; backup, dry run, rollback, live backfill, and removal each separately approved.
11. **Paper workflow:** reproducible token-bounded overlapping chunks; extraction with reflection/glean where specified; weighted parent-induced recursive Leiden until recorded unpartitionable; MECE cuts; exact degree/report ordering; row-replacing child substitution; exhaustive one-cut global query with zero/one reduce.
12. **Consumer convergence:** CLI, Python, MCP, and Nushell use production evidence services after the DL-022 atomic cutover; pre-cutover parity, exact raw adapter transcripts, independently proved commit reversion, live-data rollback planning, and separately approved legacy removal remain required.
13. **Capabilities/readiness:** advertised operations/protocols only; readiness recomputed from artifacts, heads, active lanes, aggregates, hierarchy/report/query coverage, unresolved refs, and externally verified anchor input.
14. **Evaluation:** preserve EVAL-001..010 and three distinct tiers—workflow validation, protocol replication/generalization, empirical reproduction. Same frozen questions/model/prompts/window/per-call policy; C0–C3/TS/SS semantics; blind counterbalanced judgments with five replicates; K=N=M=5 question generation; Claimify/dedupe/claim-count/complete-linkage ROUGE-L; raw artifacts/seeds/order/native envelopes/usage/cost/stats. Activation still requires separate PAAD plus corpus/license/provider/budget approval.
15. **Typed semantic relations:** preserve raw predicates; seed canonical DCTERMS URIs; versioned vocabularies; domain/range, cardinality, inverse, symmetric, transitive, temporal, and trust constraints; decide validation versus materialized/query-time inference and open versus scoped-closed-world semantics; inferred facts require explicit provenance.
16. **External fact sets:** namespaces, source/version/license provenance, vocabulary mapping, trust tiers, update/invalidation, and agent-accessible traversal/query.
17. **GraphRAG memory role:** tangential associative recall, multi-hop traversal, durable insights, external knowledge, and context selection, complementing rather than replacing Prime Agent working memory.
18. **Operational hardening:** owning failpoints at every publication/migration step, complete recovery sweep, deterministic ties/budgets, provider/cost controls, external-anchor operations, and separately approved release/promotion/removal.

The architecture slice is successful only if it lowers the cost of these milestones without weakening their contracts. A backend abstraction that cannot support exact bytes, immutable history, atomic generation publication, proofs, typed relations, claims, or evaluation artifacts must be revised rather than used to descope the roadmap.

## 13. Review checklist

A reviewer can approve an implementation milestone only when all answers are yes:

- Is the milestone’s required decision closed without silently closing later open decisions?
- Does its scenario run a positive exact count and show an intentional behavioral RED before GREEN?
- Are all requirements owned exactly once and deep-roadmap requirements still planned/deferred honestly?
- Is core static and backend-neutral, with dynamic erasure confined to one PyO3 whole-service boundary?
- Are SQLite/files authoritative semantics compared through a fixed corpus while physical differences remain backend-local?
- Can all Leit artifacts be deleted without data loss, and can RO avoid rebuilding them?
- Do mutants fail for their named invariant and failpoints reopen to pre/post only?
- Are Python’s existing capabilities preserved and proved through the test-only seam before the atomic consumer cutover?
- Are live data, dependencies, layout, common-index, promotion, cleanup, provider, evaluation, and push approvals still separate?
- Is evidence bound to exact plan/tree/fixture/toolchain digests with unrelated work preserved?
