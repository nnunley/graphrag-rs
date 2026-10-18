# Storage architecture inventory

**Scope:** read-only inventory of commit/worktree state in `paper-evidence-remediation`, captured 2026-08-12 for the approved direction: static capability traits in `graphrag-core`; all SQL in a new `graphrag-sqlite`; a second authoritative plain-file implementation; Leit limited to indexes over metadata/BoW/embeddings; whole-service dynamic erasure only at the PyO3 boundary.

## Baseline evidence

- Workspace was clean before this artifact (`git status --short` emitted no paths).
- `rtk cargo test --release --workspace` → **154 passed, 1 ignored, 19 suites, 3.46s** (warm release artifacts).
- `rtk cargo build --release -p graphrag-py` → success; `rtk cargo test --release -p graphrag-py` → **0 tests, 1 suite**. The build reported 89 crates and 1m27s.
- `maturin` is not installed on the inspected PATH; Python is 3.14.5. Therefore no import/wheel smoke test was run.
- Source contains 144 ordinary `#[test]`/`#[hegel::test]` functions; macro expansion/doctests account for the runner total. Storage-heavy suites are listed below.

## Current crate graph

Workspace members and direct internal edges (`Cargo.toml`, each crate manifest):

```text
graphrag-core ──> leit_core, leit_text, leit_index, leit_score, leit_fusion
                 rusqlite 0.32 bundled; serde/json; hashing; optional embedding/chunking deps
graphrag-llm     (no dependency on core)
graphrag-cli ──> graphrag-core + graphrag-llm
graphrag-mcp ──> graphrag-core + graphrag-llm
nu_plugin_graphrag ──> graphrag-core
graphrag-py ──> graphrag-core + graphrag-llm
```

- Default workspace builds/tests: core, llm, MCP, CLI (`Cargo.toml:19-24`). Python and Nushell are workspace members but excluded from `default-members`; explicit workspace commands still include them.
- `graphrag-core/Cargo.toml:20` owns `rusqlite`; none of the consumer manifests depends on it directly.
- Leit dependencies are absolute local paths (`graphrag-core/Cargo.toml:29-33`), so builds are not portable without Norman's adjacent Leit checkout.
- Intended graph after extraction is mechanically visible: `graphrag-core` must lose rusqlite and define models/capabilities/services; `graphrag-sqlite -> graphrag-core + rusqlite`; each binary/binding must add/select a concrete backend; a plain-file crate (or module outside core) implements the same core capabilities without SQL.

## SQL and concrete-Database ownership: exhaustive file inventory

All direct `rusqlite` references:

| File | Evidence | Responsibility |
|---|---:|---|
| `graphrag-core/Cargo.toml` | 20 | bundled dependency |
| `graphrag-core/src/db.rs` | 2, 505-506 | `Connection`, `params!`, dynamic `ToSql` |
| `graphrag-core/src/error.rs` | 5-6 | core error publicly wraps `rusqlite::Error` |
| `graphrag-core/src/mr.rs` | 26; 136,177,296,318,523,782 | direct SQL for extraction/summarization/status/embed queries |
| `graphrag-core/src/capsule_store.rs` | 11; 84-169 | SQL implementation of `CapsuleStore` on `Database` |
| `graphrag-core/src/vector_source.rs` | 16,135; 126-160 | chunk-vector SQL reads/writes |

No `transaction`, `BEGIN`, `COMMIT`, `ROLLBACK`, or rusqlite transaction API occurs anywhere in the repository.

Concrete type/export leakage:

- `graphrag-core/src/db.rs:75-77`: `Database { conn: Connection }`.
- `graphrag-core/src/db.rs:191-193`: crate-visible `conn()` exists specifically for sibling SQL providers.
- `graphrag-core/src/lib.rs:71`: publicly re-exports `Database` and row models; crate docs (`:16-42`) promise SQLite/`Database` directly.
- `graphrag-core/src/error.rs:5-6`: backend-specific error variant prevents a SQL-free core.
- `capsule_store.rs`, `vector_source.rs`, and `mr.rs` add inherent methods in separate files. Moving only `db.rs` is insufficient.

## Current schema and migrations

`Database::open` (`db.rs:80-91`) creates the parent directory, opens SQLite, calls `init_schema`, then `migrate_schema`. Thus **every read/open can mutate the database**, including export and read-only consumer commands.

`init_schema` (`db.rs:195-319`) creates 14 tables:

1. `stores`: name PK, dimension, SQLite-generated `created_at`.
2. `chunks`: global integer ID, store FK, content/source/metadata, generated timestamp.
3. `chunk_embeddings`: chunk PK/FK, little-endian f32 BLOB.
4. `extractions`: chunk PK/FK, model/count/time checkpoint.
5. `entities`: global integer ID, store FK, display name/type/properties, `UNIQUE(store,name)`.
6. `relations`: global integer ID, store and endpoint FKs, raw/canonical relation, properties.
7. `chunk_entities`: composite PK link.
8. `communities`: global integer ID, store, level, summary/modularity/time. The initial definition omits `parent_id`.
9. `entity_communities`: composite PK link.
10. `relation_synonyms`: global raw→canonical mapping (not store-scoped).
11. `entity_type_synonyms`: global raw→canonical mapping.
12. `entity_embeddings`: entity PK/FK, little-endian f32 BLOB.
13. `entity_merges`: audit rows; target FK; merged display name/time.
14. `capsules`: append-only JSON body/history; unique `(capsule_id, content_fingerprint)`.

Indexes cover store/name/link/canonical/project/parent paths (`db.rs:195-319`). `PRAGMA foreign_keys = ON` is issued at the end of `execute_batch` (`:315`), per connection.

Migration behavior (`db.rs:94-188`):

- No `user_version`, migrations table, ordered migration IDs, or schema-version contract.
- Probes `pragma_table_info` and conditionally adds `communities.parent_id` plus index.
- Probes and conditionally adds `relations.canonical_relation` plus index.
- Unconditionally `CREATE TABLE IF NOT EXISTS` for relation synonyms, entity-type synonyms, entity embeddings, and entity merges; unconditionally ensures three indexes.
- `capsules`, `chunk_embeddings`, and `extractions` are created only by `init_schema`, which always runs before migration.
- Neither init+migration nor an individual multi-statement migration is explicitly transactional. A failure can leave a partially advanced schema.
- Fresh databases also take the migration path for `parent_id`, because init deliberately omits that column.

Backend-neutral parity must preserve or intentionally version these observable contracts: global stable numeric IDs; SQLite timestamp string shape/UTC behavior; list orderings; case/whitespace entity identity; uniqueness/no-op behavior; FK cascades; append-only capsule history; optional JSON stored as strings; little-endian f32 encoding if data interchange retains it.

## Storage capability surface (all concrete methods)

The line ranges below are the complete concrete storage API to split into narrow static traits.

- **Lifecycle/store:** `open` 80; `create_store` 325; `get_store` 345; `list_stores` 361; `delete_store` 379; `store_stats` 391 (`db.rs`).
- **Chunks:** `add_chunk` 418; `list_recent_chunks` 437; `list_chunks` 468; `get_chunks_by_ids` 493 (`db.rs`).
- **Entities/relations/provenance:** `get_or_create_entity` 533; `list_entities` 573; `add_relation` 597; `get_relations_for_entity` 616; `get_entity_by_name` 638; `link_chunk_entity` 658; `get_chunks_for_entity` 666; `get_entities_for_chunk` 678; `list_relations` 702; `get_entity_by_id` 727.
- **Communities:** `clear_communities` 748; `create_community` 755; `link_entity_community` 770; `update_community_summary` 783; `list_communities` 796; `get_child_communities` 821; `get_community_entities` 847; `get_entity_communities` 872.
- **Relation vocabulary:** `get/add/add_many/remove/list` relation synonyms 904-949; lookup-by-canonical 954; refresh 971; relation-type listing 988; canonical relation query 1010.
- **Entity-type vocabulary:** canonical lookup/add/add_many/list/normalize 1043-1087.
- **Entity vectors/merge:** set/get/all/search entity embeddings 1092-1177; `merge_entities` 1185; merge history 1243; merge candidates 1260.
- **Chunk vectors:** set/get/all chunk embeddings (`vector_source.rs:120-160`); `BruteForceVectorSource::for_store` consumes `get_store + all_chunk_embeddings` (`:45-56`).
- **Map-reduce persistence:** pending extraction, persist triple, checkpoint (`mr.rs:131-187`); child summaries and pending summaries (`:292-337`); unclustered count (`:522-530`); entities without embeddings (`:777-806`).
- **Capsules:** the pre-existing backend-neutral `CapsuleStore` trait is in `capsule.rs:305-335`; its SQLite impl methods are `capsule_store.rs:75-169`.
- **Export:** `export_store(&Database,...)` (`export.rs:8-84`) reads store/chunks/entities/relations/communities and excludes embeddings, extraction checkpoints, synonyms, links, merge history, and capsules.

Static capability clusters can follow those actual caller sets; no caller needs one monolithic trait except the top-level service. Avoid retaining inherent methods spread across backend files, because that would keep concrete backend knowledge in algorithms.

## Concrete callers by component

### `graphrag-core` algorithms and helpers

- `export.rs:8-84`: concrete `&Database` over five read capabilities.
- `mr.rs:75-806`: concrete `&Database` throughout planning/apply/status/hierarchy/embed; also embeds SQL-bearing inherent methods as noted above.
- `vector_source.rs:45-160`: concrete database load plus SQL-bearing inherent methods.
- `synonyms.rs:105-108` and `entity_types.rs:139-142`: loaders call batch synonym methods.
- `capsule_store.rs:72-171`: `CapsuleStore for Database`.
- Examples `test_features.rs` and `test_synonyms.rs` open/use `Database`; `lib.rs` docs do likewise.

### CLI

`graphrag-cli/src/main.rs` imports `Database` at 3 and centralizes the usual open at `open_db/open_db_at` 207-214, path `${GRAPHRAG_DATA_DIR:-dirs::data_local_dir()/graphrag}/graphrag.db`. Commands: `note`, `ask`, `log`, `backfill-embeddings`, `enrich`, `export`, `chunk`, `embed`, `status`, `plan`, `apply` (`:25-168`; dispatch `:1000-1082`). Storage consumers:

- note: get/create store, add chunk, set vector, rebuild Leit lexical sidecar (`:255-324`).
- ask: get store, load vectors, list chunks/build BM25, fetch fused IDs (`:374-466`).
- log: store lookup + recent chunks (`:475-499`).
- backfill: store/chunk reads + per-chunk vector writes (`:327-372`).
- enrich/extraction/community persistence: `:509-693`; plan/apply/embed/status: `:697-985`.
- export: `:500-507` opens via the mutating `Database::open`, then `export_store`.
- CLI unit tests also open concrete DB at `main.rs:1137,1152,1180`; integration tests do so in `cli_note.rs`, `mr_cli.rs`, and `cli_enrich.rs`.

Leit today is **not** the authoritative vector store: chunk vectors are SQLite BLOBs and exact-scanned. CLI writes `indexes/{store}.leitseg` for BM25 only (`main.rs:196-204,300-324`), says it cannot reload the bytes yet (`:196-199`), rebuilds on every note, and directly overwrites/removes the sidecar.

### MCP

`graphrag-mcp/src/main.rs` imports concrete `Database` at 7. Server data path is resolved at `:70-84`, stored at `:226,259`, and becomes `graphrag.db` at `:309-311`. Each tool call opens a fresh concrete database (`:1286,1442,1504,1543,1622,1650,1684,1727,1776,1974,2025,2082,2123`). Registered tools (`:398-595`):

- `graphrag_recall`, `graphrag_add_document`, `graphrag_add_code`, `graphrag_summarize_community`;
- `graphrag_stores`, `graphrag_entities`, `graphrag_relations`, `graphrag_communities`;
- `graphrag_search`, `graphrag_query_global`;
- `graphrag_find_merge_candidates`, `graphrag_merge_entities`, `graphrag_embed_entities`.

The MCP helper at `:275` takes `&Database`; recall/search/query use chunk vector, graph, provenance, and community reads. Add paths write store/chunk/vector/entities/relations/links. Summary, embedding, and merge tools mutate their respective capabilities. There are no MCP integration tests in this tree.

### Nushell plugin

`graphrag-nushell/src/main.rs:6-29` fixes a `db_path` using the same environment/default directory convention. Commands register at `:42-59`: create, list, delete, add, search, entities, relations, query, communities, store-communities, get-community, update-summary, list-communities, info. Every storage command opens concrete `Database` per invocation:

- `create.rs:45`; `list.rs:35`; `delete.rs:37`; `add.rs:44`;
- `search.rs:65`; `query.rs:73`; `entities.rs:37`; `relations.rs:39`;
- `communities.rs:52`; `store_communities.rs:73`; `get_community.rs:39,131,179`.

`add.rs:95,160` and `store_communities.rs` helpers accept `&Database`. Search/query use brute-force vectors plus graph/provenance. `store-communities` has its own clear/create/link loop instead of calling the core hierarchy service. There are no Nushell tests in this tree.

### Python / PyO3

- Build contract: `graphrag-py/pyproject.toml` uses maturin 1.9..<2, package/import name `graphrag`, Python >=3.9, manifest-local `extension-module` feature. `build.rs` calls `pyo3_build_config::add_extension_module_link_args`; crate produces a `cdylib`; PyO3 0.29.
- Documented build path at workspace root (`Cargo.toml:12-13`): `maturin build -m graphrag-py/Cargo.toml`. A developer import loop would normally use `maturin develop -m graphrag-py/Cargo.toml` inside a Python venv, then import/smoke-test. Neither a Python test directory nor import smoke script exists.
- Current boundary is **not whole-service erased**: `Store { db: Mutex<Database>, name }` (`graphrag-py/src/lib.rs:99-106`). Constructor creates a directory and opens fixed `graphrag.db` (`:109-121`).
- Python methods and capabilities: `stats` (`:129-156`); hybrid `search` (`:159-213`); `plan` extract/summarize (`:217-264`); `apply` (`:267-316`); communities (`:319-338`); triples (`:341-358`); export (`:361-368`).
- This is the single natural dynamic-erasure seam: replace the concrete mutex payload with one object-safe whole-service facade selected by constructor/config. Do not expose or independently erase each storage capability in Python.

## Tests that lock current behavior

Storage-relevant integration suites and test counts:

- `graphrag-core/tests/export.rs`: 5 — record order/tags, determinism, unknown/empty, excludes embeddings.
- `graphrag-core/tests/capsule_store.rs`: 7 — roundtrip, missing, idempotent fingerprint, newest/history, fingerprint lookup, project filter, validation.
- `graphrag-core/tests/mr_contract.rs`: 27 — plan/apply/checkpoints, summary hierarchy, status, embedding.
- `graphrag-core/tests/vector_source.rs`: 7 — vector encoding/roundtrip, ranking/ties/dimensions/empty/zero vectors.
- `graphrag-core/src/db.rs`: 5 — folded entity identity and recent-chunk ordering/limit.
- `graphrag-core/src/capsule_store.rs`: 1 — corrupted body fingerprint detection.
- CLI: `cli_note` 1, `cli_log` 1, `cli_enrich` 2, `mr_cli` 6, help 1; CLI `main.rs` has 3 unit tests around note/vector/sidecar behavior.
- No migration-from-old-schema tests, crash/rollback tests, backend contract suite, file-backend tests, MCP tests, Nushell tests, Python tests, or cross-backend parity tests exist.

A misleading gap: `mr_contract::apply_is_idempotent` (`:182-198`) asserts only that entity count is unchanged. `persist_triple` always inserts a relation, so re-applying the same result duplicates relations even though the test name/comments claim full idempotency.

## Atomicity and consistency inventory

Single SQLite statements are atomic, but all important journeys span multiple statements or files and have no enclosing transaction:

| Journey | Evidence | Partial-failure state today |
|---|---|---|
| Open/init/migrate | `db.rs:80-188,195-319` | partially initialized/upgraded schema |
| Note/add document | CLI `:275-297`; MCP `:1286-1370`; Nushell `add.rs:44-203` | chunk without embedding; DB committed while Leit sidecar stale/missing; extracted graph partly written |
| Leit sidecar rebuild | CLI `:300-324` | direct overwrite is not temp+fsync+rename; delete/write failure leaves stale/absent cache |
| Backfill vectors | CLI `:346-371` | prefix of chunks embedded |
| `persist_triple` | `mr.rs:156-168` | one/both entities, relation, or provenance links only |
| `apply_extract` | `mr.rs:111-127` | prefix of triples stored without checkpoint; retry duplicates relations |
| Entity merge | `db.rs:1185-1238` | relations/links/community/audit/source deletion can split halfway; highest integrity risk |
| Community rebuild | `mr.rs:580-721`; CLI `:509-693`; Nushell `store_communities.rs:73-166` | old communities cleared before replacement is complete |
| Batch synonym loads | `db.rs:924-929,1063-1068` | prefix applied |
| Apply embed/summary | `mr.rs:277-287,764-772` | prefix applied (documented partial/out-of-order semantics, but no batch atomic option) |
| Capsule put/checkpoint/vector setters | single upsert/insert statements | statement-atomic |

A file backend must define equivalent operation boundaries explicitly. “Plain files authoritative” requires temp-file + flush/sync + atomic rename (and a recovery/manifest protocol for multi-file commits), rather than reproducing the CLI's current direct sidecar write. Leit indexes remain rebuildable derived state and must never participate as the source of truth or determine commit success.

## Mechanical migration seams

1. **Move domain types first:** `Store`, `Chunk`, `Entity`, `Relation`, `EntityInput`, `CommunityRecord` out of `db.rs` into SQL-free core modules without changing fields/serde behavior.
2. **Remove backend error leakage:** map backend failures into a core storage error boundary; move `From<rusqlite::Error>` to `graphrag-sqlite`.
3. **Extract narrow static capabilities from caller clusters:** lifecycle/store; chunk/provenance; graph; communities; vocabularies; chunk vectors; entity vectors/merge; extraction checkpoints; capsules. Algorithms receive generic capability bounds/associated types, not `Database`.
4. **Move every SQL-bearing implementation together:** all of `db.rs` implementation plus SQL/inherent blocks from `mr.rs`, `vector_source.rs`, and `capsule_store.rs`, and schema/migrations, into `graphrag-sqlite`. Leaving any one behind violates “SQL entirely in graphrag-sqlite.”
5. **Make services static compositions:** export, map-reduce, hierarchy/status, exact vector source, synonym loaders become generic core functions/services. Keep Leit adapters as derived index capabilities over IDs/metadata/text/vectors.
6. **Introduce constructor/composition roots:** CLI, MCP, and Nushell select concrete SQLite/file service at startup and pass it inward; eliminate repeated direct `Database::open`. Preserve existing default path/behavior during the first parity slice.
7. **Erase once at PyO3:** expose an object-safe whole-service API and put only that behind the Python mutex/box. Internal Rust service/backend composition stays static.
8. **Add authoritative file backend behind the same contract suite:** never write SQL-shaped files merely to mimic SQLite. Define versioned manifests, canonical domain records, stable IDs, timestamps, atomic commit/recovery, locking, and deterministic enumeration.
9. **Separate index lifecycle:** replace `rebuild_lexical_sidecar`'s direct write with derived-index rebuild/update APIs. Leit may index metadata, BoW, and embeddings; deletion/corruption must be recoverable solely from authoritative backend records.
10. **Cut consumers mechanically:** core tests → backend-parameterized contract tests; CLI → service composition; MCP/Nushell → service; Python last, after whole-service facade is proven. Remove `Database` re-export and rusqlite dependency from core only after grep gates are clean.

Useful grep gates:

```sh
rg 'rusqlite|Connection|pragma_table_info|CREATE TABLE|SELECT |INSERT |UPDATE |DELETE ' graphrag-core
rg '\bDatabase\b|Database::open' graphrag-{core,cli,mcp,nushell,py}
rg 'std::fs::(write|remove_file)' # distinguish authoritative files from derived indexes
```

## Minimal parity journeys for TDD

Run each contract against SQLite and plain-file implementations; run service journeys once per composition. These are the smallest set that covers observed consumers without pretending every method is an independent architecture.

1. **Open/create/reopen/migrate:** new backend; create store; reopen; retain dimension/timestamps; SQLite fixture from each known old shape migrates transactionally and idempotently; read-only open/export does not silently migrate unless explicitly requested.
2. **Authoritative note commit:** create-or-get store + add chunk + vector as one logical commit; injected failure at each step leaves either old state or complete note; derived Leit failure does not lose committed note and rebuild repairs it.
3. **Hybrid recall:** persisted chunks/vectors/metadata → exact vector and Leit BoW candidates → deterministic RRF → fetch IDs in requested result order (current SQL `IN` query does not itself preserve input order, so service mapping must).
4. **Graph extraction commit:** entities + raw/canonical relation + both chunk links + checkpoint atomically; exact reapply creates no duplicate relation; zero-yield checkpoint remains terminal.
5. **Community replacement:** build hierarchy in staging/transaction, atomically replace old graph, preserve one-root/parent/entity-membership invariants; failed rebuild leaves old hierarchy readable.
6. **Entity merge:** transfer both relation directions, deduplicate chunk/community links, record audit, delete source atomically; injected failure rolls back.
7. **Capsule history:** exact canonical bytes/fingerprint validation, idempotent `(id,fingerprint)`, newest/history/filter ordering, corruption detection; append commit survives reopen.
8. **Export:** deterministic current JSONL parity and explicit exclusion set; export against a copied/readonly SQLite DB proves no schema mutation. Decide separately whether a future full backup format includes presently omitted authoritative data.
9. **Vocabulary + vectors:** global synonym behavior, canonical refresh, typed dimension errors, little-endian roundtrip or a versioned file representation, deterministic cosine/tie ordering.
10. **Consumer smoke matrix:** CLI note/log/ask/plan/apply/enrich/export; one MCP add→recall→graph query; one Nushell create→add→search→community read; Python maturin build/import then `Store.stats/search/plan/apply/communities/triples/export` against both backend selections.
11. **Index disposability:** delete/corrupt every Leit artifact; authoritative records remain readable; rebuild metadata/BoW/embedding indexes and recover equivalent ranked IDs within the index's documented determinism.
12. **Concurrency/locking:** two readers + one writer and two competing writers for each backend; no torn JSON/manifest, lost ID allocation, partial merge, or index promoted ahead of authoritative commit.

## Principal risks to lock before implementation

- Trait granularity can become either a monolithic `Database` renamed as a trait or dozens of independently erased repositories. The approved seam is narrow **static** capabilities internally and one whole-service erased object only in Python.
- Existing IDs and timestamps depend implicitly on SQLite. A plain-file backend needs an owned, versioned allocation/time contract or parity will drift silently.
- Current tests mostly prove happy-path final state, not rollback/recovery. Moving code without failure injection would preserve known non-atomic behavior.
- Current extraction “idempotency” is incomplete and tests do not count relations.
- Export is lossy and opening the source mutates schema. It is not currently a backup/migration format.
- MCP/Nushell duplicate orchestration (especially add/community paths), so switching only the CLI composition root leaves concrete storage coupled elsewhere.
- Leit sidecars are currently write-only from the CLI and do not contain canonical vector truth. Treating them as authoritative during migration would reverse the approved architecture.
- `Mutex<Database>` in Python bakes in one backend and SQL concurrency characteristics; per-capability trait objects would expose internal composition and create an unstable Python ABI seam.
- Absolute Leit path dependencies and absent Python/Nushell/MCP test harnesses are independent reproducibility gaps that can obscure storage regressions.
