# Storage architecture decision record

**Date:** 2026-08-12  
**Status:** frozen for roadmap planning; implementation remains milestone-gated  
**Applies to:** GraphRAG storage abstraction, file authority, Leit derivation, embedding custody, and consumer migration

## Context

The current implementation exposes a concrete SQLite `Database` from `graphrag-core`; CLI, MCP, PyO3, Nushell, tests, and examples call it. Current Leit integration can build an in-memory BM25 index and serialize segments, but the inspected revision cannot reload/search persisted segments and has no vector index. `graphrag-py` already exists and must remain an early dogfood surface.

The architecture target is narrow static storage capabilities inside Rust, backend-neutral domain values in core, SQLite and ordinary-file implementations outside core, disposable Leit-derived indexes, and dynamic dispatch only around a fully composed service at the PyO3 boundary.

Norman selected all presented recommendations except consumer migration, where he explicitly selected the big-bang option.

## Decisions

### DL-009 — first durable contract: B

Freeze a versioned vertical-slice subset: only corpus/source/chunk/head/index-binding identifiers and codecs needed by the first storage journey. Future evidence types are additive versions.

**Consequences**
- The walking skeleton persists real versioned production data, not test-only IDs.
- This slice does not pre-create the complete immutable-evidence registry.
- A `conformance-fixture-v1` identity cannot later be relabeled as production `evidence-v1`.

### DL-023 — plain-file authority and commit model: D

Use the hybrid of managed immutable records and external content in place. User/source files remain ordinary files; managed immutable evidence/metadata records and atomic generation manifests live beside them.

**Consequences**
- Existing folders can be indexed without rewriting their source files.
- Managed records bind logical identities and commitments to source bytes.
- `AS-005` must freeze canonical bytes, layout, locking, durability claims, corruption behavior, and the atomic visibility transition before implementation claims correctness.
- Leit artifacts remain outside the authoritative set.

### DL-024 — Leit integration: A target, C staged

Target one common Leit-derived retrieval layer for both SQLite and file backends. Stage delivery through today's concrete Leit wrappers; do not block GraphRAG's walking skeleton on a universal Leit interface, persisted executable segments, vector indexing, or Leit PR #18.

**Consequences**
- Both backends emit the same domain indexing records and parity oracle.
- Initial lexical indexes may rebuild in memory.
- Persistent executable segments and vector search remain owned Leit milestones rather than fabricated GraphRAG capabilities.
- The staged adapter must not harden accidental concrete APIs into the long-term contract.

### DL-025 — embedding custody: B

Retain exact embedding vectors and their model/profile identity in each authoritative backend. Leit indexes those vectors but is not their sole custodian.

**Consequences**
- Deleting Leit artifacts cannot delete the only copy of an observation.
- Rebuilds do not require a provider/model that may have drifted or disappeared.
- Vectors are retained derived observations associated with chunks; Leit search structures remain disposable.

### DL-022 — consumer migration: C (explicit override)

Cut over Python, CLI, MCP, Nushell, tests, and examples together in one self-contained, independently revertible commit. This overrides the recommendation for surface-by-surface shadow migration.

**Consequences**
- No consumer surface lands early and no shadow comparison window exists.
- `AS-006` backend parity/recovery and all surface-native compatibility tests are hard pre-cutover gates.
- `AS-004` and `AS-007` prove the PyO3 boundary through experimental/test-only seams; they do not partially migrate the public Python surface.
- `AS-008` owns the all-consumer change, legacy core SQL evacuation, and complete caller inventory update.
- Rollback proof applies/reverts the exact cutover commit in a disposable worktree and validates the parent digest and legacy suite.
- This VCS rollback proof does not assert that post-deployment data is downgrade-compatible; live migration and rollback remain separately approval-gated.

## Confirmed boundary associated with DL-003

- `graphrag-core` owns domain values, narrow capability contracts, and generic use cases.
- Concrete SQLite and file implementations live outside core.
- Internal Rust composition uses static dispatch over complete services.
- `graphrag-py` may dynamically erase exactly one fully composed service; Python does not receive independently composable storage traits.
- Exact crate/module names are implementation details pending the first architecture review, not persisted contract.

## Approval boundary

This record freezes planning choices. It does not approve dependencies, schemas, implementation edits, live database access, data migration, backend-default changes, release, push, or deployment. Each `AS-*` milestone in the TDD roadmap requires its own evidence-locked GO.
