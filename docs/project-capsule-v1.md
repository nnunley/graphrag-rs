# Project Capsule v1

A **capsule** is the synthesis layer's resumable answer to: *what is this
project, what is verified, what was decided, what matters now, and where do I
resume?* Capsules are small typed values; large context stays in the graph.
Other systems (e.g. the laneq Attention Steward) carry a `CapsuleRef`, never
the capsule body.

## Types (`graphrag_core::capsule`)

| Type | Role |
|---|---|
| `ProjectCapsuleV1` | The capsule body: identity, purpose, verified state, decisions, open threads, commitments, next segment, evidence, freshness. |
| `CapsuleRef` | Compact validated pointer: `schema_version`, `uri`, `kind`, `capsule_id`, `project_id`, `content_fingerprint`, `generated_at`. |
| `CapsuleKind` | `project` (implemented), `segment`, `portfolio` (reserved ref kinds). |
| `EvidenceItem` | Citable observation: stable `evidence_id`, `uri`, `observed_at`, optional `fingerprint`. |
| `Freshness` | `generated_at`, `source_fingerprint`, optional `stale_after`, ordered `input_fingerprints`. |
| `CapsuleError` | Total typed validation failure. |

## Contract

- **URI**: exactly `graphrag://capsules/v1/{kind}/{capsule_id}`.
- **IDs**: nonempty ASCII `[A-Za-z0-9._-]`, max 128 bytes.
- **Fingerprints**: `sha256:` + 64 lowercase hex.
- **Timestamps**: RFC 3339 strings (shape/range checked; leap-second and
  month-length edge cases beyond day 31 are not calendar-validated).
- **Evidence discipline**: `verified_state` and `decisions` entries must cite
  at least one evidence ID, and every citation must resolve to a unique entry
  in `evidence`. Open threads and commitments carry explicit
  `status`/`owner`/`next_action` fields instead of prose-only blobs.

## Determinism

`reference()` validates, serializes the capsule with `serde_json::to_vec`, and
fingerprints those exact bytes with SHA-256. All schema types use fixed named
fields (serialized in declaration order) and vectors — never maps — so equal
values always serialize to equal bytes. **Vector order is semantically
significant**: it is the canonical input order, and reordering produces a
different capsule fingerprint by design.

Non-guarantee: no canonicalization of semantically-equal prose; the
fingerprint identifies bytes, not meaning.

## Cross-language fixture

Go consumers (agent-sandbox `queue.CapsuleRef`) test this exact JSON:

```json
{"schema_version":1,"uri":"graphrag://capsules/v1/project/graphrag-rs-current","kind":"project","capsule_id":"graphrag-rs-current","project_id":"graphrag-rs","content_fingerprint":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","generated_at":"2026-08-07T00:00:00Z"}
```

`CapsuleRef` serde roundtrips this byte-exactly (field order is struct
declaration order).

## Persistence

`CapsuleStore` (in `graphrag_core::capsule`) is the provider contract:
idempotent `put_capsule` keyed by `(capsule_id, content_fingerprint)`,
`latest_capsule`, `capsule_by_fingerprint`, newest-first `capsule_history`,
and project-filtered `list_capsules`. The SQLite provider
(`graphrag_core::capsule_store`, implemented on `Database`) stores the exact
canonical JSON bytes that were fingerprinted in an append-only `capsules`
table and verifies the fingerprint on every read before deserializing.

## Out of scope for v1

LLM synthesis of capsule content, stale refresh triggers, and
segment/portfolio capsule bodies are separate segments.
