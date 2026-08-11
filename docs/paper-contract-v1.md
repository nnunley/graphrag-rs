# Paper workflow contract v1

This document describes the provider-independent M0 wire boundary for the paper-profile workflow. The normative Rust types live in `graphrag-core::token` and `graphrag-core::paper`.

## Ownership

Rust plans canonical map and reduce inputs, persists their fingerprints, and validates returned traces. A notebook or Python executor treats `input_context`, `input_fingerprint`, IDs, profile, and tokenizer identity as opaque plan data. It calls the provider and returns the raw response plus parsed fields. It does **not** recreate Rust fingerprints or choose a different context.

The executor must count text with the exact `TokenizerId` named by the plan. Byte or character counts must not be labelled as model tokens. Rust recomputes every declared count through the matching `TokenCounter`; JSON numbers are not trusted.

## JSONL source input

One UTF-8 JSON object per line:

```json
{"schema_version":1,"document_id":"doc-1","text":"Exact source text."}
```

Unknown schema versions, blank IDs, and blank text are invalid. `SourceDocumentV1::fingerprint` is the canonical source identity.

## Planned calls

A map plan is `MapBatchV1`. Core derives a deterministic SHA-256-keyed permutation from the recorded seed, then greedily packs the maximal adjacent report groups that fit. Its prompt version, response-schema version, ordered report IDs, canonical `input_context`, composite SHA-256 `input_fingerprint`, and checked `input_token_count` bind the call. `MapResultV1` echoes those versions and the input fingerprint. Its strict v1 JSON response payload binds the parsed score, answer, and citations to the stored raw bytes.

A reduce plan is `ReduceInputV1`. Rust ranks positive map answers, filters score-zero answers, renders the maximal prefix that fits the reduce window, and creates exactly one input. The context includes the selected map answers and their raw-response hashes. Prompt/schema versions participate in the input fingerprint. `ReduceResultV1` echoes that exact versioned input. Its strict v1 JSON response payload binds the parsed answer and citations to the stored raw bytes.

`QueryTraceV1::validate_against_cut` requires the authoritative report snapshot separately. This prevents a trace from redefining “exhaustive” by deleting the same report from its inventory, order, batches, and results. Every positive map answer cites at least one report; the reducer may cite only reports actually cited by selected map answers. Each authoritative fixture report resolves both entity references and exact source-document fingerprints.

## Fingerprints

All fingerprints are lowercase `sha256:<64 hex>`. `fingerprint_text` hashes exact UTF-8 bytes. `fingerprint_parts` hashes a sequence of eight-byte big-endian lengths followed by exact part bytes. Executors should echo Rust-produced identities rather than reimplement this framing.

## Capability claims

M0 means only the versioned contracts, deterministic fake executor/oracle, and validators above are implemented. M1–M6 remain gated. M0 does not claim provider execution, storage migration, recursive Leiden, durable community reports, paper global search, or empirical reproduction.
