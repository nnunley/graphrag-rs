# GraphRAG Memory API Notes

Date: 2026-06-01

## Repo note

This repo currently lives at `~/nu_libs/graphrag-rs`.
It should be moved under `~/development` so it is not effectively orphaned from the main project workspace.

## Direction

The `brain` compatibility API should not be the long-term read surface for `graphrag-rs`.

- Keep write compatibility where it is useful.
- Move reads to new native verbs instead of pretending `brain` and `graphrag-rs` are equivalent.
- Treat the CLI as human-facing.
- Treat `graphrag-mcp` as the semantic surface for LLMs and agents.

## Compatibility policy

### Writes

- Preserve compatibility-oriented writes such as `note`.
- A compatibility write can remain the stable front door while the internal storage and enrichment model becomes more semantic.

### Reads

- Do not keep the main read path trapped behind `brain.ask`-style semantics.
- Move to native read verbs that expose GraphRAG concepts directly.
- Native verbs should be able to express richer retrieval than "search notes".

## CLI behavior

The CLI is for humans.

- `note` should record durably first.
- Inline work after the write should depend on latency and compute budget.
- Human-visible feedback should favor simple progress and next-step guidance over machine-oriented structure.

### Preferred `note` behavior

The default should be adaptive:

1. Record the note immediately.
2. Spend any remaining local budget on cheap enrichment work.
3. If work remains, leave it in backlog lanes.
4. Optionally hand work to a local worker, delegated agent, or remote executor.

The target is to stay under human-perceivable response thresholds when possible.
Rough upper bound: about 1-2 seconds, with a busy notification if work continues.

## MCP behavior

`graphrag-mcp` should expose semantic hints to the driving LLM.

- The MCP layer should provide machine-usable hints rather than CLI-style prose.
- It should be able to recommend the next semantic operation.
- It should optionally delegate heavier work to an external provider when configured.

## Processing model

The enrichment pipeline should not be a single FIFO queue.

Use separate lanes with a shared urgency model.

Suggested lanes:

- write lane
- cheap enrich lane
- semantic lane
- global lane

### Lane intent

- `write lane`: durable append, minimal latency
- `cheap enrich lane`: chunk normalization, embeddings, lexical updates
- `semantic lane`: entities, relations, linking, abstraction
- `global lane`: community updates, summaries, cross-note synthesis

## Urgency

Separate lanes should be governed by an urgency meter.

High backlog is itself a problem, so urgency should reflect more than age alone.

Suggested urgency inputs:

- queue depth
- oldest item age
- semantic staleness
- dependency blocking
- probable retrieval impact

## Worker model

Background workers should be optional.

The service model can look similar to Headroom:

- spawn on demand
- stay open after first call
- or point to a remote worker/provider

Foreground calls should still be able to opportunistically drain backlog within budget even when no worker is running.

## Architectural stance

The likely shape is:

- a compatibility-oriented write surface
- a native semantic read surface
- adaptive inline enrichment
- optional local worker
- optional delegation/remote execution

That preserves compatibility where it matters while letting `graphrag-rs` grow into a more semantically interesting system than `brain`.
