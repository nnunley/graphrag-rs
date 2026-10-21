# graphrag-rs

A local memory for notes and documents, built on the GraphRAG pipeline
([Edge et al., *From Local to Global*, arXiv:2404.16130](https://arxiv.org/abs/2404.16130)).
Everything lives in one SQLite graph. You add text; a background worker
chunks it, embeds it, extracts entities and relationships with a local LLM,
clusters the entity graph into communities, and summarizes them. Recall is
hybrid: vector, lexical and graph lanes fused with reciprocal rank fusion.

Surfaces: the `graphrag` CLI, an MCP server (`graphrag-mcp`), a Nushell
plugin and Python bindings. They all read and write the same database.

## Install

```sh
cargo install --path graphrag-cli    # the `graphrag` CLI (includes the worker)
cargo install --path graphrag-mcp    # the MCP server
```

Extraction and summaries need an OpenAI-compatible chat endpoint (llama.cpp's
`llama-server` works). Embeddings run locally with fastembed; the model
downloads on first use.

## Quick start

```sh
graphrag note "jj is a version control system that works with git repositories"
graphrag note --group rust "tokio is an async runtime for Rust"
graphrag ask "which VCS works on top of git"
graphrag status
```

`note` returns at once and starts the worker in the background. A note is
searchable by vector and BM25 immediately; it joins the graph once the worker
has extracted it.

## How it works

The pipeline follows the paper, with every step fingerprinted so it reruns
only when its inputs change:

1. **Documents.** Each added text is stored once, content-addressed and
   immutable. Identical text is deduplicated. Text added under the same
   `doc_key` (a file path or page id) becomes a new version that replaces the
   old one; notes without a key are append-only.
2. **Chunks.** A chunk is a span of its document (about 2,000 characters with
   200 overlap), never a copy of the text. Markdown is split along its heading
   tree, so a chunk never crosses a section, and recall shows its heading path.
   Code is chunked on syntax boundaries.
3. **Embed.** Each chunk gets a 768-dimension Nomic embedding.
4. **Extract.** The LLM reads each chunk and returns entities and
   relationships. Relation names are normalized to a canonical vocabulary
   (Dublin Core and Schema.org terms, plus `isa`/`genls` for classes).
5. **Generalize.** Entity names are folded to base forms and class
   (`isa`/`genls`) edges are lifted where the evidence is statistically
   significant.
6. **Communities.** Hierarchical Leiden over the entity graph, as in the
   paper: entities are nodes, relationships are edges, and an edge's weight is
   the number of distinct sections that support it. Chunks are evidence, not
   nodes.
7. **Summaries.** Community reports are written bottom-up from the entity and
   relationship descriptions, and reused while a community is unchanged.

**Groups** replace the old per-purpose stores. A document belongs to one or
more groups (`conversations` by default, `code` for code); groups can be part
of other groups, and `--in <group>` scopes recall to a group and its
subgroups. Documents, sections and groups are graph nodes too, but by default
they don't shape the communities.

Raw notes are never rewritten or deleted automatically. Merges between
entities are recorded as retractable `same_as` assertions; nothing is deleted.

## CLI

| Command | What it does |
|---|---|
| `note <text> [--group G]... [--doc-key K] [--source S]` | Add a note (default group `conversations`) |
| `ask <query> [--top N] [--in G]... [--lanes vector,lexical,graph] [--json]` | Hybrid recall; `--json` prints per-lane scores |
| `log [--limit N]` | Recent notes |
| `status [--json]` | Progress per group and step, backlog, quality counters |
| `work [--once] [--max-items N] [--retry-failed] [--accept-reextract]` | Run the worker in the foreground |
| `work --pause` / `work --resume` | Stop or restart background workers |
| `backfill-embeddings` | Embed chunks whose embedding is missing or stale |
| `chunk <path> [--format spans\|text\|review\|json] [--code]` | Preview chunk spans for a file (no database; this tool doesn't split at headings the way ingestion does) |
| `migrate v2 --plan` / `--apply --plan-fp FP` | One-time upgrade from per-store databases (see below) |
| `export memory` | Dump the graph as JSON Lines |

`plan`, `apply`, `provenance` and `embed` are the older manual map-reduce
path, for running extraction with an external executor. `enrich` is a
deprecated alias for `work`.

In the `status` table each cell is `current/stale/missing/failed`.

## The worker

Writes (`note`, MCP adds) and recalls start `graphrag work` in the
background; a second worker exits at once, so starting one is always safe.
Work is ordered by priority: chunks a recall just touched first, then new
chunks, then the backlog.

When the LLM endpoint is unreachable the worker backs off, keeps doing the
local steps (chunking and embedding), and leaves extraction for the next run.
If a model or prompt change would supersede more than 10% of existing
extractions, it refuses until you pass `--accept-reextract`.

Exit codes: `0` done (or already running), `64` refused (needs
`--accept-reextract`), `69` LLM endpoint or model unavailable, `70` local
error, `75` LLM outage.

For a periodic backstop on macOS, see [`scripts/launchd/README.md`](scripts/launchd/README.md).

## Configuration

| Variable | Default | Used for |
|---|---|---|
| `GRAPHRAG_DATA_DIR` | `~/Library/Application Support/graphrag` (macOS), `~/.local/share/graphrag` (Linux) | Where `graphrag.db` lives |
| `GRAPHRAG_LLM_BASE_URL` | `http://ndn.lan:8080/v1` | OpenAI-compatible chat endpoint for extraction and summaries |
| `GRAPHRAG_LLM_MODEL` | `qwen3.8-27b` | Model name sent to the endpoint |
| `GRAPHRAG_LLM_CONCURRENCY` | `4` | Parallel LLM requests |
| `GRAPHRAG_EMBED_MODEL` | `nomic` | `nomic` (768) or `minilm` (384); the CLI also accepts `openai`/`openai3` |
| `GRAPHRAG_EMBED_CACHE_DIR` | `~/Library/Caches/graphrag/fastembed` | Embedding model cache |
| `GRAPHRAG_OPENAI_API_KEY`, `GRAPHRAG_OPENAI_BASE_URL` | none, `https://api.openai.com/v1` | Only for the OpenAI embedders |
| `GRAPHRAG_BIN` | `graphrag` on `PATH`, then `~/.cargo/bin/graphrag` | Binary started for background work |
| `GRAPHRAG_STORE` | `memory` | MCP server only; leave as is |

One embedder serves the whole database, so changing `GRAPHRAG_EMBED_MODEL`
marks every embedding stale.

## MCP server

`graphrag-mcp` speaks MCP over stdio and takes no arguments. For Claude Code:

```sh
claude mcp add graphrag -- graphrag-mcp
```

The main tools:

- `graphrag_recall` (`query`, `top`, `in`): hybrid recall from text.
- `graphrag_add_document` (`content`, `doc_key`, `groups`, `source`) and
  `graphrag_add_code` (`content`, `file_path`, `doc_key`, `groups`): add text;
  they return at once and start the worker.
- `graphrag_status`: the same report as `graphrag status`.
- `graphrag_query_global` (`query`): map-reduce over community summaries for
  corpus-wide questions.
- `graphrag_communities`, `graphrag_summarize_community`, `graphrag_entities`,
  `graphrag_relations`: browse the graph.
- `graphrag_find_merge_candidates`, `graphrag_merge_entities`: find and merge
  duplicate entities (a retractable `same_as`).

`graphrag_search` takes a precomputed embedding; use `graphrag_recall` for
text. A `store` argument, where accepted, is a deprecated alias for a group.

## Nushell

```sh
cargo build --release -p nu_plugin_graphrag
plugin add target/release/nu_plugin_graphrag
plugin use graphrag
```

```nushell
{content: "tokio is an async runtime", source: "notes"} | graphrag add rust
graphrag list                    # groups with document counts
graphrag store-communities memory  # build or reuse the current community run
graphrag info                    # data directory and database path
```

`add` takes `content` and optionally `groups`, `source`, `doc_key`, `title`,
and `entities` (a list of `{head, relation, tail}` recorded as assertions).
It doesn't start the worker; run `graphrag work` or rely on the launchd
backstop.

The remaining commands (`create`, `delete`, `search`, `query`, `entities`,
`relations`, `get-community`, `list-communities`, `update-summary`) still have
the older per-store signatures; `search` and `query` need a precomputed
embedding. **`graphrag delete memory` deletes the whole graph.**

## Python

```sh
maturin build --release -m graphrag-py/Cargo.toml
pip install target/wheels/graphrag-*.whl
```

```python
import graphrag
s = graphrag.Store("~/Library/Application Support/graphrag")
s.add_document("tokio is an async runtime", groups=["rust"])
for hit in s.search("async runtime", top=5):
    print(hit.rrf, hit.groups, hit.text)
print(s.status_report())   # JSON
```

Like Nushell, `add_document` doesn't start the worker.

## Upgrading from a per-store database

Opening an older database with this version converts it in place (each old
chunk becomes a document plus one chunk spanning it), and there is no
automatic backup. Back up first:

```sh
cd ~/Library/Application\ Support/graphrag
sqlite3 graphrag.db ".backup graphrag-pre-v2.db"
```

Then merge the old stores into one graph:

```sh
graphrag migrate v2 --plan          # read-only; prints the plan and its fingerprint
graphrag migrate v2 --apply --plan-fp <fingerprint>
graphrag work --accept-reextract    # re-extract everything with the current prompt
```

`--apply` runs in one transaction and refuses if the plan changed since you
reviewed it, or if any source can't be resolved.

## Evaluating retrieval

[`scripts/retrieval-eval/`](scripts/retrieval-eval/README.md) scores recall@10
and MRR per query type (known item, paraphrase, identifier, cross-group) for
any combination of lanes, using `graphrag ask --json`. Keep the query set
outside the repository if it quotes private notes.

## Development

```sh
cargo build --release
cargo test --release
```

The `docs/` directory holds design notes and plans; they record why things
are the way they are and aren't needed to use the tool.
