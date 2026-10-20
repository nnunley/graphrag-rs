# Retrieval eval

One frozen query set, one scorer, many arms. Every retrieval change (chunk
size, lanes, fusion, embedders, rerankers, PixelRAG-style screenshot
embeddings) is an arm scored on the same queries.

- `docs/retrieval-eval-queries.jsonl` is the frozen set. Relevance is an exact
  quoted span, so it survives re-chunking and migration. A new version gets a
  new file, so old scores stay comparable.
- `make_queries.py` built it from a read-only DB copy with qwen3.8-27b on
  llama.cpp (known-item, paraphrase, identifier, cross-group).
- `run.py` runs one arm (`graphrag ask --json`, or the pre-v2 binary with
  `--old-stores`), `score.py` prints recall@5/10, MRR@10, nDCG@10 and latency
  per query kind.

Never point `--data-dir` at the live DB: a v2 binary upgrades the schema on open.
Use `sqlite3 <live> ".backup <copy>"`.

```sh
python3 -I run.py --queries ../../docs/retrieval-eval-queries.jsonl --arm v1-baseline \
  --bin ~/.cargo/bin/graphrag --data-dir COPY_DIR --old-stores conversations,operation,mica-docs,graphrag-paper,9front-kb
python3 -I run.py --queries ../../docs/retrieval-eval-queries.jsonl --arm v2-all --bin target/release/graphrag --data-dir V2_COPY
python3 -I run.py --queries ... --arm v2-lexical --lanes lexical ...
python3 -I score.py --queries ../../docs/retrieval-eval-queries.jsonl out/*.jsonl
```
