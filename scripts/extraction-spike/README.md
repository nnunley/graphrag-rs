# Extraction prompt spike

The blind two-judge comparison used to choose the extraction profile (see
`docs/2026-10-19-ingestion-workflow-design.md`, "Extraction profile spike").
It reads the graphrag DB read-only and writes everything under this directory's `out/`.

1. `python3 -I run.py`: samples 3 chunks from each of 4 stores (blake2b order,
   300–6,000 characters) and runs every arm in `ARMS` against llama.cpp (4 at a time).
2. `python3 -I pack.py`: parses the outputs, prints per-arm mechanical metrics and
   writes `packets.txt` (outputs shuffled under X/Y/Z labels) plus `blind_key.json`.
3. Run the two judges on `judge_prompt.md` + `packets.txt`, writing their JSON to
   `claude_judge.json` and `codex_judge.json`:
   - Claude: a fresh subagent told not to open `blind_key.json`, `metrics.json`, the
     scripts or `out/`.
   - Codex: `codex exec --skip-git-repo-check -m gpt-6-sol -s read-only -C . -o codex_judge.json "$(cat judge_prompt.md)"`
     (the configured default model `gpt-6.1-sol` isn't available on a ChatGPT account).
4. `python3 -I score.py claude_judge.json codex_judge.json`: unblinds and prints scores by arm and store.

Outputs (`out/`, `packets.txt`, `*_judge.json`, `blind_key.json`, `metrics.json`) are not committed.

## Production-size re-spike (plan Task 20)

Arms `C_prod` (arm C's prompt), `V11_prod` (`paper-json-v1.1`, the shipped prompt)
and `V11_small` (v1.1 at 1,000/100). Each source chunk is cut by the production
chunker and the pieces' extractions are merged per source chunk, so quality and
timing are measured at production size. Prompt and chunker come from
`target/release/examples/spike_tools` (`cargo build --release -p graphrag-worker --example spike_tools`).

```sh
export SPIKE_DB=<a .backup copy> SPIKE_OUT=out2 SPIKE_PACK=v11_ SPIKE_ARMS=C_prod,V11_prod,V11_small
python3 -I run.py && python3 -I pack.py
# judges read v11_packets.txt; then
python3 -I score.py v11_claude_judge.json v11_codex_judge.json
python3 -I classes.py   # distinct classes per 100 entities, Person-typed tools, example handles
```
