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
