"""Run one retrieval arm over the frozen query set.

  python3 -I run.py --queries ../../docs/retrieval-eval-queries.jsonl --arm NAME \
      --bin PATH --data-dir DIR [--lanes vector,lexical,graph] [--old-stores s1,s2,...]

New binaries are called with `ask --json --top 10`. For the pre-v2 baseline,
pass --old-stores: the old binary has no --json and searches one store at a
time, so each store is queried and the hits are merged by RRF score.
Writes out/<arm>.jsonl with each query's ranked hits and latency.
"""
import argparse, json, os, re, subprocess, time

def new_hits(a, q):
    cmd = [a.bin, "ask", q, "--top", "10", "--json"]
    if a.lanes:
        cmd += ["--lanes", a.lanes]
    out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "GRAPHRAG_DATA_DIR": a.data_dir, "GRAPHRAG_BIN": "/usr/bin/true"})
    if out.returncode != 0:
        raise SystemExit(out.stderr)
    return [{"source": h.get("source"), "content": h["content"]} for h in json.loads(out.stdout or "[]")]

LINE = re.compile(r"^#\d+\trrf=([0-9.]+)\t.*?\t\[([^\]]*)\]\t(.*)$")

def old_hits(a, q):
    merged = []
    for store in a.old_stores.split(","):
        out = subprocess.run([a.bin, "ask", q, "--top", "10", "--store", store], capture_output=True, text=True,
                             env={**os.environ, "GRAPHRAG_DATA_DIR": a.data_dir})
        # Hits start with "#N\trrf="; chunk text can span several lines.
        for line in out.stdout.splitlines():
            m = LINE.match(line)
            if m:
                merged.append((float(m.group(1)), {"source": m.group(2), "content": m.group(3)}))
            elif merged and not line.startswith("#"):
                merged[-1][1]["content"] += "\n" + line
    merged.sort(key=lambda x: -x[0])
    return [h for _, h in merged[:10]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lanes")
    ap.add_argument("--old-stores")
    a = ap.parse_args()
    os.makedirs("out", exist_ok=True)
    with open(a.queries) as f, open(f"out/{a.arm}.jsonl", "w") as w:
        for line in f:
            q = json.loads(line)
            t = time.time()
            hits = old_hits(a, q["query"]) if a.old_stores else new_hits(a, q["query"])
            w.write(json.dumps({"id": q["id"], "ms": round((time.time() - t) * 1000), "hits": hits}) + "\n")
    print("wrote", f"out/{a.arm}.jsonl")

if __name__ == "__main__":
    main()
