"""Score arms: recall@5/10, MRR@10, nDCG@10 (binary gains), latency, per kind.

  python3 -I score.py --queries ../../docs/retrieval-eval-queries.jsonl out/*.jsonl

A hit is relevant to a target when the target's anchor text occurs in the
hit's content (whitespace-normalized). Sources are not compared: they change
across migration while the text doesn't.
"""
import argparse, json, math, re, statistics
from collections import defaultdict

def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()

def score_query(q, hits):
    targets = [norm(r["anchor"]) for r in q["relevant"]]
    rel = [next((i for i, t in enumerate(targets) if t and t in norm(h["content"])), None) for h in hits[:10]]
    found = lambda k: len({r for r in rel[:k] if r is not None})
    first = next((i for i, r in enumerate(rel) if r is not None), None)
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rel) if r is not None and r not in rel[:i])
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(targets), 10)))
    return {"r5": found(5) / len(targets), "r10": found(10) / len(targets),
            "mrr": 0 if first is None else 1 / (first + 1), "ndcg": dcg / idcg if idcg else 0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    qs = {q["id"]: q for q in map(json.loads, open(a.queries))}
    print(f"{'arm':<22}{'kind':<13}{'n':>4}{'R@5':>7}{'R@10':>7}{'MRR':>7}{'nDCG':>7}{'p50ms':>8}{'p95ms':>8}")
    for path in a.arms:
        arm = path.rsplit("/", 1)[-1].removesuffix(".jsonl")
        rows = [json.loads(l) for l in open(path)]
        by = defaultdict(list)
        for r in rows:
            q = qs[r["id"]]
            s = score_query(q, r["hits"])
            s["ms"] = r["ms"]
            by[q["kind"]].append(s)
            by["ALL"].append(s)
        for kind in ["known_item", "paraphrase", "identifier", "cross_group", "ALL"]:
            v = by.get(kind)
            if not v:
                continue
            m = lambda k: sum(x[k] for x in v) / len(v)
            ms = sorted(x["ms"] for x in v)
            p95 = ms[min(len(ms) - 1, int(0.95 * len(ms)))]
            print(f"{arm:<22}{kind:<13}{len(v):>4}{m('r5'):>7.3f}{m('r10'):>7.3f}{m('mrr'):>7.3f}{m('ndcg'):>7.3f}{statistics.median(ms):>8.0f}{p95:>8}")

if __name__ == "__main__":
    main()
