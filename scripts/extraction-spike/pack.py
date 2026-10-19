"""build blind judge packets + mechanical metrics from out/."""
import hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ARMS = ["A_triples_nothink", "B_paperjson_nothink", "C_paperjson_think"]
LABELS = ["X", "Y", "Z"]


def parse(arm, content):
    """-> (entities[name,type,desc], relations[src,rel,tgt,desc,strength], ok)"""
    if arm.startswith("A"):
        ents, rels, bad = {}, [], 0
        for line in content.splitlines():
            m = re.match(r"\s*(.+?)\s*\(([^)]*)\)\s*-\[(.+?)\]->\s*(.+?)\s*\(([^)]*)\)\s*$", line)
            if not m:
                bad += bool(line.strip())
                continue
            h, ht, r, t, tt = m.groups()
            ents.setdefault(h, (h, ht, "")); ents.setdefault(t, (t, tt, ""))
            rels.append((h, r, t, "", None))
        return list(ents.values()), rels, bad == 0
    try:
        d = json.loads(content)
    except Exception:
        return [], [], False
    ents = [(e.get("name", ""), e.get("type", ""), e.get("description", "")) for e in d.get("entities", [])]
    rels = [(r.get("source", ""), r.get("relation", ""), r.get("target", ""), r.get("description", ""), r.get("strength"))
            for r in d.get("relationships", [])]
    return ents, rels, True


def fragmentish(name):
    w = name.split()
    return len(w) > 6 or name.rstrip().endswith((".", "!", "?")) or len(name) > 60


def render(ents, rels):
    lines = ["ENTITIES:"]
    for n, t, d in ents:
        lines.append(f"- {n} ({t})" + (f": {d}" if d else ""))
    lines.append("RELATIONSHIPS:")
    for s, r, t, d, st in rels:
        lines.append(f"- {s} -[{r}]-> {t}" + (f" [strength {st}]" if st is not None else "") + (f": {d}" if d else ""))
    return "\n".join(lines)


chunks = json.load(open(os.path.join(OUT, "chunks.json")))
metrics, key, packets = [], {}, []
for c in chunks:
    cid = c["chunk_id"]
    order = sorted(ARMS, key=lambda a: hashlib.blake2b(f"blind:{cid}:{a}".encode()).hexdigest())
    key[cid] = dict(zip(LABELS, order))
    parts = [f"=== CHUNK {cid} (store: {c['store']}) ===\n--- SOURCE TEXT ---\n{c['content']}\n"]
    for lab, arm in zip(LABELS, order):
        r = json.load(open(os.path.join(OUT, f"{cid}.{arm}.json")))
        ents, rels, ok = parse(arm, r.get("content", "")) if "error" not in r else ([], [], False)
        metrics.append(dict(chunk=cid, store=c["store"], arm=arm, ok=ok, wall=r.get("wall"), finish=r.get("finish"),
                            n_ent=len(ents), n_rel=len(rels), frag=sum(fragmentish(e[0]) for e in ents),
                            out_tokens=(r.get("usage") or {}).get("completion_tokens"), error=r.get("error")))
        parts.append(f"--- EXTRACTION {lab} ---\n{render(ents, rels) if ok else '(UNPARSEABLE OUTPUT)'}\n")
    packets.append("\n".join(parts))

open(os.path.join(HERE, "packets.txt"), "w").write("\n\n".join(packets))
json.dump(key, open(os.path.join(HERE, "blind_key.json"), "w"), indent=1)
json.dump(metrics, open(os.path.join(HERE, "metrics.json"), "w"), indent=1)

# summary per arm
for arm in ARMS:
    ms = [m for m in metrics if m["arm"] == arm]
    def avg(k):
        v = [m[k] for m in ms if m[k] is not None]; return round(sum(v) / len(v), 1) if v else None
    print(f"{arm:22} parsed {sum(m['ok'] for m in ms)}/{len(ms)}  wall {avg('wall')}s  out_tok {avg('out_tokens')}  "
          f"ents {avg('n_ent')}  rels {avg('n_rel')}  fragment-ish ents {sum(m['frag'] for m in ms)}  "
          f"truncated {sum(m['finish']=='length' for m in ms)}")
