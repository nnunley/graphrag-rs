"""unblind judge scores and summarize per arm and per store."""
import json, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.environ.get("SPIKE_PACK", "")
key = json.load(open(os.path.join(HERE, f"{PACK}blind_key.json")))
chunks = {str(c["chunk_id"]): c["store"] for c in json.load(open(os.path.join(HERE, os.environ.get("SPIKE_OUT", "out"), "chunks.json")))}
CRIT = ["ground", "recall", "entity", "useful"]


def load(path):
    txt = open(path).read()
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0))


for judge in sys.argv[1:]:
    d = load(os.path.join(HERE, judge))
    tot, crit, store = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for cid, labs in d["scores"].items():
        for lab, s in labs.items():
            arm = key[str(cid)][lab]
            tot[arm].append(sum(s))
            for c, v in zip(CRIT, s):
                crit[arm][c].append(v)
            store[arm][chunks[str(cid)]].append(sum(s))
    print(f"== {judge} ==")
    for arm in sorted(tot):
        avg = lambda xs: round(sum(xs) / len(xs), 1)
        print(f"{arm:22} total {avg(tot[arm]):5}/20  " + "  ".join(f"{c} {avg(crit[arm][c])}" for c in CRIT)
              + "  | " + "  ".join(f"{s} {avg(v)}" for s, v in sorted(store[arm].items())))
    print("notes:", d.get("notes", "")[:900])
