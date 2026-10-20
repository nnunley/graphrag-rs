"""Mechanical gate checks for the re-spike: distinct classes per 100
entities (V11 arms), tools/models typed Person, example-handle entities."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("SPIKE_OUT", "out2"))
ARMS = os.environ.get("SPIKE_ARMS", "C_prod,V11_prod,V11_small").split(",")
TOOLS = {"jj", "git", "opus", "claude", "codex", "qwen", "ollama", "llama.cpp", "nushell", "cargo"}
HANDLES = {"alice", "bob", "foo", "bar", "carol", "eve"}

for arm in ARMS:
    ents, kinds, person_tools, handles = 0, set(), [], []
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(f".{arm}.json"):
            continue
        d = json.load(open(os.path.join(OUT, f)))
        try:
            g = json.loads(d.get("content", ""))
        except Exception:
            continue
        for e in g.get("entities", []):
            ents += 1
            name = e.get("name", "").strip().lower()
            if e.get("kind"):
                kinds.add(e["kind"].strip().lower())
            if name in TOOLS and e.get("type", "").lower() == "person":
                person_tools.append(name)
            if name in HANDLES:
                handles.append(name)
    per100 = round(100 * len(kinds) / ents, 1) if ents and kinds else None
    print(f"{arm:12} entities {ents:4}  classes/100 {per100}  person-typed tools {person_tools}  example handles {handles}")
