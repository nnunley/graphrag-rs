"""Spike: compare extraction prompt arms on qwen3.8-27b (llama.cpp).
Read-only on the graphrag DB. Writes results to ./out/."""
import hashlib, json, os, sqlite3, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("SPIKE_OUT", "out"))
os.makedirs(OUT, exist_ok=True)
DB = os.environ.get("SPIKE_DB") or os.path.expanduser("~/Library/Application Support/graphrag/graphrag.db")
# Production chunker + production prompt, from graphrag-worker/examples/spike_tools.rs
TOOLS = os.environ.get("SPIKE_TOOLS", os.path.join(HERE, "../../target/release/examples/spike_tools"))
URL = "http://ndn.lan:8080/v1/chat/completions"
MODEL = "qwen3.8-27b"
STORES = ["operation", "conversations", "mica-docs", "graphrag-paper"]
PER_STORE = 3

TRIPLE_LINE_SYSTEM = """You are an entity extraction system. Extract entities and their relationships from the text.
Entity types MUST be one of: Person | Software | Concept | Organization | Location | Other.

Output ONE triple per line and NOTHING else, in exactly this format:
head (head_type) -[relation]-> tail (tail_type)

Example:
tokio (Software) -[chosen_over]-> async-std (Software)

Rules:
- Extract concrete, specific entities (not generic terms)
- Relations should be active verbs (uses, implements, created, manages, ...)
- If no clear entities exist, output nothing"""

PAPER_SYSTEM = """You extract a knowledge graph from a text for later retrieval.

1. Identify every significant named or specific entity. For each give:
   - name: the canonical name, as it would be referred to elsewhere (not a sentence or clause)
   - type: one of Person, Software, Concept, Organization, Location, Event, Other
   - description: a self-contained description of the entity based only on this text
2. Identify every pair of entities from step 1 that are clearly related in the text. For each give:
   - source, target: entity names exactly as in step 1
   - relation: a short snake_case verb phrase
   - description: why the source and target are related, based only on this text
   - strength: integer 1-10 for how strong and explicit the relationship is in the text

Only use information stated in the text. If there are no entities, return empty lists.
Return JSON only."""

PAPER_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "type": {"type": "string"}, "description": {"type": "string"}},
            "required": ["name", "type", "description"]}},
        "relationships": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "target": {"type": "string"}, "relation": {"type": "string"},
            "description": {"type": "string"}, "strength": {"type": "integer"}},
            "required": ["source", "target", "relation", "description", "strength"]}},
    },
    "required": ["entities", "relationships"],
}

# Production-size arms (re-spike, plan Task 20): each source chunk is cut by the
# production chunker and the pieces' extractions are merged per source chunk.
PROD_ARMS = {
    "C_prod": dict(prompt="arm_c", size=2000, overlap=200),
    # "shipped" = whatever graphrag-llm's paper_json builds now (spike_tools).
    "V11_prod": dict(prompt="shipped", size=2000, overlap=200),
    "V11_small": dict(prompt="shipped", size=1000, overlap=100),
    "V12_prod": dict(prompt="shipped", size=2000, overlap=200),
    "V12_small": dict(prompt="shipped", size=1000, overlap=100),
    "V13_prod": dict(prompt="shipped", size=2000, overlap=200),
    "V13_small": dict(prompt="shipped", size=1000, overlap=100),
}

ARMS = {
    "A_triples_nothink": dict(system=TRIPLE_LINE_SYSTEM, schema=None, think=False, max_tokens=1000),
    "B_paperjson_nothink": dict(system=PAPER_SYSTEM, schema=PAPER_SCHEMA, think=False, max_tokens=4000),
    "C_paperjson_think": dict(system=PAPER_SYSTEM, schema=PAPER_SCHEMA, think=True, max_tokens=12000),
}


def sample():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = []
    for s in STORES:
        cands = con.execute(
            "select id, store, content from chunks where store=? and length(content) between 300 and 6000", (s,)
        ).fetchall()
        cands.sort(key=lambda r: hashlib.blake2b(f"spike:{r[0]}".encode()).hexdigest())
        rows += cands[:PER_STORE]
    return rows


def call(arm, chunk):
    a = ARMS[arm]
    body = {
        "model": MODEL, "temperature": 0, "max_tokens": a["max_tokens"],
        "messages": [{"role": "system", "content": a["system"]},
                     {"role": "user", "content": f"Extract entities and relationships from this text:\n\n{chunk}"}],
        "chat_template_kwargs": {"enable_thinking": a["think"]},
    }
    if a["schema"]:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "graph", "schema": a["schema"]}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return dict(wall=round(time.time() - t, 1), content=m.get("content") or "",
                reasoning_chars=len(m.get("reasoning_content") or ""), usage=d.get("usage"),
                finish=d["choices"][0].get("finish_reason"))


def tool(args, stdin):
    return json.loads(subprocess.run([TOOLS] + args, input=stdin, capture_output=True, text=True, check=True).stdout)


def pieces(content, size, overlap):
    raw = content.encode()
    return [(raw[s["start"]:s["end"]].decode(), s["heading_path"]) for s in tool(["plan", str(size), str(overlap)], content)]


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return time.time() - t, m.get("content") or "", d.get("usage") or {}, d["choices"][0].get("finish_reason")


def call_prod(arm, chunk):
    cfg = PROD_ARMS[arm]
    ents, rels, wall, toks, finishes = {}, {}, 0.0, 0, []
    for text, path in pieces(chunk, cfg["size"], cfg["overlap"]):
        if cfg["prompt"] == "shipped":
            p = tool(["prompt"], json.dumps({"chunk": text, "heading_path": path, "known_classes": []}))
            body = {"model": MODEL, "temperature": 0, "max_tokens": p["max_tokens"],
                    "messages": [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}],
                    "chat_template_kwargs": {"enable_thinking": p["thinking"]},
                    "response_format": {"type": "json_schema", "json_schema": {"name": "graph", "schema": p["schema"]}}}
        else:
            a = ARMS["C_paperjson_think"]
            body = {"model": MODEL, "temperature": 0, "max_tokens": a["max_tokens"],
                    "messages": [{"role": "system", "content": a["system"]},
                                 {"role": "user", "content": f"Extract entities and relationships from this text:\n\n{text}"}],
                    "chat_template_kwargs": {"enable_thinking": True},
                    "response_format": {"type": "json_schema", "json_schema": {"name": "graph", "schema": a["schema"]}}}
        w, content, usage, finish = post(body)
        wall += w
        toks += usage.get("completion_tokens") or 0
        finishes.append(finish)
        try:
            d = json.loads(content)
        except Exception:
            continue
        for e in d.get("entities", []):
            ents.setdefault(e.get("name", "").strip().lower(), e)
        for r in d.get("relationships", []):
            rels.setdefault((r.get("source", "").lower(), r.get("relation", ""), r.get("target", "").lower()), r)
    merged = json.dumps({"entities": list(ents.values()), "relationships": list(rels.values())})
    return dict(wall=round(wall, 1), content=merged, usage={"completion_tokens": toks},
                finish="length" if "length" in finishes else "stop", pieces=len(finishes))


def job(arm, cid, store, content):
    path = os.path.join(OUT, f"{cid}.{arm}.json")
    if os.path.exists(path):
        return
    try:
        res = call_prod(arm, content) if arm in PROD_ARMS else call(arm, content)
    except Exception as e:
        res = dict(error=str(e))
    res.update(chunk_id=cid, store=store, arm=arm)
    json.dump(res, open(path, "w"), indent=1)
    print(arm, cid, res.get("wall"), res.get("finish"), res.get("error", ""), flush=True)


if __name__ == "__main__":
    rows = sample()
    json.dump([dict(chunk_id=r[0], store=r[1], content=r[2]) for r in rows], open(os.path.join(OUT, "chunks.json"), "w"), indent=1)
    arms = os.environ.get("SPIKE_ARMS", ",".join(ARMS)).split(",")
    with ThreadPoolExecutor(4) as ex:  # llama.cpp has 4 slots
        for arm in arms:
            for cid, store, content in rows:
                ex.submit(job, arm, cid, store, content)
