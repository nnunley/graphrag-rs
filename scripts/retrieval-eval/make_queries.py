"""Generate the retrieval eval query set from a COPY of a graphrag DB.

Kinds (see docs/2026-10-19-ingestion-workflow-plan.md, Task 19A):
  known_item  - a question one sampled passage answers; anchor = an exact quote from it
  paraphrase  - same, but the question shares no content word with the anchor
  identifier  - a flag / path / identifier copied from the text
  cross_group - a note and a non-note passage that share a rare term; the query uses
                the note's wording and both passages are relevant

Relevance is anchored by quoted text, not chunk ids, so the set survives
re-chunking and migration. Reads the DB read-only. Usage:
  python3 -I make_queries.py --db COPY.db --out ../../docs/retrieval-eval-queries.jsonl
"""
import argparse, hashlib, json, re, sqlite3, urllib.request
from concurrent.futures import ThreadPoolExecutor

STOP = set("the a an and or of to in on for with is are was were be by as at it this that from how what why when which who does do did can should would about into over".split())
SYSTEM = """You write evaluation questions for a personal knowledge base search engine.
Given a passage, return JSON {"question": ..., "anchor": ...}:
- anchor: an EXACT, contiguous quote from the passage (10-120 characters) that answers the question
- question: a natural question someone would type to find this passage later
Return JSON only."""
PARA_RULE = "\n- the question must NOT reuse any content word (noun, verb, adjective) that appears in the anchor; paraphrase it"


def chunks(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cols = {r[1] for r in con.execute("pragma table_info(chunks)")}
    if "content" in cols:  # v1 schema
        rows = con.execute("select id, store, coalesce(source,''), content from chunks").fetchall()
    else:
        rows = con.execute("select c.id, c.store, coalesce(c.source,''), c.content from chunk_view c").fetchall()
    return [r for r in rows if r[1] not in ("graphrag-paper-dense",) and 200 <= len(r[3]) <= 4000]


def ordered(rows, salt):
    return sorted(rows, key=lambda r: hashlib.blake2b(f"{salt}:{r[0]}".encode()).hexdigest())


def words(s):
    return {w for w in re.findall(r"[a-z][a-z0-9_-]{3,}", s.lower()) if w not in STOP}


def ask(url, model, system, user):
    body = {"model": model, "temperature": 0, "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=300))
        return json.loads(d["choices"][0]["message"]["content"])
    except Exception as e:
        return {"error": str(e)}


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://ndn.lan:8080/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--per-kind", type=int, default=40)
    a = ap.parse_args()
    rows = chunks(a.db)
    out = []

    def gen(kind, row, rule=""):
        cid, store, source, text = row
        r = ask(a.url, a.model, SYSTEM + rule, f"Passage:\n{text}")
        q, anchor = r.get("question", ""), r.get("anchor", "")
        if not q or not anchor or norm(anchor) not in norm(text):
            return None
        if kind == "paraphrase" and words(q) & words(anchor):
            return None
        return {"kind": kind, "query": q, "relevant": [{"source": source, "anchor": norm(anchor)}],
                "origin": "llm", "store": store}

    for kind, rule, salt in [("known_item", "", "ki"), ("paraphrase", PARA_RULE, "pp")]:
        cands = ordered(rows, salt)[: a.per_kind * 3]
        with ThreadPoolExecutor(4) as ex:
            got = [g for g in ex.map(lambda r: gen(kind, r, rule), cands) if g]
        out += got[: a.per_kind]

    ident = re.compile(r"(--[a-z][a-z0-9-]{2,}|\b[a-z_]+_[a-z_]+\b|\b[\w.-]+/[\w./-]+\.\w+\b|\b[A-Z][a-zA-Z]+::[A-Za-z_]+\b)")
    seen = set()
    for cid, store, source, text in ordered(rows, "id"):
        m = [t for t in ident.findall(text) if len(t) >= 6 and t not in seen]
        if not m:
            continue
        tok = m[0]
        seen.add(tok)
        line = next(l for l in text.splitlines() if tok in l)
        out.append({"kind": "identifier", "query": tok, "relevant": [{"source": source, "anchor": norm(tok)}],
                    "origin": "text", "store": store, "context": norm(line)[:200]})
        if sum(1 for o in out if o["kind"] == "identifier") >= a.per_kind:
            break

    # cross_group: a rare capitalized term shared by a note and a non-note passage.
    notes = [r for r in rows if r[1] == "conversations"]
    others = [r for r in rows if r[1] != "conversations"]
    term = re.compile(r"\b[A-Z][A-Za-z0-9]{3,}\b")
    df = {}
    for r in rows:
        for t in set(term.findall(r[3])):
            df[t] = df.get(t, 0) + 1
    count = 0
    for n in ordered(notes, "cg"):
        rare = [t for t in set(term.findall(n[3])) if 2 <= df[t] <= 6]
        for t in sorted(rare):
            o = next((r for r in ordered(others, t) if t in r[3]), None)
            if not o:
                continue
            sent_n = next(s for s in re.split(r"(?<=[.!?\n])\s+", n[3]) if t in s)
            sent_o = next(s for s in re.split(r"(?<=[.!?\n])\s+", o[3]) if t in s)
            r = ask(a.url, a.model, "Rewrite the sentence as a search question in the same wording style. JSON {\"question\": ...}.", sent_n)
            if not r.get("question"):
                continue
            out.append({"kind": "cross_group", "query": r["question"],
                        "relevant": [{"source": n[2], "anchor": norm(sent_n)[:160]}, {"source": o[2], "anchor": norm(sent_o)[:160]}],
                        "origin": "llm", "store": n[1], "term": t})
            count += 1
            break
        if count >= a.per_kind // 2:
            break

    with open(a.out, "w") as f:
        for i, o in enumerate(out):
            o["id"] = f"q{i:03d}"
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print({k: sum(1 for o in out if o["kind"] == k) for k in ["known_item", "paraphrase", "identifier", "cross_group"]})


if __name__ == "__main__":
    main()
