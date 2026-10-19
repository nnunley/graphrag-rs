You are judging knowledge-graph extractions for a personal GraphRAG memory store (engineering notes, conversation notes, docs, a research paper). The graph is used for retrieval and for building Leiden communities that get summarized, so good extractions name real, reusable entities and connect them with grounded, meaningful relationships.

Read the file packets.txt in this directory. For each CHUNK there is the source text and three blind extractions labeled X, Y, Z. They come from different prompting setups; do not try to guess which.

Score each extraction on four criteria, 0-5 each (20 max):
1. Grounding/precision: everything stated is supported by the source text; no invented entities, facts, or relationships.
2. Recall: the important entities and relationships in the text are captured (judge by what matters for later retrieval, not exhaustiveness).
3. Entity quality: entities are canonical, specific names (not sentence fragments, clauses, or generic words); types are sensible; duplicates are not split.
4. Usefulness: would this extraction make the graph better for retrieval and community summaries? Descriptions, if present, count only if accurate and informative.

Output ONLY a JSON object, no prose, of the form:
{"scores": {"<chunk_id>": {"X": [g, r, e, u], "Y": [g, r, e, u], "Z": [g, r, e, u]}, ...},
 "notes": "<=150 words on systematic differences you saw between extractions, referring to them by label patterns only>"}
