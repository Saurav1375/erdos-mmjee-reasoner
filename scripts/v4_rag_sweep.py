"""W4c: LOYO retrieval coverage of kb_v4 (kb_v3 + OlympiadBench) vs kb_v3 (CPU, no test data).

Same procedure as scripts/v3_rag_sweep.py (hybrid w = 0.9, top-1 hit, LOYO proxy tau at
precision 0.70), run on both full KBs so the two numbers are comparable.

usage: .venv/bin/python scripts/v4_rag_sweep.py  (needs artifacts_v4/rag/kb_v3 and kb_v4);
writes artifacts_v4/analysis/rag_sweep_kb4.json."""
import sys, json
from collections import Counter
import numpy as np
sys.path.insert(0, "src")
from mmjee_reasoner.config import CONFIG_DIR, load_config
from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.embed import embed_texts
from mmjee_reasoner.rag.kb import load_kb, kb_embeddings, load_problem_vecs
from mmjee_reasoner.rag.retrieve import search, topic_match, _pick_tau
from mmjee_reasoner.rag.tagger import load_tags, load_english_transcripts

W = 0.9
cfg = load_config(*[CONFIG_DIR / f for f in ("v3.yaml", "v3_final.yaml", "v4.yaml")])
tags, tx = load_tags(cfg), load_english_transcripts(cfg)
qs = load_questions(cfg)
q = qs[(qs["split"] == "train") & qs["uid"].isin(tags.keys())].reset_index(drop=True)
m = cfg["rag"]["embedding_model"]
qt = embed_texts([tags[u]["query"] for u in q["uid"]], m)
qx = embed_texts([tx.get(u, "") for u in q["uid"]], m)
res = {}
for kb_name in ("kb_v3", "kb_v4"):
    cfg["rag"].update(kb_name=kb_name)
    entries, index, meta = load_kb(cfg)
    kv, kp = kb_embeddings(index), load_problem_vecs(cfg)
    years = np.array([e["year"] for e in entries])
    ext = np.array([e.get("source", "mmjee") != "mmjee" for e in entries])
    sims_all, rel, src = [], [], []
    for i, row in q.iterrows():
        allowed = (years != row["year"]) | ext
        sims = W * (kv @ qt[i]) + (1 - W) * (kp @ qx[i])
        hits = search(qt[i], kv, entries, row["subject"], row["twin_key"], 1, -1.0, allowed, sims=sims)
        if hits:
            j, s = hits[0]
            sims_all.append(s); src.append(entries[j].get("source", "mmjee"))
            rel.append(topic_match(tags[row["uid"]]["topic"], entries[j]["topic"]))
    sims_all = np.array(sims_all)
    t = _pick_tau(sims_all, np.array(rel, float), 0.7, "loyo")
    above = Counter(s for s, x in zip(src, sims_all) if x >= t["value"])
    res[kb_name] = {**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()},
                    "entries": len(entries), "hits_by_source": dict(above)}
    print(kb_name, res[kb_name], flush=True)
json.dump(res, open("artifacts_v4/analysis/rag_sweep_kb4.json", "w"), indent=1)
