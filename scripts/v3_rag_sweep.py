"""LOYO sweep of retrieval variants on the training years (CPU, no test data).

usage: .venv/bin/python scripts/v3_rag_sweep.py  (needs artifacts_v3/rag/kb_dev2024_v3 built with
the JEEBench entries); writes artifacts_v3/analysis/rag_sweep_loyo.json."""
import sys, json
import numpy as np
sys.path.insert(0, "src")
from mmjee_reasoner.config import CONFIG_DIR, load_config
from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.embed import embed_texts
from mmjee_reasoner.rag.kb import load_kb, kb_embeddings, load_problem_vecs
from mmjee_reasoner.rag.retrieve import search, topic_match, _pick_tau
from mmjee_reasoner.rag.tagger import load_tags, load_english_transcripts

cfg = load_config(CONFIG_DIR / "v3.yaml")
cfg["rag"].update(kb_name="kb_dev2024_v3")
entries, index, meta = load_kb(cfg)
kv, kp = kb_embeddings(index), load_problem_vecs(cfg)
tags, tx = load_tags(cfg), load_english_transcripts(cfg)
qs = load_questions(cfg)
q = qs[(qs["split"] == "train") & qs["uid"].isin(tags.keys())].reset_index(drop=True)
m = cfg["rag"]["embedding_model"]
qt = embed_texts([tags[u]["query"] for u in q["uid"]], m)
qx = embed_texts([tx.get(u, "") for u in q["uid"]], m)
years = np.array([e["year"] for e in entries]); ext = np.array([e.get("source","mmjee")!="mmjee" for e in entries])
res = {}
for use_ext in (False, True):
    for w in (1.0, 0.9, 0.8, 0.7, 0.5):
        sims_all, rel, yrs = [], [], []
        for i, row in q.iterrows():
            allowed = ((years != row["year"]) & ~ext) | (ext & use_ext)
            sims = w * (kv @ qt[i]) + (1 - w) * (kp @ qx[i])
            hits = search(qt[i], kv, entries, row["subject"], row["twin_key"], 1, -1.0, allowed, sims=sims)
            if hits:
                j, s = hits[0]; sims_all.append(s); rel.append(topic_match(tags[row["uid"]]["topic"], entries[j]["topic"]))
        t = _pick_tau(np.array(sims_all), np.array(rel, float), 0.7, "loyo")
        res[f"ext={use_ext} w={w}"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()}
        print(f"ext={use_ext!s:5} w={w}", res[f"ext={use_ext} w={w}"], flush=True)
json.dump(res, open("artifacts_v3/analysis/rag_sweep_loyo.json", "w"), indent=1)
