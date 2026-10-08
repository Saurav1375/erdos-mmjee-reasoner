"""Knowledge base of solved 2019-2024 problems for exemplar retrieval (Module D).

One entry per exam question (English/Hindi twins merged):

* ``problem``  -- VLM transcription of the question image (English version preferred)
* ``query``    -- concept tag (TOPIC + METHOD) used as the retrieval key
* ``solution`` -- a worked solution produced by the Solver on the *training*
  questions and verified correct by the upstream scorer (no human solutions).

Leakage control: only train-split (2019-2024) questions enter the KB, and
entries whose transcription is a near-duplicate (cosine >= ``dedup_threshold``)
of any 2025 transcription are removed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.embed import embed_texts
from mmjee_reasoner.pipeline.common import load_candidates, read_jsonl, write_jsonl
from mmjee_reasoner.rag.tagger import load_tags, load_transcripts

log = logging.getLogger(__name__)


def kb_dir(cfg: dict) -> Path:
    from mmjee_reasoner.pipeline.common import subset_suffix

    return artifacts_dir(cfg, "rag", f"kb{subset_suffix(cfg)}", "x").parent


def shorten(text: str, limit: int) -> str:
    """Keep the head and the tail (which holds the final answer) of long solutions."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    return text[:head].rstrip() + "\n[...]\n" + text[-(limit - head):].lstrip()


def pick_solution(sols: pd.DataFrame) -> pd.Series | None:
    """Prefer English, code that ran, then the shortest verified-correct solution."""
    if sols.empty:
        return None
    s = sols.assign(
        _lang=(sols["language"] != "English").astype(int),
        _code=(sols["n_code_ok"] == 0).astype(int),
        _len=sols["text"].str.len(),
    ).sort_values(["_lang", "_code", "_len", "uid", "sample"])
    return s.iloc[0]


def build_entries(cfg: dict, questions: pd.DataFrame, candidates: pd.DataFrame,
                  tags: dict, transcripts: dict) -> tuple[list[dict], dict]:
    train = questions[questions["split"] == "train"]
    good = candidates[candidates["correct"].astype(bool)
                      & ~candidates["used_fallback"].astype(bool)
                      & candidates["text"].str.contains(r"\\boxed", regex=True)]
    good = good[good["uid"].isin(set(train["uid"]))]
    entries, no_solution = [], 0
    for twin, grp in train.groupby("twin_key", sort=True):
        en = grp[grp["language"] == "English"]
        rep = (en if len(en) else grp).sort_values("uid").iloc[0]
        sol = pick_solution(good[good["twin_key"] == twin])
        if sol is None or rep["uid"] not in tags or rep["uid"] not in transcripts:
            no_solution += 1
            continue
        tag = tags[rep["uid"]]
        entries.append({
            "kb_id": twin, "uid": rep["uid"], "year": int(rep["year"]),
            "subject": rep["subject"], "question_type": rep["question_type"],
            "problem": transcripts[rep["uid"]]["text"],
            "solution": shorten(sol["text"], cfg["rag"]["max_solution_chars"]),
            "solution_uid": sol["uid"], "solution_sample": int(sol["sample"]),
            "topic": tag["topic"], "method": tag["method"], "query": tag["query"],
        })
    return entries, {"twins": int(train["twin_key"].nunique()), "no_solution": no_solution}


def dedup_against_test(cfg: dict, entries: list[dict], test_texts: list[str]) -> tuple[list, list]:
    if not entries or not test_texts:
        return entries, []
    name = cfg["rag"]["embedding_model"]
    e_kb = embed_texts([e["problem"] for e in entries], name)
    e_test = embed_texts(test_texts, name)
    max_sim = (e_kb @ e_test.T).max(axis=1)
    keep = max_sim < cfg["rag"]["dedup_threshold"]
    dropped = [{"kb_id": e["kb_id"], "max_sim": float(s)}
               for e, s, k in zip(entries, max_sim, keep) if not k]
    return [e for e, k in zip(entries, keep) if k], dropped


def build_kb(cfg: dict, source_exp: str = "pot_train") -> dict:
    import faiss

    questions = load_questions(cfg)
    tags, transcripts = load_tags(cfg), load_transcripts(cfg)
    entries, stats = build_entries(cfg, questions, load_candidates(cfg, source_exp), tags,
                                   transcripts)
    test_uids = questions.loc[questions["split"] == "test", "uid"]
    test_texts = [transcripts[u]["text"] for u in test_uids if u in transcripts]
    entries, dropped = dedup_against_test(cfg, entries, test_texts)

    # Leakage guards.
    test_twins = set(questions.loc[questions["split"] == "test", "twin_key"])
    assert all(e["year"] in cfg["data"]["train_years"] for e in entries)
    assert not {e["kb_id"] for e in entries} & test_twins

    if not entries:
        raise RuntimeError("knowledge base is empty (no verified solutions / all deduplicated)")
    vecs = embed_texts([e["query"] for e in entries], cfg["rag"]["embedding_model"])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    d = kb_dir(cfg)
    faiss.write_index(index, str(d / "index.faiss"))
    write_jsonl(d / "kb.jsonl", entries)
    meta = {**stats, "entries": len(entries), "dedup_dropped": dropped,
            "test_transcripts": len(test_texts), "source_exp": source_exp,
            "embedding_model": cfg["rag"]["embedding_model"]}
    from mmjee_reasoner.rag.retrieve import tune_threshold

    meta["threshold"] = tune_threshold(cfg, entries, vecs, questions, tags)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("KB: %d entries (%d twins without verified solution, %d dropped by dedup), "
             "threshold %.3f", len(entries), stats["no_solution"], len(dropped),
             meta["threshold"]["value"])
    return meta


def load_kb(cfg: dict):
    import faiss

    d = kb_dir(cfg)
    entries = read_jsonl(d / "kb.jsonl")
    index = faiss.read_index(str(d / "index.faiss"))
    meta = json.loads((d / "meta.json").read_text())
    return entries, index, meta


def kb_embeddings(index) -> np.ndarray:
    return index.reconstruct_n(0, index.ntotal)
