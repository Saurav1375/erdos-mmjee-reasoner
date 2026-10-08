"""Exemplar retrieval with a similarity threshold (Module D).

Query = concept tag of the question image. Candidates are KB entries of the
same subject (``same_subject_only``) and never the question itself (its twin
key). The top-k entries with cosine similarity >= tau are used; on a weak
match nothing is retrieved.

Threshold tuning (``rag.threshold: auto``) uses only training data: queries
are the 2024 questions, the KB is restricted to 2019-2023 entries. As a proxy
for "uses the same method" we call a retrieval relevant when the topic phrases
of query and entry overlap (token Jaccard >= 0.5). tau is the smallest
similarity at which the top-1 precision of this proxy reaches
``proxy_precision_target``.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.embed import embed_texts
from mmjee_reasoner.prompts.templates import format_exemplars

log = logging.getLogger(__name__)

_STOP = {"of", "and", "the", "in", "a", "an", "to", "for", "on", "with", "by", "its", "using"}


def topic_tokens(topic: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", topic.lower()) if t not in _STOP}


def topic_match(a: str, b: str, min_jaccard: float = 0.5) -> bool:
    ta, tb = topic_tokens(a), topic_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= min_jaccard


def search(query_vec: np.ndarray, kb_vecs: np.ndarray, entries: list[dict], subject: str | None,
           exclude_twin: str | None, top_k: int, threshold: float,
           allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Return [(entry index, similarity)] best-first, after filters and threshold."""
    if len(entries) == 0:
        return []
    sims = kb_vecs @ query_vec
    order = np.argsort(-sims)
    out = []
    for i in order:
        if allowed is not None and not allowed[i]:
            continue
        e = entries[i]
        if subject and e["subject"] != subject:
            continue
        if exclude_twin and e["kb_id"] == exclude_twin:
            continue
        if sims[i] < threshold:
            break
        out.append((int(i), float(sims[i])))
        if len(out) == top_k:
            break
    return out


def tune_threshold(cfg: dict, entries: list[dict], kb_vecs: np.ndarray, questions: pd.DataFrame,
                   tags: dict) -> dict:
    rag = cfg["rag"]
    if rag["threshold"] != "auto":
        return {"value": float(rag["threshold"]), "method": "fixed"}
    dev_year = cfg["verifier"]["dev_year"]
    allowed = np.array([e["year"] < dev_year for e in entries])
    dev = questions[(questions["year"] == dev_year) & questions["uid"].isin(tags.keys())]
    if allowed.sum() == 0 or dev.empty:
        return {"value": 0.0, "method": "fallback-empty"}
    q_vecs = embed_texts([tags[u]["query"] for u in dev["uid"]], rag["embedding_model"])
    sims, rel = [], []
    for qv, (_, q) in zip(q_vecs, dev.iterrows()):
        hits = search(qv, kb_vecs, entries, q["subject"] if rag["same_subject_only"] else None,
                      q["twin_key"], 1, -1.0, allowed)
        if hits:
            i, s = hits[0]
            sims.append(s)
            rel.append(topic_match(tags[q["uid"]]["topic"], entries[i]["topic"]))
    sims, rel = np.array(sims), np.array(rel, dtype=float)
    target = rag["proxy_precision_target"]
    if len(sims) == 0:
        return {"value": float("inf"), "method": "fallback-no-dev-hits"}
    chosen, best = None, (-1.0, float("inf"))
    for tau in np.unique(sims):
        mask = sims >= tau
        if mask.sum() < max(5, 0.05 * len(sims)):
            continue
        prec = rel[mask].mean()
        if prec > best[0]:
            best = (prec, float(tau))
        if chosen is None and prec >= target:
            chosen = float(tau)
    value = chosen if chosen is not None else best[1]
    mask = sims >= value
    return {"value": value, "method": "auto-proxy" if chosen is not None else "auto-best",
            "dev_queries": len(sims), "coverage": float(mask.mean()),
            "proxy_precision": float(rel[mask].mean()) if mask.any() else 0.0,
            "base_precision": float(rel.mean()) if len(rel) else 0.0}


def exemplars_for_questions(cfg: dict, questions: pd.DataFrame, engine=None) -> dict[str, dict]:
    """{uid: {"text": exemplar block or None, "kb_ids": [...], "sims": [...]}}."""
    from mmjee_reasoner.rag.kb import kb_embeddings, load_kb
    from mmjee_reasoner.rag.tagger import load_tags

    entries, index, meta = load_kb(cfg)
    kb_vecs = kb_embeddings(index)
    tags = load_tags(cfg)
    missing = [u for u in questions["uid"] if u not in tags]
    if missing:
        raise RuntimeError(f"{len(missing)} questions have no concept tag - run `mmjee tag`")
    rag = cfg["rag"]
    q_vecs = embed_texts([tags[u]["query"] for u in questions["uid"]], rag["embedding_model"])
    out, log_rows = {}, []
    for qv, (_, q) in zip(q_vecs, questions.iterrows()):
        hits = search(qv, kb_vecs, entries, q["subject"] if rag["same_subject_only"] else None,
                      q["twin_key"], rag["top_k"], meta["threshold"]["value"])
        examples = [{"problem": entries[i]["problem"], "solution": entries[i]["solution"]}
                    for i, _ in hits]
        out[q["uid"]] = {"text": format_exemplars(examples),
                         "kb_ids": [entries[i]["kb_id"] for i, _ in hits],
                         "sims": [s for _, s in hits]}
        log_rows.append({"uid": q["uid"], "kb_ids": out[q["uid"]]["kb_ids"],
                         "sims": out[q["uid"]]["sims"], "query": tags[q["uid"]]["query"]})
    split = questions["split"].iloc[0] if len(questions) else "none"
    path = artifacts_dir(cfg, "rag", f"retrieval_{split}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in log_rows)
    n_hit = sum(bool(v["kb_ids"]) for v in out.values())
    log.info("retrieval: %d/%d questions got exemplars", n_hit, len(out))
    return out
