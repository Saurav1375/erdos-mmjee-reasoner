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
           allowed: np.ndarray | None = None,
           sims: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Return [(entry index, similarity)] best-first, after filters and threshold.

    ``sims`` (v3 hybrid retrieval) replaces the tag-only similarities ``kb_vecs @ query_vec``.
    """
    if len(entries) == 0:
        return []
    if sims is None:
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


def hybrid_weight(cfg: dict) -> float:
    """Weight of the concept-tag similarity (1.0 = v2 tag-only retrieval)."""
    rag = cfg["rag"]
    return 1.0 if rag.get("retrieval", "tag") != "hybrid" else float(rag.get("hybrid_weight", 0.5))


def query_sims(cfg: dict, q_tag: np.ndarray, q_text: np.ndarray | None, kb_vecs: np.ndarray,
               kb_problem: np.ndarray | None) -> np.ndarray:
    """Similarity of one query to every KB entry: w * tag-sim + (1-w) * problem-text-sim."""
    w = hybrid_weight(cfg)
    sims = kb_vecs @ q_tag
    if w < 1.0 and q_text is not None and kb_problem is not None:
        sims = w * sims + (1.0 - w) * (kb_problem @ q_text)
    return sims


def _query_text_vecs(cfg: dict, uids: list[str]) -> list[np.ndarray | None]:
    """English transcription embeddings of the queries (None where no transcription exists)."""
    if hybrid_weight(cfg) >= 1.0:
        return [None] * len(uids)
    from mmjee_reasoner.rag.tagger import load_english_transcripts

    tx = load_english_transcripts(cfg)
    have = [u for u in uids if u in tx]
    vecs = dict(zip(have, embed_texts([tx[u] for u in have], cfg["rag"]["embedding_model"])))
    return [vecs.get(u) for u in uids]


def tune_threshold_loyo(cfg: dict, entries: list[dict], kb_vecs: np.ndarray,
                        questions: pd.DataFrame, tags: dict) -> dict:
    """v3: tau from leave-one-year-out over the training years (queries of year y against
    the KB without year y; external entries are always allowed), same proxy as v2."""
    from mmjee_reasoner.rag.kb import load_problem_vecs

    rag = cfg["rag"]
    kb_problem = load_problem_vecs(cfg) if hybrid_weight(cfg) < 1.0 else None
    q = questions[(questions["split"] == "train") & questions["uid"].isin(tags.keys())]
    q_tag = embed_texts([tags[u]["query"] for u in q["uid"]], rag["embedding_model"])
    q_text = _query_text_vecs(cfg, list(q["uid"]))
    years = np.array([e["year"] for e in entries])
    ext = np.array([e.get("source", "mmjee") != "mmjee" for e in entries])
    sims_all, rel = [], []
    for qt, qx, (_, row) in zip(q_tag, q_text, q.iterrows()):
        allowed = (years != row["year"]) | ext
        sims = query_sims(cfg, qt, qx, kb_vecs, kb_problem)
        hits = search(qt, kb_vecs, entries, row["subject"] if rag["same_subject_only"] else None,
                      row["twin_key"], 1, -1.0, allowed, sims=sims)
        if hits:
            i, s = hits[0]
            sims_all.append(s)
            rel.append(topic_match(tags[row["uid"]]["topic"], entries[i]["topic"]))
    return _pick_tau(np.array(sims_all), np.array(rel, dtype=float),
                     rag["proxy_precision_target"], "loyo")


def _pick_tau(sims: np.ndarray, rel: np.ndarray, target: float, tag: str) -> dict:
    if len(sims) == 0:
        return {"value": float("inf"), "method": f"{tag}-no-hits"}
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
    return {"value": value, "method": f"{tag}-proxy" if chosen is not None else f"{tag}-best",
            "dev_queries": len(sims), "coverage": float(mask.mean()),
            "proxy_precision": float(rel[mask].mean()) if mask.any() else 0.0,
            "base_precision": float(rel.mean())}


def tune_threshold(cfg: dict, entries: list[dict], kb_vecs: np.ndarray, questions: pd.DataFrame,
                   tags: dict) -> dict:
    rag = cfg["rag"]
    if rag["threshold"] != "auto":
        return {"value": float(rag["threshold"]), "method": "fixed"}
    if rag.get("threshold_tuning") == "loyo":
        return tune_threshold_loyo(cfg, entries, kb_vecs, questions, tags)
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

    from mmjee_reasoner.rag.kb import load_problem_vecs

    entries, index, meta = load_kb(cfg)
    kb_vecs = kb_embeddings(index)
    kb_problem = load_problem_vecs(cfg) if hybrid_weight(cfg) < 1.0 else None
    tags = load_tags(cfg)
    missing = [u for u in questions["uid"] if u not in tags]
    if missing:
        raise RuntimeError(f"{len(missing)} questions have no concept tag - run `mmjee tag`")
    rag = cfg["rag"]
    q_vecs = embed_texts([tags[u]["query"] for u in questions["uid"]], rag["embedding_model"])
    q_text = _query_text_vecs(cfg, list(questions["uid"]))
    out, log_rows = {}, []
    for qv, qx, (_, q) in zip(q_vecs, q_text, questions.iterrows()):
        sims = query_sims(cfg, qv, qx, kb_vecs, kb_problem) if kb_problem is not None else None
        hits = search(qv, kb_vecs, entries, q["subject"] if rag["same_subject_only"] else None,
                      q["twin_key"], rag["top_k"], meta["threshold"]["value"], sims=sims)
        examples = [{"problem": entries[i]["problem"], "solution": entries[i]["solution"]}
                    for i, _ in hits]
        out[q["uid"]] = {"text": format_exemplars(examples),
                         "kb_ids": [entries[i]["kb_id"] for i, _ in hits],
                         "sims": [s for _, s in hits]}
        log_rows.append({"uid": q["uid"], "kb_ids": out[q["uid"]]["kb_ids"],
                         "sims": out[q["uid"]]["sims"], "query": tags[q["uid"]]["query"]})
    from mmjee_reasoner.pipeline.common import test_tag

    split = questions["split"].iloc[0] if len(questions) else "none"
    kb_name = cfg["rag"].get("kb_name", "kb")
    name = f"retrieval_{split}.jsonl" if kb_name == "kb" and not test_tag(cfg) else \
        f"retrieval_{split}{test_tag(cfg)}_{kb_name}.jsonl"   # v3: never overwrite v2 logs
    path = artifacts_dir(cfg, "rag", name)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in log_rows)
    n_hit = sum(bool(v["kb_ids"]) for v in out.values())
    log.info("retrieval: %d/%d questions got exemplars", n_hit, len(out))
    return out
