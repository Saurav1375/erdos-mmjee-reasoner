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

    name = cfg["rag"].get("kb_name", "kb")
    suffix = subset_suffix(cfg)
    if suffix.startswith(".limit"):
        # The KB is built from training data only, so it does not depend on the questions a
        # smoke run (--limit N) solves: smoke runs read the full KB. --pilot keeps its own KB.
        suffix = ""
    return artifacts_dir(cfg, "rag", f"{name}{suffix}", "x").parent


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
    excl = cfg["rag"].get("kb_exclude_years") or []   # v3 dev runs: KB without the dev year
    train = train[~train["year"].isin(excl)]
    if "external" in set(questions["split"]):         # v3: external text-only questions
        train = pd.concat([train, questions[questions["split"] == "external"]])
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
            "source": (rep.get("source_dataset") or "jeebench") if rep["split"] == "external"
            else "mmjee",
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

    from mmjee_reasoner.pipeline.common import subset_suffix

    if subset_suffix(cfg).startswith(".limit"):
        raise ValueError("build-kb does not take --limit: smoke runs read the full KB "
                         "(it would be overwritten by a KB built from a random subset)")

    questions = load_questions(cfg)
    tags, transcripts = load_tags(cfg), load_transcripts(cfg)
    candidates = load_candidates(cfg, source_exp)
    ext_years: set[int] = set()
    for src in cfg["rag"].get("extra_kb") or []:      # v3: e.g. {exp: kb_ext_pot, dataset: ..}
        from mmjee_reasoner.data.external import load_external

        ext = load_external(cfg, src["dataset"])
        ext_years |= set(ext["year"])
        questions = pd.concat([questions, ext], ignore_index=True)
        candidates = pd.concat([candidates, load_candidates(cfg, src["exp"])], ignore_index=True)
        transcripts = {**transcripts, **{u: {"text": t} for u, t in
                                         zip(ext["uid"], ext["question_text"])}}
    entries, stats = build_entries(cfg, questions, candidates, tags, transcripts)
    # v3: de-duplicate against every held-out year (2025 and 2026), not only the active test set
    held = cfg["rag"].get("dedup_years")
    test_mask = questions["year"].isin(held) if held else questions["split"] == "test"
    test_uids = questions.loc[test_mask, "uid"]
    test_texts = [transcripts[u]["text"] for u in test_uids if u in transcripts]
    entries, dropped = dedup_against_test(cfg, entries, test_texts)

    # Leakage guards.
    test_twins = set(questions.loc[test_mask | (questions["split"] == "test"), "twin_key"])
    allowed_years = set(cfg["data"]["train_years"]) | ext_years
    assert all(e["year"] in allowed_years for e in entries)
    assert not ext_years & (set(cfg["data"]["test_years"]) | set(held or []))
    assert not {e["kb_id"] for e in entries} & test_twins

    if not entries:
        raise RuntimeError("knowledge base is empty (no verified solutions / all deduplicated)")
    vecs = embed_texts([e["query"] for e in entries], cfg["rag"]["embedding_model"])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    d = kb_dir(cfg)
    faiss.write_index(index, str(d / "index.faiss"))
    if cfg["rag"].get("retrieval", "tag") == "hybrid":  # v3: problem-text vectors as well
        np.save(d / "problem_vecs.npy",
                embed_texts([e["problem"] for e in entries], cfg["rag"]["embedding_model"]))
    write_jsonl(d / "kb.jsonl", entries)
    meta = {**stats, "entries": len(entries), "dedup_dropped": dropped,
            "test_transcripts": len(test_texts), "source_exp": source_exp,
            "embedding_model": cfg["rag"]["embedding_model"],
            "entries_by_source": pd.Series([e.get("source", "mmjee") for e in entries])
            .value_counts().to_dict(),
            "retrieval": cfg["rag"].get("retrieval", "tag")}
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


def load_problem_vecs(cfg: dict) -> np.ndarray | None:
    """v3 hybrid retrieval: embeddings of the KB problem texts (None for a v2 KB)."""
    p = kb_dir(cfg) / "problem_vecs.npy"
    return np.load(p) if p.exists() else None


def kb_embeddings(index) -> np.ndarray:
    return index.reconstruct_n(0, index.ntotal)
