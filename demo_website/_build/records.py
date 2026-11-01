"""One question -> the record the demo page renders (shared by the export and the live runner).

The page's stage renderers read exactly this shape, so a live run and a saved run look the same.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from mmjee_reasoner.verifier.features import answer_key
from mmjee_reasoner.verifier.model import select

MAX_TEXT = 14000          # longer trajectories are cut in the middle (marked in the text)
MAX_FIX = 6000
TYPE_SHORT = {"MCQ-Single": "MCQ", "MCQ-Multiple": "MCQM", "Numerical": "Num", "Matching": "Match"}


def cut(t: str | None, n: int) -> str:
    t = (t or "").strip()
    if len(t) <= n:
        return t
    return t[: n // 2] + f"\n\n[... {len(t) - n} characters not shown ...]\n\n" + t[-n // 2:]


def short_id(uid: str) -> str:
    # 2026_P1_English_Chemistry_MCQ-Multiple_q1_MCQ-Multiple_page28 -> P1-EN-Chem-MCQM-q1
    y, p, lang, subj, qt, q = uid.split("_")[:6]
    return f"{p}-{lang[:2].upper()}-{subj[:4]}-{TYPE_SHORT[qt]}-{q}"


def chain_view(ch: dict | None, gate_taken: bool | None) -> dict | None:
    """Critic rounds of one chain (``asdict(Chain)`` or a saved chains.json entry)."""
    if not ch:
        return None
    rounds = []
    for r, crit in enumerate(ch["critiques"]):
        nxt = ch["solutions"][r + 1] if r + 1 < len(ch["solutions"]) else None
        rounds.append({
            "error": bool(crit["has_error"]), "step": crit.get("step"),
            "why": cut(crit.get("explanation"), 900),
            "forced": bool(crit.get("forced_verdict")),
            "fix": None if nxt is None else {
                "answer": nxt.get("answer"), "failed": bool(nxt.get("correction_failed")),
                "text": cut(nxt.get("text"), MAX_FIX)},
        })
    return {"rounds": rounds, "stop": ch["stop_reason"], "gate_taken": gate_taken}


def question_record(m, rid: str, image: str, g: pd.DataFrame, pot: pd.DataFrame,
                    chains: dict, gate: pd.DataFrame | None, transcript: str | None,
                    tag: dict, retrieval: dict, kb: dict, selection: str,
                    cot: pd.DataFrame | None = None, sc_n: int = 0,
                    has_key: bool = True) -> tuple[dict, dict]:
    """Build the page record of one question.

    ``m``: question row (uid, twin_key, paper, language, subject, question_type,
    requires_image, answer); ``g``: its 16-answer pool with a ``score`` column, in pool order;
    ``pot``: its original solver samples; ``chains``: cid -> chain dict; ``gate``: gate_choices
    indexed by (uid, sample); ``cot``: its chain-of-thought baseline samples (exam questions).
    Returns (record, check) where check holds the per-question values behind Table I.
    """
    uid, qt = m["uid"], m["question_type"]
    pick = int(np.argmax(g["score"].to_numpy()))
    fin = select(g, g["score"].to_numpy(), selection)
    maj = select(g, None, "majority")
    pool_votes = Counter(answer_key(p, qt) for p in g["pred"])

    # the trajectory shown = the sample the verifier's pick came from
    s = int(g.loc[pick, "sample"])
    t = pot[pot["sample"] == s].iloc[0]
    gk = (uid, s)
    gate_taken = bool(gate.loc[gk, "taken"]) if gate is not None and gk in gate.index else None
    oracle = bool(g["correct"].astype(bool).any())
    outcome = "correct" if fin["correct"] else ("selector_miss" if oracle else "never_in_pool")
    methods = {"major": {"pred": maj["pred"], "ok": bool(maj["correct"]),
                         "votes": pool_votes.most_common(4)},
               "final": {"pred": fin["pred"], "ok": bool(fin["correct"])}}
    check = {"final": fin["correct"]}
    if cot is not None:
        c = cot.sort_values("sample")
        csub = c[c["sample"] < sc_n].reset_index(drop=True)
        sc = select(csub, None, "majority")
        sc_votes = Counter(answer_key(p, qt) for p in csub["pred"])
        methods = {"cot0": {"pred": c.iloc[0]["pred"], "ok": bool(c.iloc[0]["correct"])},
                   "cot_rate": [int(c["correct"].astype(bool).sum()), len(c)],
                   "sc": {"pred": sc["pred"], "ok": bool(sc["correct"]),
                          "votes": sc_votes.most_common(4), "n": len(csub)},
                   **methods}
        check.update(sc=sc["correct"], base=float(c["correct"].astype(float).mean()))
    rec = {
        "uid": uid, "id": rid, "twin": m["twin_key"], "paper": m["paper"],
        "lang": m["language"], "subj": m["subject"], "type": qt,
        "diagram": bool(m["requires_image"]), "gold": m["answer"] if has_key else None,
        "image": image,
        "transcript": cut(transcript, 3000),
        "tag": {k: tag.get(k) for k in ("subject", "topic", "method", "query")},
        "retrieval": [{"id": k, "sim": round(float(v), 4), "year": kb[k]["year"],
                       "subj": kb[k]["subject"], "type": kb[k]["question_type"],
                       "problem": cut(kb[k].get("problem"), 700)}
                      for k, v in zip(retrieval.get("kb_ids", []), retrieval.get("sims", []))],
        "shown": {"sample": s, "text": cut(t["text"], MAX_TEXT), "pred": t["pred"],
                  "ok": bool(t["correct"]), "blocks": int(t["n_code_blocks"]),
                  "blocks_ok": int(t["n_code_ok"]), "tokens": int(t["completion_tokens"]),
                  "finish": t["finish"], "fallback": bool(t["used_fallback"])},
        "chain": chain_view(chains.get(f"{uid}|s{s}"), gate_taken),
        "pool": [{"src": "c" if r["source"] == "corrected" else "o", "s": int(r["sample"]),
                  "pred": r["pred"] if isinstance(r["pred"], str) else None,
                  "ok": bool(r["correct"]), "score": round(float(r["score"]), 4)}
                 for _, r in g.iterrows()],
        "pick": pick,
        "methods": methods,
        "pot_rate": [int(pot["correct"].astype(bool).sum()), int(len(pot))],
        "outcome": outcome,
    }
    if not has_key:
        rec["key"] = False
    return rec, check
