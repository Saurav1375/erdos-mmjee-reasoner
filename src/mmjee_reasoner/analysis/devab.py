"""Paired dev A/B comparisons on the 2024 dev year (v3, plan Phases 5, 6 and 6b).

Every dev arm re-runs a training experiment on the 2024 questions with the same seeds and
one change (RAG exemplars, a transcript block, or continuing truncated answers). Rows are
paired with the baseline on ``(uid, sample)``; accuracy is averaged per question and the
difference gets a paired bootstrap CI over questions (as for the RAG A/B already logged).

``changed_rows`` describes the rows whose text differs from the baseline. For the adaptive
budget these are exactly the continued (truncated) trajectories, because the first pass is
cache-identical.
"""

from __future__ import annotations

import pandas as pd

from mmjee_reasoner.scoring.metrics import paired_bootstrap

KEYS = ["uid", "sample"]


def pair(base: pd.DataFrame, arm: pd.DataFrame) -> pd.DataFrame:
    """Inner join of the two arms on (uid, sample); suffixes _b (baseline) and _a (arm)."""
    cols = KEYS + ["correct", "pred", "text", "completion_tokens", "finish"]
    meta = ["twin_key", "language", "requires_image", "question_type", "subject"]
    b = base[cols + [c for c in meta if c in base]]
    a = arm[cols]
    m = b.merge(a, on=KEYS, suffixes=("_b", "_a"), validate="one_to_one")
    if len(m) != len(arm):
        raise ValueError(f"{len(arm) - len(m)} arm rows have no baseline row")
    for s in ("_b", "_a"):
        m[f"correct{s}"] = m[f"correct{s}"].astype(float)
    return m


def compare(m: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Accuracy of both arms (%), and arm - baseline with a paired bootstrap CI."""
    per_q = m.groupby("uid")[["correct_b", "correct_a"]].mean()
    d = paired_bootstrap(per_q["correct_a"], per_q["correct_b"], n_boot=n_boot, seed=seed)
    return {"n_questions": int(len(per_q)), "n_rows": int(len(m)),
            "acc_base": float(per_q["correct_b"].mean() * 100),
            "acc_arm": float(per_q["correct_a"].mean() * 100), **d}


def by_group(m: pd.DataFrame, col: str, **kw) -> dict:
    return {str(k): compare(g, **kw) for k, g in m.groupby(col)}


def changed_rows(m: pd.DataFrame) -> dict:
    """Rows whose text changed: flips, answer coverage and extra tokens."""
    c = m[m["text_b"] != m["text_a"]]
    boxed_b = c["pred_b"].fillna("").astype(str).str.len() > 0
    boxed_a = c["pred_a"].fillna("").astype(str).str.len() > 0
    extra = c["completion_tokens_a"] - c["completion_tokens_b"]
    return {
        "n_changed": int(len(c)), "share_changed": float(len(c) / max(len(m), 1)),
        "acc_base": float(c["correct_b"].mean() * 100) if len(c) else float("nan"),
        "acc_arm": float(c["correct_a"].mean() * 100) if len(c) else float("nan"),
        "wrong_to_right": int(((c["correct_b"] == 0) & (c["correct_a"] == 1)).sum()),
        "right_to_wrong": int(((c["correct_b"] == 1) & (c["correct_a"] == 0)).sum()),
        "answered_base": int(boxed_b.sum()), "answered_arm": int(boxed_a.sum()),
        "still_cut_off": int((c["finish_a"] == "length").sum()),
        "extra_tokens_total": int(extra.sum()),
        "extra_tokens_per_changed_row": float(extra.mean()) if len(c) else 0.0,
        "tokens_per_row_base": float(m["completion_tokens_b"].mean()),
        "tokens_per_row_arm": float(m["completion_tokens_a"].mean()),
    }
