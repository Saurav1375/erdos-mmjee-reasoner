"""Verifier v4b (Module C, Phase 4b, V4B_PREREG.md track A): an answer-level verifier.

Phase-4 selectors scored candidates one by one; selection then failed where the right answer is a
minority of the pool. Here each row is one *distinct answer* of a pool (its supporters' features
aggregated), labelled "this answer is correct", and the pick is the most probable answer. Five
fixed configurations are compared by leave-one-year-out (LOYO) over 2019-2024 and the one with the
best *worst-year* pick accuracy on the v3 16-pools is chosen. Nothing here touches 2025/2026.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mmjee_reasoner.verifier.features import QTYPES, answer_key
from mmjee_reasoner.verifier.model import selection_accuracy
from mmjee_reasoner.verifier.v4 import eval_pools, hand_features, loyo, pool_groups

log = logging.getLogger(__name__)

SUPPORT_MEANS = ["code_ok_frac", "truncated", "log_tokens", "n_steps", "hedges_per_kchar",
                 "valid_format"]
CHAIN_MEANS = ["corr_critic_flagged_first", "corr_answer_changed"]
CONFIGS = {"logreg": ("logreg", None, None),
           "xgb_d2_t100": ("xgboost", 2, 100), "xgb_d2_t300": ("xgboost", 2, 300),
           "xgb_d3_t100": ("xgboost", 3, 100), "xgb_d3_t300": ("xgboost", 3, 300)}
GATE = {"v3_16pool": 63.0, "v4_pool": 61.1}     # best v3-ranker LOYO + 1.0 (V4B_PREREG.md)


def answer_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pool, distinct answer): meta columns + ``F_*`` features + ``label``."""
    hand = hand_features(df)
    d = pd.DataFrame({
        "group": pool_groups(df), "pool": df["pool"].to_numpy(), "uid": df["uid"].to_numpy(),
        "year": df["year"].to_numpy(), "correct": df["correct"].astype(float).to_numpy(),
        "key": [answer_key(p, t) for p, t in zip(df["pred"].fillna("").astype(str),
                                                 df["question_type"])],
        "corr": df["is_corrected"].to_numpy() > 0, "chain": df["has_chain"].to_numpy() > 0,
        "pos": np.arange(len(df)),
        **{c: hand[c].to_numpy() for c in SUPPORT_MEANS},
        **{c: df[c].astype(float).to_numpy() for c in CHAIN_MEANS},
        "qtype": df["question_type"].to_numpy(),
        "requires_image": hand["requires_image"].to_numpy(),
        "lang_hindi": hand["lang_hindi"].to_numpy(),
    })
    pool_n = d.groupby("group")["pos"].transform("size")
    d["n_orig_pool"] = (~d["corr"]).groupby(d["group"]).transform("sum")
    d["n_corr_pool"] = d["corr"].groupby(d["group"]).transform("sum")
    d["pool_n"] = pool_n
    for c in CHAIN_MEANS:                       # chain features count only rows with a chain
        d[c] = d[c].where(d["chain"])
    g = d.groupby(["group", "key"], sort=False)
    a = g.agg(pool=("pool", "first"), uid=("uid", "first"), year=("year", "first"),
              label=("correct", "max"), first=("pos", "min"), n=("pos", "size"),
              n_orig=("corr", lambda s: int((~s).sum())), n_corr=("corr", "sum"),
              pool_n=("pool_n", "first"), n_orig_pool=("n_orig_pool", "first"),
              n_corr_pool=("n_corr_pool", "first"), qtype=("qtype", "first"),
              requires_image=("requires_image", "first"), lang_hindi=("lang_hindi", "first"),
              **{c: (c, "mean") for c in SUPPORT_MEANS + CHAIN_MEANS}).reset_index()
    a = a.sort_values(["group", "first"]).reset_index(drop=True)
    by = a.groupby("group")
    a["F_vote_share"] = a["n"] / a["pool_n"]
    order = a.assign(neg=-a["n"]).sort_values(["group", "neg", "first"])
    a.loc[order.index, "rank"] = order.groupby("group").cumcount().to_numpy()
    n_ans = by["n"].transform("size")
    a["F_vote_rank"] = np.where(n_ans > 1, a["rank"] / (n_ans - 1).clip(lower=1), 0.0)
    a["F_gap_top"] = by["F_vote_share"].transform("max") - a["F_vote_share"]
    a["F_distinct_frac"] = n_ans / a["pool_n"]
    a["F_orig_share"] = a["n_orig"] / a["n_orig_pool"].clip(lower=1)
    a["F_corr_share"] = np.where(a["n_corr_pool"] > 0,
                                 a["n_corr"] / a["n_corr_pool"].clip(lower=1), -1.0)
    for c in SUPPORT_MEANS:
        a[f"F_{c}"] = a[c]
    for c in CHAIN_MEANS:
        a[f"F_{c}"] = a[c].fillna(-1.0)
    for t in QTYPES:
        a[f"F_type_{t}"] = (a["qtype"] == t).astype(float)
    a["F_requires_image"] = a["requires_image"]
    a["F_lang_hindi"] = a["lang_hindi"]
    return a


def feature_cols(a: pd.DataFrame) -> list[str]:
    return [c for c in a.columns if c.startswith("F_")]


def make_clf(name: str, seed: int):
    kind, depth, trees = CONFIGS[name]
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=trees, max_depth=depth, learning_rate=0.05,
                         eval_metric="logloss", random_state=seed, n_jobs=4)


def loyo_probs(a: pd.DataFrame, name: str, seed: int) -> np.ndarray:
    X, y, years = a[feature_cols(a)].to_numpy(np.float32), a["label"].to_numpy(int), \
        a["year"].to_numpy()
    out = np.zeros(len(a))
    for yr in sorted(set(years)):
        te = years == yr
        clf = make_clf(name, seed).fit(X[~te], y[~te])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return out


def picks(a: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Per pool group: the label of the most probable answer (ties: the earliest answer)."""
    t = a.assign(p=probs).sort_values(["group", "first"])
    best = t.loc[t.groupby("group", sort=False)["p"].idxmax()]
    return best[["group", "pool", "uid", "year", "label"]].set_index("group")


def kind_groups(df: pd.DataFrame) -> dict[str, set]:
    groups = pd.Series(pool_groups(df), index=df.index)
    return {k: set(groups.loc[p.index]) for k, p in eval_pools(df).items()}


def summarise(pk: pd.DataFrame, groups: set) -> dict:
    p = pk.loc[pk.index.isin(groups)]
    by_year = (100 * p.groupby("year")["label"].mean()).round(2)
    return {"mean": float(100 * p["label"].mean()), "n": len(p),
            "by_year": {int(k): float(v) for k, v in by_year.items()},
            "worst_year": float(by_year.min())}


def v3_ref_by_year(cfg: dict) -> dict:
    """The v3 ranker (xgb_ranker / max_prob) refit by LOYO on the base frame, per year."""
    from mmjee_reasoner.verifier.neural import training_frame

    df = training_frame(cfg, with_extra=False)
    scores = loyo(hand_features(df), df["correct"].astype(int).to_numpy(), df["year"].to_numpy(),
                  pool_groups(df), cfg)
    p = eval_pools(df)["v3_16pool"]
    _, per_q = selection_accuracy(p, pd.Series(scores, index=df.index).loc[p.index].to_numpy(),
                                  "max_prob")
    year = df.drop_duplicates("uid").set_index("uid")["year"]
    by_year = (100 * per_q.groupby(year.loc[per_q.index].to_numpy()).mean()).round(2)
    return {"mean": float(100 * per_q.mean()),
            "by_year": {int(k): float(v) for k, v in by_year.items()},
            "worst_year": float(by_year.min())}


def out_dir(cfg: dict) -> Path:
    d = Path(cfg["paths"]["artifacts"]) / "verifier_v4b" / "answer_level"
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_verifier_v4b(cfg: dict) -> dict:
    """LOYO table for the 5 configurations, worst-year choice and gate (no test data)."""
    from mmjee_reasoner.verifier.neural import load_oof

    df, _ = load_oof(cfg, "extra", with_extra=True)     # the Phase-4 training frame
    held_out = set(cfg["data"]["test_years"]) | {2025, 2026}
    assert not df["year"].isin(held_out).any(), "held-out years in the training frame"
    a = answer_rows(df)
    kinds = kind_groups(df)
    report: dict = {"rows_candidates": len(df), "rows_answers": len(a),
                    "features": feature_cols(a), "gate": GATE, "configs": {}}
    for name in CONFIGS:
        pk = picks(a, loyo_probs(a, name, cfg["seed"]))
        report["configs"][name] = {k: summarise(pk, g) for k, g in kinds.items()}
    ranked = sorted(CONFIGS, key=lambda n: (report["configs"][n]["v3_16pool"]["worst_year"],
                                            report["configs"][n]["v3_16pool"]["mean"]),
                    reverse=True)
    chosen = ranked[0]
    ref = v3_ref_by_year(cfg)
    res = report["configs"][chosen]
    report.update(chosen=chosen, v3_ref=ref, majority={
        k: float(100 * selection_accuracy(p, None, "majority")[0])
        for k, p in eval_pools(df).items()})
    report["gate_pass"] = bool(all(res[k]["mean"] > GATE[k] for k in GATE if k in res)
                               and res["v3_16pool"]["worst_year"] >= ref["worst_year"])
    (out_dir(cfg) / "report.json").write_text(json.dumps(report, indent=2))
    log.info("verifier v4b: chosen %s, %s, v3_ref %.1f (worst %.1f), gate %s", chosen,
             {k: round(v["mean"], 1) for k, v in res.items()}, ref["mean"], ref["worst_year"],
             report["gate_pass"])
    return report
