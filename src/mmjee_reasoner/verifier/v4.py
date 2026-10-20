"""Verifier v4 (Module C, Phase 4): selector variants judged by leave-one-year-out (V4_PREREG.md).

Variants (all on the handcrafted + chain + pool features of verifier v3, model xgb_ranker):

* ``v3_ref``     - verifier v3 as frozen, refit inside this LOYO (the reference to beat)
* ``no_length``  - the same without the length features (``LENGTH_COLS``)
* ``neural``     - the neural verifier's out-of-year score alone (verifier/neural.py)
* ``stacked``    - v3 features + the neural out-of-year score as one more feature

With ``with_extra`` the training frame also contains the Phase-4 v3-style pools (pool = "v4"),
and selection is reported separately on the v3 16-pools and on the Phase-4 pools. A variant
passes its gate when its best LOYO selection accuracy beats ``v3_ref`` on every reported pool
kind. Nothing here touches 2025/2026; test scoring is ``score_test_pool`` (called once, W5).
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from mmjee_reasoner.verifier.features import handcrafted_features
from mmjee_reasoner.verifier.model import FeatureMatrix, selection_accuracy
from mmjee_reasoner.verifier.v3 import (
    CHAIN_COLS,
    METHODS,
    fit_model,
    pool_agreement,
    predict_model,
)

log = logging.getLogger(__name__)

LENGTH_COLS = ["log_chars", "log_tokens", "n_steps", "truncated"]
VARIANTS = ("v3_ref", "no_length", "neural", "stacked")


def hand_features(df: pd.DataFrame) -> pd.DataFrame:
    """verifier v3 handcrafted + chain + pool-agreement features, computed per pool kind."""
    parts = []
    for _, g in df.groupby("pool", sort=False):
        hand = handcrafted_features(g, pool_cols=("uid", "source"))
        extra = pd.concat([g[["is_corrected", "has_chain", *CHAIN_COLS]].astype(np.float32),
                           pool_agreement(g)], axis=1)
        parts.append(pd.concat([hand, extra], axis=1))
    return pd.concat(parts).loc[df.index]


def variant_matrix(variant: str, hand: pd.DataFrame, neural: np.ndarray | None) -> pd.DataFrame:
    if variant == "v3_ref":
        return hand
    if variant == "no_length":
        return hand.drop(columns=LENGTH_COLS)
    if variant == "stacked":
        return hand.assign(neural_score=np.asarray(neural, dtype=np.float32))
    raise ValueError(variant)


def _fm(X: pd.DataFrame) -> FeatureMatrix:
    return FeatureMatrix(hand=X.reset_index(drop=True), emb=np.zeros((len(X), 0), np.float32))


def fit_ranker(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: dict):
    return fit_model("xgb_ranker", "handcrafted", _fm(X), y, groups, cfg)


def predict_ranker(pipe, X: pd.DataFrame) -> np.ndarray:
    return predict_model(pipe, "xgb_ranker", "handcrafted", _fm(X))


def pool_groups(df: pd.DataFrame) -> np.ndarray:
    """Ranking groups: one per (question, pool kind). v3 rows keep the bare uid, so a v3-only
    frame reproduces verifier v3's LOYO exactly (XGBRanker depends on the row order)."""
    return np.where(df["pool"] == "v3", df["uid"], df["uid"] + "|" + df["pool"])


def loyo(X: pd.DataFrame, y: np.ndarray, years: np.ndarray, groups: np.ndarray, cfg: dict
         ) -> np.ndarray:
    oof = np.zeros(len(y))
    for yr in sorted(set(years)):
        te = years == yr
        pipe = fit_ranker(X[~te], y[~te], groups[~te], cfg)
        oof[te] = predict_ranker(pipe, X[te])
    return oof


def eval_pools(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pool kinds selection is judged on: v3 16-pools (questions with chains) and v4 pools."""
    v3 = df[df["pool"] == "v3"]
    full16 = v3.groupby("uid")["has_chain"].transform("max") > 0
    out = {"v3_16pool": v3[full16]}
    if (df["pool"] == "v4").any():
        out["v4_pool"] = df[df["pool"] == "v4"]
    return out


def selection_table(df: pd.DataFrame, scores: np.ndarray) -> dict:
    s = pd.Series(scores, index=df.index)
    out = {}
    for kind, p in eval_pools(df).items():
        out[kind] = {m: 100 * selection_accuracy(p, s.loc[p.index].to_numpy(), m)[0]
                     for m in METHODS}
    return out


def baselines(df: pd.DataFrame) -> dict:
    return {kind: {m: 100 * selection_accuracy(p, None, m)[0]
                   for m in ("majority", "random_expected", "oracle")}
            for kind, p in eval_pools(df).items()}


def train_verifier_v4(cfg: dict, neural_tag: str = "base", with_extra: bool = False) -> dict:
    """LOYO report for every variant + gate decisions; fits the passing variants on all years."""
    from mmjee_reasoner.verifier.neural import load_oof, neural_dir

    df, neural = load_oof(cfg, neural_tag, with_extra)
    hand = hand_features(df)
    y = df["correct"].astype(int).to_numpy()
    years = df["year"].to_numpy()
    groups = pool_groups(df)
    tag = neural_tag   # one selector report per neural run (e.g. base; extra = --with-extra)
    report: dict = {"tag": tag, "rows": len(df), "pools": df["pool"].value_counts().to_dict(),
                    "baselines": baselines(df), "variants": {}}
    for v in VARIANTS:
        scores = neural if v == "neural" else loyo(variant_matrix(v, hand, neural), y, years,
                                                   groups, cfg)
        report["variants"][v] = selection_table(df, scores)
    ref = report["variants"]["v3_ref"]
    for v, tab in report["variants"].items():
        best = max(METHODS, key=lambda m: tab["v3_16pool"][m])
        tab["best_method"] = best
        tab["gate_pass"] = v != "v3_ref" and all(
            tab[k][best] > max(ref[k].values()) for k in eval_pools(df))
    out = neural_dir(cfg, neural_tag).parent / f"selectors_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    for v in ("no_length", "stacked"):
        if report["variants"][v]["gate_pass"]:
            pipe = fit_ranker(variant_matrix(v, hand, neural), y, groups, cfg)
            joblib.dump(pipe, out / f"{v}.joblib")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    log.info("verifier v4 [%s]: %s", tag, {v: (round(t["v3_16pool"][t["best_method"]], 1),
                                               t["gate_pass"])
                                           for v, t in report["variants"].items()})
    return report
