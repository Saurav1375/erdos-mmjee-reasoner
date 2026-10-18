"""Verifier v3 (Module C, v3): trained on the pools it has to rank, chosen by leave-one-year-out.

What v2 did not do (see PLAN_v3.md, Phase 4):

* **Training pools look like test pools.** v2 learned only from original PoT samples, but
  at test time it ranks originals + corrected solutions. v3 trains on 2019-2024 pools that
  contain both (``train_corr_cross``: the correction loop run on a stratified subset of the
  training questions) plus the original-only pools of the remaining questions.
* **Critic signals as features.** Whether the critic flagged the original, how many rounds
  ran, whether the answer changed, whether the correction failed - for the corrected row
  and for the original row of the same chain. Plus agreement over the whole 16-pool and
  with the other half of the pool.
* **A ranking model** (XGBRanker, pairwise, grouped by question) next to logreg / XGBoost.
* **Leave-one-year-out model selection** over 2019-2024 (6 folds) instead of one 2024 dev
  year (204 questions, where 1 question = 0.5 points).
* **Learned correction acceptance**: keep a chain's corrected answer only if
  score(corrected) - score(original) > delta, with delta chosen on out-of-year scores.

No 2025/2026 data is used anywhere in this module.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mmjee_reasoner.verifier.features import answer_key, handcrafted_features
from mmjee_reasoner.verifier.model import (
    FeatureMatrix,
    best_f1_threshold,
    build_features,
    classification_metrics,
    make_model,
    selection_accuracy,
)

log = logging.getLogger(__name__)

CHAIN_COLS = ["corr_critic_flagged_first", "corr_n_flags", "corr_rounds", "corr_answer_changed",
              "correction_failed_last"]
MODELS = ("logreg", "xgboost", "xgb_ranker")
FEATURE_SETS = ("handcrafted", "all")
METHODS = ("max_prob", "weighted_vote", "log_odds_vote")
DELTA_GRID = np.round(np.arange(-0.30, 0.301, 0.02), 2)


# ---------------------------------------------------------------------------
# Pools and features
# ---------------------------------------------------------------------------
def prepare_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Originals + corrected rows of the same chains -> one pool frame with chain features.

    Corrected rows (``source == "corrected"``) carry the chain columns; they are copied onto
    the original row of the same (uid, sample). Rows without a chain get zeros and
    ``has_chain = 0``.
    """
    df = df.reset_index(drop=True).copy()
    corr = df["source"] == "corrected"
    df["is_corrected"] = corr.astype(float)
    for c in CHAIN_COLS:
        if c not in df:
            df[c] = np.nan
    chain = df.loc[corr, ["uid", "sample", *CHAIN_COLS]].drop_duplicates(["uid", "sample"])
    orig = df.loc[~corr, ["uid", "sample"]].merge(chain, on=["uid", "sample"], how="left")
    for c in CHAIN_COLS:
        df.loc[~corr, c] = orig[c].to_numpy()
    df["has_chain"] = df["corr_rounds"].notna().astype(float)
    for c in CHAIN_COLS:
        df[c] = df[c].astype(float).fillna(0.0)
    return df


def pool_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Agreement over the whole pool of a question and with the other half of the pool."""
    keys = pd.Series([answer_key(p, t) for p, t in zip(df["pred"].fillna("").astype(str),
                                                         df["question_type"])], index=df.index)
    out = pd.DataFrame(index=df.index, columns=["full_agree", "full_is_majority",
                                                "other_half_agree"], dtype=float)
    for _, g in df.groupby("uid", sort=False):
        k = keys.loc[g.index]
        counts = Counter(k)
        n = len(k)
        top = counts.most_common(1)[0][0]
        out.loc[g.index, "full_agree"] = [(counts[x] - 1) / max(1, n - 1) for x in k]
        out.loc[g.index, "full_is_majority"] = [float(x == top) for x in k]
        corr = g["is_corrected"].to_numpy() > 0
        for half in (True, False):
            idx = g.index[corr == half]
            other = Counter(k[g.index[corr != half]])
            m = sum(other.values())
            out.loc[idx, "other_half_agree"] = [other[x] / m if m else -1.0 for x in k[idx]]
    return out.astype(np.float32)


def v3_features(df: pd.DataFrame, cfg: dict, cache_dir: Path | None) -> FeatureMatrix:
    """v2 features (sub-pool agreement per source) + chain and whole-pool features."""
    fm = build_features(df, cfg, cache_dir=cache_dir, pool_cols=("uid", "source"))
    extra = pd.concat([df[["is_corrected", "has_chain", *CHAIN_COLS]].astype(np.float32),
                       pool_agreement(df)], axis=1)
    hand = pd.concat([fm.hand, extra.set_index(fm.hand.index)], axis=1)
    return FeatureMatrix(hand=hand, emb=fm.emb)


def _subset(fm: FeatureMatrix, mask: np.ndarray) -> FeatureMatrix:
    pos = np.flatnonzero(mask)
    return FeatureMatrix(hand=fm.hand.iloc[pos].reset_index(drop=True), emb=fm.emb[pos])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_model(model: str, fs: str, fm: FeatureMatrix, y: np.ndarray, groups: np.ndarray,
              cfg: dict) -> Pipeline:
    X, h_idx, e_idx = fm.select(fs)
    pca = cfg["verifier"]["pca_dims"]
    if model == "xgb_ranker":
        from xgboost import XGBRanker

        base = make_model("logreg", h_idx, e_idx, pca, 1.0, cfg["seed"])  # reuse preprocessing
        pipe = Pipeline([("pre", base.named_steps["pre"]), ("clf", XGBRanker(
            objective="rank:pairwise", n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=2, random_state=cfg["seed"],
            n_jobs=4))])
        order = np.argsort(groups, kind="stable")       # XGBRanker needs grouped rows
        _, qid = np.unique(groups[order], return_inverse=True)
        pipe.fit(X[order], y[order], clf__qid=qid)
        return pipe
    pos = max(1, int(y.sum()))
    pipe = make_model(model, h_idx, e_idx, pca, pos_weight=(len(y) - pos) / pos,
                      seed=cfg["seed"])
    pipe.fit(X, y)
    return pipe


def predict_model(pipe: Pipeline, model: str, fs: str, fm: FeatureMatrix) -> np.ndarray:
    X, _, _ = fm.select(fs)
    if model == "xgb_ranker":
        return _sigmoid(pipe.predict(X))
    return pipe.predict_proba(X)[:, 1]


@dataclass
class TrainedVerifierV3:
    """Same interface as v2's TrainedVerifier for the Evaluator, plus ``predict_pool``."""

    model_name: str
    feature_set: str
    pipeline: Pipeline
    threshold: float
    delta: float
    hand_columns: list[str] = field(default_factory=list)

    def predict(self, fm: FeatureMatrix) -> np.ndarray:
        return predict_model(self.pipeline, self.model_name, self.feature_set, fm)

    def predict_pool(self, df: pd.DataFrame, cfg: dict, cache_dir: Path | None) -> np.ndarray:
        pool = prepare_pool(df)
        return self.predict(v3_features(pool, cfg, cache_dir))


# ---------------------------------------------------------------------------
# Correction acceptance gate
# ---------------------------------------------------------------------------
def gate_choices(pool: pd.DataFrame, scores: np.ndarray, delta: float) -> pd.DataFrame:
    """Per chain: take the corrected answer iff s(corrected) - s(original) > delta.

    Returns one row per chain (uid, sample, taken, final_correct)."""
    p = pool.assign(_s=scores)
    o = p[(p["is_corrected"] == 0) & (p["has_chain"] > 0)].set_index(["uid", "sample"])
    c = p[p["is_corrected"] == 1].set_index(["uid", "sample"]).loc[o.index]
    take = (c["_s"].to_numpy() - o["_s"].to_numpy()) > delta
    fin = np.where(take, c["correct"].astype(bool).to_numpy(), o["correct"].astype(bool).to_numpy())
    return pd.DataFrame({"uid": o.index.get_level_values(0), "sample": o.index.get_level_values(1),
                         "taken": take, "final_correct": fin.astype(float)})


def gate_outcome(pool: pd.DataFrame, scores: np.ndarray, delta: float) -> dict:
    """Per-chain choice: corrected answer iff s(corrected) - s(original) > delta."""
    p = pool.assign(_s=scores)
    o = p[(p["is_corrected"] == 0) & (p["has_chain"] > 0)].set_index(["uid", "sample"])
    c = p[p["is_corrected"] == 1].set_index(["uid", "sample"]).loc[o.index]
    take = (c["_s"].to_numpy() - o["_s"].to_numpy()) > delta
    orig = o["correct"].astype(bool).to_numpy()
    corr = c["correct"].astype(bool).to_numpy()
    fin = np.where(take, corr, orig)
    return {"delta": float(delta), "n_chains": int(len(o)),
            "acc_original": 100.0 * orig.mean(), "acc_corrected": 100.0 * corr.mean(),
            "acc_gated": 100.0 * fin.mean(), "accepted": int(take.sum()),
            "wrong_to_right_ungated": int((~orig & corr).sum()),
            "right_to_wrong_ungated": int((orig & ~corr).sum()),
            "wrong_to_right_gated": int((~orig & fin).sum()),
            "right_to_wrong_gated": int((orig & ~fin).sum())}


def choose_delta(pool: pd.DataFrame, scores: np.ndarray) -> tuple[float, list[dict]]:
    """delta maximising per-chain accuracy (ties -> delta closest to 0)."""
    grid = [gate_outcome(pool, scores, d) for d in DELTA_GRID]
    best = max(grid, key=lambda g: (round(g["acc_gated"], 9), -abs(g["delta"])))
    return best["delta"], grid


def gate_nested_loyo(pool: pd.DataFrame, scores: np.ndarray) -> dict:
    """Honest gate estimate: for each year, delta is chosen on the other years only and
    applied to that year (scores are already out-of-fold); counts are summed."""
    keys = ["n_chains", "accepted", "wrong_to_right_ungated", "right_to_wrong_ungated",
            "wrong_to_right_gated", "right_to_wrong_gated"]
    tot, deltas, fin, orig, corr = dict.fromkeys(keys, 0), {}, 0.0, 0.0, 0.0
    years = pool["year"].to_numpy()
    for y in sorted(set(years)):
        tr, te = years != y, years == y
        d, _ = choose_delta(pool[tr].reset_index(drop=True), scores[tr])
        g = gate_outcome(pool[te].reset_index(drop=True), scores[te], d)
        deltas[int(y)] = d
        for k in keys:
            tot[k] += g[k]
        fin += g["acc_gated"] * g["n_chains"]
        orig += g["acc_original"] * g["n_chains"]
        corr += g["acc_corrected"] * g["n_chains"]
    n = tot["n_chains"]
    return {**tot, "deltas_by_year": deltas, "acc_original": orig / n,
            "acc_corrected": corr / n, "acc_gated": fin / n}


# ---------------------------------------------------------------------------
# Training with leave-one-year-out selection
# ---------------------------------------------------------------------------
def verifier_v3_dir(cfg: dict) -> Path:
    from mmjee_reasoner.verifier.train import verifier_dir

    return verifier_dir(cfg)


def training_pool(cfg: dict, originals_exp: str = "pot_train",
                  corrected_exp: str = "train_corr_cross") -> pd.DataFrame:
    from mmjee_reasoner.pipeline.common import load_candidates

    o = load_candidates(cfg, originals_exp)
    c = load_candidates(cfg, corrected_exp)
    df = pd.concat([o, c], ignore_index=True)
    held_out = set(cfg["data"]["test_years"]) | set(cfg["rag"].get("dedup_years", []))
    assert not df["year"].isin(held_out).any(), "held-out years in verifier training data"
    df = df[df["year"].isin(cfg["data"]["train_years"])]
    return prepare_pool(df)


def loyo_scores(model: str, fs: str, fm: FeatureMatrix, y: np.ndarray, years: np.ndarray,
                groups: np.ndarray, cfg: dict) -> np.ndarray:
    oof = np.zeros(len(y))
    for yr in sorted(set(years)):
        te = years == yr
        pipe = fit_model(model, fs, _subset(fm, ~te), y[~te], groups[~te], cfg)
        oof[te] = predict_model(pipe, model, fs, _subset(fm, te))
    return oof


def train_verifier_v3(cfg: dict) -> dict:
    pool = training_pool(cfg)
    out_dir = verifier_v3_dir(cfg)
    fm = v3_features(pool, cfg, out_dir / "emb_cache")
    y = pool["correct"].astype(int).to_numpy()
    years = pool["year"].to_numpy()
    groups = pool["uid"].to_numpy()
    full16 = pool.groupby("uid")["has_chain"].transform("max").to_numpy() > 0
    p16, p8 = pool[full16], pool[(~full16) & (pool["is_corrected"] == 0)]
    report: dict = {
        "n_rows": len(pool), "n_questions": int(pool["uid"].nunique()),
        "n_questions_16pool": int(p16["uid"].nunique()), "pos_rate": float(y.mean()),
        "baselines_16pool": {m: 100 * selection_accuracy(p16, None, m)[0]
                             for m in ("majority", "random_expected", "oracle")},
        "baselines_8pool": {m: 100 * selection_accuracy(p8, None, m)[0]
                            for m in ("majority", "random_expected", "oracle")},
        "configs": []}
    oofs = {}
    for model in MODELS:
        for fs in FEATURE_SETS:
            oof = loyo_scores(model, fs, fm, y, years, groups, cfg)
            oofs[(model, fs)] = oof
            s = pd.Series(oof, index=pool.index)
            entry = {"model": model, "feature_set": fs,
                     "loyo": classification_metrics(y, oof, best_f1_threshold(y, oof)),
                     "sel_16pool": {m: 100 * selection_accuracy(p16, s.loc[p16.index].to_numpy(),
                                                                m)[0] for m in METHODS},
                     "sel_8pool": {m: 100 * selection_accuracy(p8, s.loc[p8.index].to_numpy(),
                                                               m)[0] for m in METHODS},
                     "sel_16pool_by_year": {
                         int(yr): {m: 100 * selection_accuracy(
                             g, s.loc[g.index].to_numpy(), m)[0] for m in METHODS}
                         for yr, g in p16.groupby("year")}}
            report["configs"].append(entry)
            log.info("%-10s %-11s AUC %.3f  sel16 %s  sel8 %s", model, fs, entry["loyo"]["auc"],
                     {k: round(v, 1) for k, v in entry["sel_16pool"].items()},
                     {k: round(v, 1) for k, v in entry["sel_8pool"].items()})

    def key(e):
        m = max(e["sel_16pool"], key=e["sel_16pool"].get)
        return (round(e["sel_16pool"][m], 6), round(e["sel_8pool"][m], 6), e["loyo"]["auc"])

    best = max(report["configs"], key=key)
    method = max(best["sel_16pool"], key=best["sel_16pool"].get)
    report["primary"] = {"model": best["model"], "feature_set": best["feature_set"],
                         "selection": method}
    report["primary_beats_majority_16pool"] = bool(
        best["sel_16pool"][method] > report["baselines_16pool"]["majority"])
    oof = oofs[(best["model"], best["feature_set"])]
    s16 = pd.Series(oof, index=pool.index).loc[p16.index].to_numpy()
    delta, grid = choose_delta(p16.reset_index(drop=True), s16)
    report["gate"] = {"delta": delta, "chosen": gate_outcome(p16.reset_index(drop=True), s16,
                                                             delta),
                      "grid": grid,
                      "nested_loyo": gate_nested_loyo(p16.reset_index(drop=True), s16)}

    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    for e in report["configs"]:
        pipe = fit_model(e["model"], e["feature_set"], fm, y, groups, cfg)
        thr = e["loyo"]["threshold"]
        tv = TrainedVerifierV3(e["model"], e["feature_set"], pipe, thr, delta,
                               list(fm.hand.columns))
        joblib.dump(tv, out_dir / "models" / f"{e['model']}__{e['feature_set']}.joblib")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    log.info("verifier v3 primary: %s (beats majority on 16-pools: %s), gate delta %.2f",
             report["primary"], report["primary_beats_majority_16pool"], delta)
    return report
