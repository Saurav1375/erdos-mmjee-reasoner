"""Training and model selection for the learned verifier (Module C).

Data: PoT solver samples on the 2019-2024 questions (``pot_train``), labelled
automatically by the upstream scorer against the gold answers.

Protocol (no 2025 data is touched):

1. Model selection: for every (model, feature set) pair
   * GroupKFold CV on 2019-2023 (groups = twin key, so the English and Hindi
     versions of a question never fall on both sides) -> OOF AUC/AP,
     F1-optimal threshold;
   * fit on 2019-2023, evaluate on 2024 (dev): AUC/AP/F1 and best-of-N
     selection accuracy (max-prob and verifier-weighted vote) vs. majority vote.
2. The primary configuration (model, feature set, selection rule) is the one
   with the best dev selection accuracy (ties: dev AUC).
3. Every configuration is refit on all of 2019-2024 and saved; the decision
   threshold for F1 is the F1-optimal threshold of 2019-2024 OOF predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.pipeline.common import load_candidates
from mmjee_reasoner.verifier.model import (
    FeatureMatrix,
    TrainedVerifier,
    best_f1_threshold,
    build_features,
    classification_metrics,
    make_model,
    selection_accuracy,
)

log = logging.getLogger(__name__)


def verifier_dir(cfg: dict) -> Path:
    from mmjee_reasoner.pipeline.common import subset_suffix

    return artifacts_dir(cfg, f"verifier{subset_suffix(cfg)}", "x").parent


def _subset(fm: FeatureMatrix, rows: np.ndarray) -> FeatureMatrix:
    """Rows by boolean mask or integer positions (GroupKFold yields positions)."""
    rows = np.asarray(rows)
    pos = np.flatnonzero(rows) if rows.dtype == bool else rows
    return FeatureMatrix(hand=fm.hand.iloc[pos].reset_index(drop=True), emb=fm.emb[pos])


def _fit(model: str, fs: str, fm: FeatureMatrix, y: np.ndarray, cfg: dict):
    X, h_idx, e_idx = fm.select(fs)
    pos = max(1, int(y.sum()))
    pipe = make_model(model, h_idx, e_idx, cfg["verifier"]["pca_dims"],
                      pos_weight=(len(y) - pos) / pos, seed=cfg["seed"])
    pipe.fit(X, y)
    return pipe


def oof_predictions(model: str, fs: str, fm: FeatureMatrix, y: np.ndarray, groups: np.ndarray,
                    cfg: dict) -> np.ndarray:
    oof = np.zeros(len(y), dtype=float)
    for tr, va in GroupKFold(n_splits=cfg["verifier"]["cv_folds"]).split(fm.hand, y, groups):
        pipe = _fit(model, fs, _subset(fm, tr), y[tr], cfg)
        X, _, _ = _subset(fm, va).select(fs)
        oof[va] = pipe.predict_proba(X)[:, 1]
    return oof


def train_verifier(cfg: dict, source_exp: str = "pot_train") -> dict:
    v = cfg["verifier"]
    df = load_candidates(cfg, source_exp)
    assert not df["year"].isin(cfg["data"]["test_years"]).any(), "test data in verifier source"
    df = df[df["year"].isin(cfg["data"]["train_years"])].reset_index(drop=True)
    y = df["correct"].astype(int).to_numpy()
    groups = df["twin_key"].to_numpy()
    out_dir = verifier_dir(cfg)
    fm = build_features(df, cfg, cache_dir=out_dir / "emb_cache")
    log.info("verifier data: %d solutions, %d questions, positive rate %.3f",
             len(df), df["uid"].nunique(), y.mean())

    dev_mask = (df["year"] == v["dev_year"]).to_numpy()
    cv_mask = (df["year"] < v["dev_year"]).to_numpy()
    df_dev = df[dev_mask]
    report = {"n_solutions": len(df), "n_questions": int(df["uid"].nunique()),
              "pos_rate": float(y.mean()), "configs": [], "dev_baselines": {}}
    for m in ("majority", "random_expected", "oracle"):
        report["dev_baselines"][m] = selection_accuracy(df_dev, None, m)[0]

    for model in v["models"]:
        for fs in v["feature_sets"]:
            fm_cv = _subset(fm, cv_mask)
            oof = oof_predictions(model, fs, fm_cv, y[cv_mask], groups[cv_mask], cfg)
            thr = best_f1_threshold(y[cv_mask], oof)
            pipe = _fit(model, fs, fm_cv, y[cv_mask], cfg)
            X_dev, _, _ = _subset(fm, dev_mask).select(fs)
            p_dev = pipe.predict_proba(X_dev)[:, 1]
            entry = {
                "model": model, "feature_set": fs,
                "cv": classification_metrics(y[cv_mask], oof, thr),
                "dev": classification_metrics(y[dev_mask], p_dev, thr),
                "dev_selection": {
                    meth: selection_accuracy(df_dev, p_dev, meth)[0]
                    for meth in ("max_prob", "weighted_vote")
                },
            }
            report["configs"].append(entry)
            log.info("%-8s %-11s cvAUC %.3f devAUC %.3f devF1 %.3f sel %s", model, fs,
                     entry["cv"]["auc"], entry["dev"]["auc"], entry["dev"]["f1"],
                     {k: round(x, 3) for k, x in entry["dev_selection"].items()})

    def key(e):
        best_sel = max(e["dev_selection"].values())
        auc = e["dev"]["auc"]
        return (round(best_sel, 6), auc if auc == auc else -1.0)

    best = max(report["configs"], key=key)
    best_method = max(best["dev_selection"], key=best["dev_selection"].get)
    report["primary"] = {"model": best["model"], "feature_set": best["feature_set"],
                         "selection": best_method}
    # Honest flag: does the learned selector beat plain majority voting on dev?
    report["primary_beats_majority_on_dev"] = bool(
        best["dev_selection"][best_method] > report["dev_baselines"]["majority"])

    # Refit every configuration on all training years; threshold from full OOF.
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    for e in report["configs"]:
        oof = oof_predictions(e["model"], e["feature_set"], fm, y, groups, cfg)
        thr = best_f1_threshold(y, oof)
        e["full_oof"] = classification_metrics(y, oof, thr)
        pipe = _fit(e["model"], e["feature_set"], fm, y, cfg)
        tv = TrainedVerifier(e["model"], e["feature_set"], pipe, thr, list(fm.hand.columns))
        joblib.dump(tv, out_dir / "models" / f"{e['model']}__{e['feature_set']}.joblib")
    # Feature importances of the primary model (interpretability).
    report["primary_importance"] = feature_importance(
        load_verifier(cfg, best["model"], best["feature_set"]))

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    log.info("primary verifier: %s", report["primary"])
    return report


def load_verifier(cfg: dict, model: str | None = None, feature_set: str | None = None):
    d = verifier_dir(cfg)
    if model is None:
        prim = json.loads((d / "report.json").read_text())["primary"] if (
            d / "report.json").exists() else None
        if prim is None:
            raise FileNotFoundError("no trained verifier - run `mmjee train-verifier`")
        model, feature_set = prim["model"], prim["feature_set"]
    path = d / "models" / f"{model}__{feature_set}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def primary_selection(cfg: dict) -> dict:
    return json.loads((verifier_dir(cfg) / "report.json").read_text())["primary"]


def feature_importance(tv: TrainedVerifier, top: int = 15) -> list[tuple[str, float]]:
    clf = tv.pipeline.named_steps["clf"]
    n_hand = len(tv.hand_columns) if tv.feature_set in ("handcrafted", "all") else 0
    names = (tv.hand_columns if n_hand else [])
    pre = tv.pipeline.named_steps["pre"]
    n_total = sum(len(t[2]) if t[0] == "hand" else t[1].n_components_
                  for t in pre.transformers_ if t[0] in ("hand", "emb"))
    names = names + [f"emb_pc{i}" for i in range(n_total - len(names))]
    if hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_[0])
    else:
        vals = clf.feature_importances_
    order = np.argsort(-vals)[:top]
    return [(names[i], float(vals[i])) for i in order]
