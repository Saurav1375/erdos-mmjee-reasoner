"""Verifier models, metrics and best-of-N selection."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mmjee_reasoner.verifier.features import (
    answer_key,
    embedding_features,
    handcrafted_features,
)

log = logging.getLogger(__name__)


@dataclass
class FeatureMatrix:
    hand: pd.DataFrame
    emb: np.ndarray

    def select(self, feature_set: str) -> tuple[np.ndarray, list[int], list[int]]:
        """Return X and the column indices of handcrafted / embedding parts."""
        h, e = self.hand.to_numpy(np.float32), self.emb.astype(np.float32)
        if feature_set == "handcrafted":
            return h, list(range(h.shape[1])), []
        if feature_set == "embedding":
            return e, [], list(range(e.shape[1]))
        if feature_set == "all":
            X = np.hstack([h, e])
            return X, list(range(h.shape[1])), list(range(h.shape[1], X.shape[1]))
        raise ValueError(feature_set)


def build_features(df: pd.DataFrame, cfg: dict, cache_dir: Path | None = None,
                   pool_cols: tuple[str, ...] = ("uid",)) -> FeatureMatrix:
    """Handcrafted + embedding features (embeddings cached on disk by content hash)."""
    v = cfg["verifier"]
    hand = handcrafted_features(df, pool_cols)
    emb = None
    if cache_dir is not None:
        h = hashlib.sha256()
        h.update(v["embedding_model"].encode())
        h.update(str(v["embedding_tail_chars"]).encode())
        for t in df["text"]:
            h.update((t or "").encode("utf-8", "ignore"))
            h.update(b"\0")
        path = cache_dir / f"emb_{h.hexdigest()[:16]}.npy"
        if path.exists():
            emb = np.load(path)
    if emb is None:
        emb = embedding_features(df, v["embedding_model"], v["embedding_tail_chars"])
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
    return FeatureMatrix(hand=hand, emb=emb)


def make_model(model: str, hand_idx: list[int], emb_idx: list[int], pca_dims: int,
               pos_weight: float, seed: int) -> Pipeline:
    parts = []
    if hand_idx:
        parts.append(("hand", StandardScaler(), hand_idx))
    if emb_idx:
        parts.append(("emb", PCA(n_components=min(pca_dims, len(emb_idx)), random_state=seed),
                      emb_idx))
    pre = ColumnTransformer(parts)
    if model == "logreg":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
    elif model == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
            scale_pos_weight=pos_weight, eval_metric="logloss", random_state=seed, n_jobs=4,
        )
    else:
        raise ValueError(model)
    return Pipeline([("pre", pre), ("clf", clf)])


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    out = {"n": len(y), "pos_rate": float(y.mean()) if len(y) else 0.0,
           "threshold": float(threshold),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0))}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    else:
        out["auc"] = out["ap"] = float("nan")
    return out


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.unique(np.round(p, 3))
    if len(grid) == 0:
        return 0.5
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


# ---------------------------------------------------------------------------
# Best-of-N selection
# ---------------------------------------------------------------------------
SELECTION_METHODS = ("max_prob", "weighted_vote", "majority", "random_expected", "oracle")


def select(pool: pd.DataFrame, scores: np.ndarray | None, method: str) -> dict:
    """Pick one answer for one question's pool; returns {"pred", "correct"} (float)."""
    keys = [answer_key(p, t) for p, t in zip(pool["pred"], pool["question_type"])]
    correct = pool["correct"].astype(float).to_numpy()
    if method == "oracle":
        return {"pred": None, "correct": float(correct.max())}
    if method == "random_expected":
        return {"pred": None, "correct": float(correct.mean())}
    if method == "majority" or scores is None:
        weights = np.ones(len(keys))
    elif method == "weighted_vote":
        weights = np.asarray(scores, dtype=float)
    elif method == "max_prob":
        i = int(np.argmax(scores))  # first max on ties (sample order)
        return {"pred": pool["pred"].iloc[i], "correct": correct[i]}
    else:
        raise ValueError(method)
    totals: dict[str, float] = {}
    first: dict[str, int] = {}
    for i, (k, w) in enumerate(zip(keys, weights)):
        totals[k] = totals.get(k, 0.0) + float(w)
        first.setdefault(k, i)
    best = max(totals, key=lambda k: (totals[k], -first[k]))  # ties -> earliest answer
    i = first[best]
    return {"pred": pool["pred"].iloc[i], "correct": correct[i]}


def selection_accuracy(df: pd.DataFrame, scores: np.ndarray | None, method: str,
                       group_col: str = "uid") -> tuple[float, pd.Series]:
    """Mean correctness of the selected answer over questions; also per-question series."""
    s = pd.Series(scores, index=df.index) if scores is not None else None
    per_q = {}
    for uid, g in df.groupby(group_col, sort=False):
        per_q[uid] = select(g, None if s is None else s.loc[g.index].to_numpy(), method)["correct"]
    ser = pd.Series(per_q)
    return float(ser.mean()), ser


@dataclass
class TrainedVerifier:
    model_name: str
    feature_set: str
    pipeline: Pipeline
    threshold: float
    hand_columns: list[str] = field(default_factory=list)

    def predict(self, fm: FeatureMatrix) -> np.ndarray:
        X, _, _ = fm.select(self.feature_set)
        return self.pipeline.predict_proba(X)[:, 1]
