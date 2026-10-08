"""Aggregate metrics: Pass@1, self-consistency, bootstrap CIs, breakdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmjee_reasoner.scoring import calculate_statistics

BREAKDOWNS = ("question_type", "language", "subject", "requires_image")


def per_question_mean(df: pd.DataFrame) -> pd.Series:
    """Per-question mean correctness over samples (Pass@1 contribution)."""
    return df.groupby("uid")["correct"].mean().astype(float)


def pass_at_1(df: pd.DataFrame) -> dict:
    """Pass@1 as in the base paper: mean of per-run accuracies, one run per sample index.

    Uses the upstream ``calculate_statistics`` (mean, std, 95% t-interval over runs).
    Requires the same samples for every question.
    """
    runs = df.groupby("sample")["correct"].mean().astype(float) * 100
    stats = calculate_statistics(runs.tolist())
    return {
        "pass1": float(per_question_mean(df).mean() * 100),
        "runs": len(runs),
        "run_mean": float(stats["mean_accuracy"]),
        "run_std": float(stats["std_accuracy"]) if len(runs) > 1 else 0.0,
        "run_ci95": [float(x) for x in stats["confidence_interval_95"]],
    }


def _cluster_means(x: pd.Series, clusters: pd.Series | None, n_boot: int,
                   seed: int) -> np.ndarray:
    """Bootstrap means (in %) resampling clusters (English/Hindi twins) with replacement.

    The 190 test questions are 95 EN/HI pairs, so resampling pairs instead of
    questions gives honest (wider) intervals. ``clusters=None`` resamples questions.
    """
    rng = np.random.default_rng(seed)
    if clusters is None:
        groups = [np.array([i]) for i in range(len(x))]
    else:
        c = clusters.reindex(x.index).fillna(pd.Series(x.index, index=x.index))
        groups = [np.flatnonzero((c == k).to_numpy()) for k in pd.unique(c)]
    vals = x.to_numpy(dtype=float)
    sums = np.array([vals[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    pick = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    return sums[pick].sum(axis=1) / sizes[pick].sum(axis=1) * 100


def bootstrap_ci(scores: pd.Series, n_boot: int = 2000, seed: int = 0,
                 clusters: pd.Series | None = None) -> tuple[float, float]:
    """95% percentile (cluster) bootstrap CI (in %) of the mean per-question score."""
    if len(scores) == 0:
        return (float("nan"), float("nan"))
    means = _cluster_means(scores, clusters, n_boot, seed)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 0,
                     clusters: pd.Series | None = None) -> dict:
    """Difference (a - b, in points) with 95% CI and one-sided p = P(diff <= 0)."""
    common = a.index.intersection(b.index)
    diff = a.loc[common] - b.loc[common]
    d = diff.to_numpy(dtype=float)
    means = _cluster_means(diff, clusters, n_boot, seed)
    return {"diff": float(d.mean() * 100), "ci95": [float(np.percentile(means, 2.5)),
                                                    float(np.percentile(means, 97.5))],
            "p_le_0": float((means <= 0).mean()), "n": len(d)}


def mcnemar(a: pd.Series, b: pd.Series) -> dict:
    """Exact McNemar test for paired binary outcomes (a, b indexed by question).

    Note: treats questions as independent although EN/HI twins are correlated, so
    the p-value is optimistic; the cluster bootstrap CI is the primary statistic.
    """
    from scipy.stats import binomtest

    common = a.index.intersection(b.index)
    x, y = a.loc[common].round().astype(int), b.loc[common].round().astype(int)
    n01 = int(((x == 0) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only": n10, "b_only": n01, "p": float(p)}


def breakdown(scores: pd.Series, questions: pd.DataFrame) -> dict:
    """Mean score (%) per category for each breakdown column."""
    q = questions.set_index("uid").loc[scores.index]
    out = {}
    for col in BREAKDOWNS:
        out[col] = {str(k): float(scores[g.index].mean() * 100)
                    for k, g in q.groupby(col)}
    return out
