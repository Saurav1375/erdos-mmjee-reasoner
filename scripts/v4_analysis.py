"""Phase 4 diagnostics (W1, W4d) on existing generations (CPU only).

Writes artifacts_v4/analysis/w1.json. Every number on 2025/2026 here concerns the FROZEN v3
verifier and runs that already exist (they were seen when v3 was evaluated), so nothing new is
fitted on or selected with test data. Sections:

* verifier_vs_majority - per year: verifier v3 pick vs majority vote vs oracle on the v3_full
  pool, paired cluster bootstrap of (verifier - majority)
* shift      - pool statistics (tokens, truncation, agreement) and standardized mean differences
  of the verifier's handcrafted features: training pools vs 2025 vs 2026
* by_truncation - verifier minus majority by how much of a question's pool is cut off
* errors_2026 - where the final 2026 picks fail (selector miss / answer never in the pool) by
  language, image, subject
* rag_fires  - RAG exemplars per question and accuracy where RAG fires vs not (pot_rag vs pot)
* mixed_pools - W4d: majority vote over 8 PoT, 8 PoT+RAG, or 4 + 4 mixed samples

usage: .venv/bin/python scripts/v4_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmjee_reasoner.config import CONFIG_DIR, load_config  # noqa: E402
from mmjee_reasoner.pipeline.evaluate import Evaluator  # noqa: E402
from mmjee_reasoner.scoring.metrics import paired_bootstrap  # noqa: E402
from mmjee_reasoner.verifier.features import handcrafted_features  # noqa: E402
from mmjee_reasoner.verifier.model import selection_accuracy  # noqa: E402
from mmjee_reasoner.verifier.v3 import pool_agreement, prepare_pool, training_pool  # noqa: E402

OUT = ROOT / "artifacts_v4" / "analysis" / "w1.json"
SHIFT_FEATURES = ["log_tokens", "log_chars", "n_steps", "truncated", "n_code_blocks",
                  "code_ok_frac", "hedges_per_kchar", "valid_format", "agree_frac",
                  "is_majority", "distinct_frac"]


def year_cfg(year: int) -> dict:
    """Frozen v3 evaluation config, reading the copies in artifacts_v4 (verifier v3)."""
    files = [CONFIG_DIR / "v3.yaml"]
    if year == 2026:
        files.append(CONFIG_DIR / "v3_test2026.yaml")
    files += [CONFIG_DIR / "v3_final.yaml", CONFIG_DIR / "v3_eval.yaml"]
    cfg = load_config(*files)
    cfg["paths"]["artifacts"] = str(ROOT / "artifacts_v4")
    return cfg


def pool_stats(df: pd.DataFrame) -> dict:
    trunc = (df["finish"] == "length").mean() if "finish" in df else float("nan")
    return {"rows": int(len(df)), "questions": int(df["uid"].nunique()),
            "mean_tokens": float(df["completion_tokens"].mean()),
            "truncated_pct": float(100 * trunc), "acc_pct": float(100 * df["correct"].mean())}


def smd(a: pd.Series, b: pd.Series) -> float:
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((b.mean() - a.mean()) / s) if s > 0 else 0.0


def main() -> None:
    out: dict = {"note": "frozen v3 verifier and existing runs only; 2025/2026 already seen"}
    train = training_pool(year_cfg(2025))
    hf_train = handcrafted_features(train, ("uid", "source"))
    out["shift"] = {"train": pool_stats(train)}
    out["verifier_vs_majority"], out["by_truncation"] = {}, {}
    per_year = {}
    for year in (2025, 2026):
        cfg = year_cfg(year)
        ev = Evaluator(cfg)
        pool = ev.pool("v3_full")
        scores = ev.verifier_scores(pool)
        method = ev.verifiers()["primary"]["selection"]
        _, ver = selection_accuracy(pool, scores, method)
        _, maj = selection_accuracy(pool, None, "majority")
        _, orc = selection_accuracy(pool, None, "oracle")
        clusters = ev.clusters
        out["verifier_vs_majority"][year] = {
            "verifier": 100 * ver.mean(), "majority": 100 * maj.mean(), "oracle": 100 * orc.mean(),
            "verifier_minus_majority": paired_bootstrap(ver, maj, clusters=clusters),
            "selection": method}
        out["shift"][year] = pool_stats(pool)
        hf = handcrafted_features(pool, ("uid", "source"))
        out["shift"][f"smd_train_to_{year}"] = {f: smd(hf_train[f], hf[f]) for f in SHIFT_FEATURES}
        tr = pool.assign(t=(pool["finish"] == "length").astype(float)).groupby("uid")["t"].mean()
        bucket = pd.cut(tr, [-0.01, 0.25, 0.5, 0.75, 1.0], labels=["0-25", "25-50", "50-75",
                                                                  "75-100"])
        out["by_truncation"][year] = {
            str(b): {"n": int((bucket == b).sum()),
                     "verifier": float(100 * ver[bucket == b].mean()),
                     "majority": float(100 * maj[bucket == b].mean())}
            for b in bucket.cat.categories}
        per_year[year] = (ev, pool, ver, maj, orc)

    ev, pool, ver, maj, orc = per_year[2026]
    q = ev.questions.set_index("uid")
    cat = pd.Series("correct", index=ver.index)
    cat[(ver < 0.5) & (orc > 0.5)] = "selector_miss"
    cat[orc < 0.5] = "never_in_pool"
    agree = pool_agreement(prepare_pool(pool))
    pool = pool.assign(maj_share=agree["full_agree"].to_numpy())
    right_share = pool[pool["correct"].astype(bool)].groupby("uid").size() / \
        pool.groupby("uid").size()
    errs = {"counts": cat.value_counts().to_dict(),
            "selector_miss_right_answer_share": float(
                right_share.reindex(cat[cat == "selector_miss"].index).fillna(0).mean())}
    for col in ("language", "requires_image", "subject", "question_type"):
        g = pd.crosstab(q.loc[cat.index, col], cat, normalize="index") * 100
        errs[col] = {str(k): {c: float(v) for c, v in row.items()} for k, row in g.iterrows()}
    out["errors_2026"] = errs

    out["rag_fires"], out["mixed_pools"] = {}, {}
    for year in (2025, 2026):
        ev = per_year[year][0]
        res = {}
        for exp in ("pot_rag", "v3_pot"):
            if not ev.has(exp):
                continue
            c = ev.cands(exp)
            fired = c.groupby("uid")["exemplars"].apply(lambda s: any(len(x) > 0 for x in s))
            res[exp] = {"fired_pct": float(100 * fired.mean())}
            if ev.has("pot") and exp == "pot_rag":
                a = c.groupby("uid")["correct"].mean()
                b = ev.cands("pot").groupby("uid")["correct"].mean()
                for name, m in (("fired", fired), ("not_fired", ~fired)):
                    u = m[m].index
                    res[exp][name] = {"n": int(len(u)), "rag": float(100 * a.loc[u].mean()),
                                      "pot": float(100 * b.loc[u].mean())}
        out["rag_fires"][year] = res
        if ev.has("pot") and ev.has("pot_rag"):
            p, r = ev.cands("pot"), ev.cands("pot_rag")
            mixed = pd.concat([p[p["sample"] < 4], r[r["sample"] < 4].assign(sample=lambda d:
                                                                              d["sample"] + 4)])
            out["mixed_pools"][year] = {
                name: 100 * selection_accuracy(df[df["sample"] < 8], None, "majority")[0]
                for name, df in (("pot_8", p), ("pot_rag_8", r), ("mixed_4_4", mixed))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("verifier_vs_majority", "by_truncation",
                                          "mixed_pools")}, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
