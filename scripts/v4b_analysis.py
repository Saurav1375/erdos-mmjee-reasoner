"""Phase 4b analysis (V4B_PREREG.md): track C on existing results and the track-B dev gate.

Writes artifacts_v4b/analysis/v4b.json. Reads artifacts_v4b (a copy of artifacts_v4; Phase 4 is
never written). Sections:

* decomposition - per test year: gain of the v3 framework over matched SC split into the pool
  gain (majority vote on the v3 pool - matched SC) and the selection gain (v3 pick - majority on
  the v3 pool), paired cluster bootstrap. FROZEN v3 verifier and existing pools only.
* discrimination - candidate AUC of the frozen verifier over all candidates vs the mean AUC
  inside pools that hold both right and wrong answers (what selection needs).
* errors_2026 - the W1 error budget (correct / selector miss / never in the pool), copied.
* dev_b (only once v4b_tx_diag_dev exists) - 2024 dev: majority@8 of pot_tx_dev s0-s3 +
  diagram-first s4-s7 vs pot_tx_dev s0-s7 (the gate), plus matched per-sample accuracy, oracle@8,
  distinct answers, tokens and truncation, overall and by requires_image.

usage: .venv/bin/python scripts/v4b_analysis.py
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
from mmjee_reasoner.pipeline.common import candidates_path, load_candidates  # noqa: E402
from mmjee_reasoner.pipeline.evaluate import Evaluator  # noqa: E402
from mmjee_reasoner.scoring.metrics import paired_bootstrap  # noqa: E402
from mmjee_reasoner.verifier.features import answer_key  # noqa: E402
from mmjee_reasoner.verifier.model import selection_accuracy  # noqa: E402

ART = ROOT / "artifacts_v4b"
OUT = ART / "analysis" / "v4b.json"
PER_Q = {2025: "per_question.v3.csv", 2026: "per_question.y2026.v3.csv"}
SC_ROW = "Self-consistency (matched compute)"
FULL_ROW = "v3 framework (verifier v3)"


def year_cfg(year: int) -> dict:
    """Frozen v3 evaluation config, reading the copies in artifacts_v4b (verifier v3)."""
    files = [CONFIG_DIR / "v3.yaml"]
    if year == 2026:
        files.append(CONFIG_DIR / "v3_test2026.yaml")
    files += [CONFIG_DIR / "v3_final.yaml", CONFIG_DIR / "v3_eval.yaml"]
    cfg = load_config(*files)
    cfg["paths"]["artifacts"] = str(ART)
    return cfg


def within_pool_auc(pool: pd.DataFrame, scores: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    s = pd.Series(scores, index=pool.index)
    aucs = [roc_auc_score(g["correct"].astype(int), s.loc[g.index])
            for _, g in pool.groupby("uid") if 0 < g["correct"].mean() < 1]
    return {"mean": float(np.mean(aucs)), "n_mixed_pools": len(aucs)}


def decomposition() -> tuple[dict, dict]:
    from sklearn.metrics import roc_auc_score

    dec, disc = {}, {}
    res_dir = ART / "eval"                 # = results/v3 in the repo
    for year in (2025, 2026):
        cfg = year_cfg(year)
        ev = Evaluator(cfg)
        pool = ev.pool("v3_full")
        scores = ev.verifier_scores(pool)
        method = ev.verifiers()["primary"]["selection"]
        _, ver = selection_accuracy(pool, scores, method)
        _, maj = selection_accuracy(pool, None, "majority")
        pq = pd.read_csv(res_dir / PER_Q[year]).set_index("uid")
        sc = pq[SC_ROW].loc[ver.index]
        assert np.allclose(pq[FULL_ROW].loc[ver.index], ver), "v3 pick does not reproduce"
        cl = ev.clusters
        dec[year] = {
            "matched_sc": float(100 * sc.mean()), "majority_v3_pool": float(100 * maj.mean()),
            "v3_pick": float(100 * ver.mean()),
            "pool_gain": paired_bootstrap(maj, sc, clusters=cl),
            "selection_gain": paired_bootstrap(ver, maj, clusters=cl),
            "total_gain": paired_bootstrap(ver, sc, clusters=cl)}
        disc[year] = {"auc_all_candidates": float(roc_auc_score(pool["correct"].astype(int),
                                                                scores)),
                      "within_pool": within_pool_auc(pool, scores)}
    return dec, disc


def pool_view(c: pd.DataFrame) -> pd.Series:
    return c.groupby("uid").apply(
        lambda g: len({answer_key(p, t) for p, t in zip(g["pred"].fillna("").astype(str),
                                                         g["question_type"])}))


def dev_b() -> dict | None:
    cfg = load_config(CONFIG_DIR / "v3.yaml", CONFIG_DIR / "v4.yaml", CONFIG_DIR / "v4b.yaml")
    cfg["paths"]["artifacts"] = str(ART)
    if not candidates_path(cfg, "v4b_tx_diag_dev").exists():
        return None
    ref = load_candidates(cfg, "pot_tx_dev")
    diag = load_candidates(cfg, "v4b_tx_diag_dev")
    assert set(diag["sample"]) == {4, 5, 6, 7} and set(diag["uid"]) == set(ref["uid"])
    multi = pd.concat([ref[ref["sample"] < 4], diag], ignore_index=True)
    qs = ref.drop_duplicates("uid").set_index("uid")
    clusters = qs["twin_key"]
    img = qs["requires_image"].fillna(False).astype(bool)

    def stats(c: pd.DataFrame) -> pd.DataFrame:
        _, maj = selection_accuracy(c, None, "majority")
        _, orc = selection_accuracy(c, None, "oracle")
        g = c.groupby("uid")
        return pd.DataFrame({"majority": maj, "oracle": orc, "distinct": pool_view(c),
                             "acc": g["correct"].mean(),
                             "tokens": g["completion_tokens"].mean(),
                             "truncated": g["finish"].apply(lambda s: (s == "length").mean())})

    r, m = stats(ref), stats(multi)
    a4, d4 = stats(ref[ref["sample"] >= 4]), stats(diag)    # matched samples s4-s7

    def block(mask: pd.Series) -> dict:
        u = mask[mask].index
        return {"n": len(u),
                "majority8": {"reference": float(100 * r.loc[u, "majority"].mean()),
                              "multi_view": float(100 * m.loc[u, "majority"].mean()),
                              "diff": paired_bootstrap(m.loc[u, "majority"],
                                                       r.loc[u, "majority"],
                                                       clusters=clusters)},
                "oracle8": {"reference": float(100 * r.loc[u, "oracle"].mean()),
                            "multi_view": float(100 * m.loc[u, "oracle"].mean())},
                "distinct_answers": {"reference": float(r.loc[u, "distinct"].mean()),
                                     "multi_view": float(m.loc[u, "distinct"].mean())},
                "per_sample_s4_s7": {"current": float(100 * a4.loc[u, "acc"].mean()),
                                     "diagram": float(100 * d4.loc[u, "acc"].mean()),
                                     "diff": paired_bootstrap(d4.loc[u, "acc"],
                                                              a4.loc[u, "acc"],
                                                              clusters=clusters)},
                "tokens_s4_s7": {"current": float(a4.loc[u, "tokens"].mean()),
                                 "diagram": float(d4.loc[u, "tokens"].mean())},
                "truncated_s4_s7": {"current": float(100 * a4.loc[u, "truncated"].mean()),
                                    "diagram": float(100 * d4.loc[u, "truncated"].mean())}}

    out = {"all": block(img | ~img), "diagram_questions": block(img),
           "text_questions": block(~img)}
    d = out["all"]["majority8"]["diff"]
    out["gate_pass"] = bool(d["diff"] > 0 and d["ci95"][0] > 0)
    return out


def main() -> None:
    out: dict = {"note": "Phase 4b (V4B_PREREG.md); frozen v3 verifier and existing pools for "
                         "decomposition/discrimination; 2025/2026 already seen"}
    out["decomposition"], out["discrimination"] = decomposition()
    w1 = json.loads((ART / "analysis" / "w1.json").read_text())
    out["errors_2026"] = w1["errors_2026"]
    b = dev_b()
    if b is not None:
        out["dev_b"] = b
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    for y, d in out["decomposition"].items():
        print(y, {k: round(v["diff"], 1) if isinstance(v, dict) else round(v, 1)
                  for k, v in d.items()})
    print("discrimination", json.dumps(out["discrimination"]))
    if b is not None:
        print("dev_b gate", b["gate_pass"], json.dumps(b["all"]["majority8"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
