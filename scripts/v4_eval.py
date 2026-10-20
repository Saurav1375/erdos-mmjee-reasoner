"""Phase 4 one-shot post-hoc evaluation (W5, V4_PREREG.md). Writes artifacts_v4/eval/v4_results.json.

* selectors - S0 (frozen verifier v3), S1 (majority) and every Phase-4 selector that passed
  its LOYO gate (verifier_v4/selectors_<tag>/report.json), on the frozen v3_full pools of 2025
  and 2026; paired cluster bootstrap and McNemar against S0 and majority
* rag_dev   - 2024 dev A/B with matched seeds: v4_rag_dev_sol / v4_rag_dev_card vs pot_train,
  on all questions and where RAG fires (no test data)
* kb4_2026  - post-hoc v4_pot_kb4 (kb_v4) vs v3_pot on 2026, overall and where retrieval changed

Sections whose inputs do not exist yet are skipped. Run once; do not re-run selectors after
looking at their 2026 numbers.

usage: .venv/bin/python scripts/v4_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from v4_analysis import year_cfg  # noqa: E402

from mmjee_reasoner.config import CONFIG_DIR, load_config  # noqa: E402
from mmjee_reasoner.data.dataset import load_questions  # noqa: E402
from mmjee_reasoner.pipeline.common import candidates_path, load_candidates  # noqa: E402
from mmjee_reasoner.pipeline.evaluate import Evaluator  # noqa: E402
from mmjee_reasoner.scoring.metrics import bootstrap_ci, mcnemar, paired_bootstrap  # noqa: E402
from mmjee_reasoner.verifier.model import selection_accuracy  # noqa: E402
from mmjee_reasoner.verifier.v3 import prepare_pool  # noqa: E402

OUT = ROOT / "artifacts_v4" / "eval" / "v4_results.json"
TAGS = ("base", "extra")   # neural runs: base (v3 pools), extra (+ Phase-4 pools)


def v4_cfg(*extra: str) -> dict:
    files = [CONFIG_DIR / "v3.yaml", *[CONFIG_DIR / e for e in extra], CONFIG_DIR / "v4.yaml"]
    return load_config(*files)


def compare(a: pd.Series, b: pd.Series, clusters: pd.Series) -> dict:
    return {"a": 100 * a.mean(), "b": 100 * b.mean(),
            **paired_bootstrap(a, b, clusters=clusters), "mcnemar": mcnemar(a, b)}


def selector_scores(name: str, tag: str, method: str, pool: pd.DataFrame, cfg4: dict
                    ) -> np.ndarray:
    from mmjee_reasoner.verifier import neural
    from mmjee_reasoner.verifier.v4 import hand_features, predict_ranker, variant_matrix

    prep = prepare_pool(pool).assign(pool="test")
    s_neural = neural.score_pool(cfg4, tag, prep) if name in ("neural", "stacked") else None
    if name == "neural":
        return s_neural
    pipe = joblib.load(neural.neural_dir(cfg4, tag).parent / f"selectors_{tag}" /
                       f"{name}.joblib")
    return predict_ranker(pipe, variant_matrix(name, hand_features(prep), s_neural))


def selectors_section(cfg4: dict) -> dict:
    from mmjee_reasoner.verifier.neural import neural_dir

    reports = {}
    for tag in TAGS:
        p = neural_dir(cfg4, "base").parent / f"selectors_{tag}" / "report.json"
        if p.exists():
            reports[tag] = json.loads(p.read_text())
    out = {"gates": {t: {v: {"pass": r["variants"][v]["gate_pass"],
                             "loyo_16pool": r["variants"][v]["v3_16pool"][
                                 r["variants"][v]["best_method"]],
                             "method": r["variants"][v]["best_method"]}
                         for v in r["variants"]} for t, r in reports.items()}}
    for year in (2025, 2026):
        ev = Evaluator(year_cfg(year))
        pool = ev.pool("v3_full")
        _, s0 = selection_accuracy(pool, ev.verifier_scores(pool), "max_prob")
        _, maj = selection_accuracy(pool, None, "majority")
        res = {"S0_v3": {"acc": 100 * s0.mean(), "ci95": bootstrap_ci(s0, clusters=ev.clusters)},
               "S1_majority": {"acc": 100 * maj.mean()}}
        for tag, r in reports.items():
            for v, tab in r["variants"].items():
                if not tab["gate_pass"]:
                    continue
                key = f"{v}[{tag}]"
                pool_cfg = v4_cfg("v3_test2026.yaml") if year == 2026 else v4_cfg()
                s = selector_scores(v, tag, tab["best_method"], pool, pool_cfg)
                _, ser = selection_accuracy(pool, s, tab["best_method"])
                res[key] = {"acc": 100 * ser.mean(), "method": tab["best_method"],
                            "vs_S0": compare(ser, s0, ev.clusters),
                            "vs_majority": compare(ser, maj, ev.clusters)}
        out[year] = res
    return out


def _fired(c: pd.DataFrame) -> pd.Series:
    return c.groupby("uid")["exemplars"].apply(lambda s: any(len(x) > 0 for x in s))


def rag_dev_section() -> dict | None:
    cfg = v4_cfg("v3_kbdev_v3.yaml", "v4_rag_matched.yaml")
    if not candidates_path(cfg, "v4_rag_dev_sol").exists():
        return None
    q = load_questions(cfg)
    q = q[q["year"] == 2024].set_index("uid")
    base = load_candidates(cfg, "pot_train")
    base = base[base["year"] == 2024].groupby("uid")["correct"].mean()
    arms = {"sol": load_candidates(cfg, "v4_rag_dev_sol")}
    if candidates_path(cfg, "v4_rag_dev_card").exists():
        arms["card"] = load_candidates(cfg, "v4_rag_dev_card")
    acc = {k: c.groupby("uid")["correct"].mean() for k, c in arms.items()}
    fired = _fired(arms["sol"])
    out = {"fired": int(fired.sum()), "questions": int(len(fired))}
    pairs = [(k, "pot") for k in acc] + ([("card", "sol")] if "card" in acc else [])
    for a, b in pairs:
        sa, sb = acc[a], acc[b] if b in acc else base
        out[f"{a}_vs_{b}"] = {"all": compare(sa, sb, q["twin_key"]),
                              "fired": compare(sa[fired], sb.reindex(sa[fired].index),
                                               q["twin_key"])}
    return out


def kb4_section() -> dict | None:
    cfg = v4_cfg("v3_test2026.yaml", "v3_final.yaml", "v4_kb.yaml")
    if not candidates_path(cfg, "v4_pot_kb4").exists():
        return None
    q = load_questions(cfg, "test").set_index("uid")
    new, old = load_candidates(cfg, "v4_pot_kb4"), load_candidates(cfg, "v3_pot")
    ex_new = new.groupby("uid")["exemplars"].first().map(tuple)
    ex_old = old.groupby("uid")["exemplars"].first().map(tuple)
    changed = ex_new != ex_old.reindex(ex_new.index)
    a, b = new.groupby("uid")["correct"].mean(), old.groupby("uid")["correct"].mean()
    return {"fired_kb4": int((ex_new.map(len) > 0).sum()),
            "fired_kb3": int((ex_old.map(len) > 0).sum()),
            "changed": int(changed.sum()), "questions": int(len(changed)),
            "all": compare(a, b, q["twin_key"]),
            "changed_only": compare(a[changed], b.reindex(a[changed].index), q["twin_key"])}


def main() -> None:
    cfg4 = v4_cfg()
    out = {"note": "post-hoc (V4_PREREG.md); the v3 2026 clean result stays the headline",
           "selectors": selectors_section(cfg4), "rag_dev": rag_dev_section(),
           "kb4_2026": kb4_section()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
