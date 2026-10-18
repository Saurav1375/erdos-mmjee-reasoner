"""v3 analyses on existing generations (CPU only). Writes artifacts_v3/analysis/analysis.json.

Sections run only when their inputs exist:
* metacog   - paper-style EP / EC / false alarms per correction experiment (2025 test)
* pass_at_k - pass@1/3/5 of the sampled rows (paper Table 4 reference)
* gating    - accept a correction only when both critics flag (pot_corr_same vs _cross)
* probes    - memorisation probes by year (answer-only accuracy, completion overlap)

usage: .venv/bin/python scripts/v3_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmjee_reasoner.analysis.metacog import (  # noqa: E402
    chain_outcomes,
    critic_gating,
    load_chains,
    metacog_summary,
    pass_at_k,
)
from mmjee_reasoner.analysis.probes import summarise_by_year  # noqa: E402
from mmjee_reasoner.config import CONFIG_DIR, load_config  # noqa: E402
from mmjee_reasoner.data.dataset import load_questions  # noqa: E402
from mmjee_reasoner.pipeline.common import candidates_path, load_candidates  # noqa: E402

CORRECTION_EXPS = ["pot_corr_same", "pot_corr_cross", "full", "q25_corr_same", "q25_corr_cross"]
SAMPLED_EXPS = ["cot", "pot", "pot_rag", "q25_cot", "q25_pot", "repro_qwen25"]


def main() -> None:
    cfg = load_config(CONFIG_DIR / "v3.yaml")
    qs = load_questions(cfg)
    gen = Path(cfg["paths"]["artifacts"]) / "generations"
    out: dict = {"metacog": {}, "pass_at_k": {}}

    for e in CORRECTION_EXPS:
        p = gen / e / "chains.json"
        if p.exists():
            out["metacog"][e] = metacog_summary(chain_outcomes(load_chains(p), qs))
    for e in SAMPLED_EXPS:
        if candidates_path(cfg, e).exists():
            c = load_candidates(cfg, e)
            out["pass_at_k"][e] = {f"pass@{k}": pass_at_k(c, k) for k in (1, 3, 5)}
    if all(candidates_path(cfg, e).exists() for e in ("pot_corr_same", "pot_corr_cross")):
        out["gating"] = critic_gating(load_candidates(cfg, "pot_corr_same"),
                                      load_candidates(cfg, "pot_corr_cross"))

    probes: dict = {}
    for e in ("probe_answer_q3", "probe_answer_q25"):
        if candidates_path(cfg, e).exists():
            c = load_candidates(cfg, e)
            c["correct"] = c["correct"].astype(float) * 100
            c["en"] = c["language"] == "English"
            probes[e] = {
                "all": summarise_by_year(c, ["correct"]).reset_index().to_dict("records"),
                "english": summarise_by_year(c[c["en"]], ["correct"]).reset_index()
                .to_dict("records")}
    for role in ("solver", "repro"):
        p = Path(cfg["paths"]["artifacts"]) / "probes" / f"completion_{role}.jsonl"
        if p.exists():
            c = pd.read_json(p, lines=True)
            probes[f"completion_{role}"] = summarise_by_year(
                c, ["rouge_l", "ngram5"]).reset_index().to_dict("records")
    if probes:
        out["probes"] = probes

    path = Path(cfg["paths"]["artifacts"]) / "analysis" / "analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {path}")
    for e, s in out["metacog"].items():
        print(f"{e:16s} EP {s['EP']:.1f}  EC_r1|det {s['EC_r1_of_detected']:.1f}  "
              f"EP->EC_r1 {s['EP_to_EC_r1']:.1f}  EC_final|det {s['EC_final_of_detected']:.1f}  "
              f"FA {s['false_alarm']:.1f}")


if __name__ == "__main__":
    main()
