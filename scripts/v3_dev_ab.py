"""v3 dev A/Bs on 2024 (CPU only). Writes artifacts_v3/analysis/dev_ab.json.

Arms (each paired with its training baseline on the same 2024 questions and seeds):
* rag_v2 / rag_v3   - pot_rag_dev_v2 / pot_rag_dev_v3 vs pot_train        (Phase 5)
* transcript        - pot_tx_dev vs pot_train, plus language / diagram groups (Phase 6)
* extend_pot / _cot - pot_ext_dev vs pot_train, cot_ext_dev vs cot_train   (Phase 6b)

Pre-set inclusion rules (PLAN_v3):
* transcript: overall gain > 0 and Hindi / diagram questions not significantly worse
  (their CI upper bound >= 0);
* adaptive budget: gain > 0 and CI lower bound > 0 (PoT arm, the one used in the framework).

usage: .venv/bin/python scripts/v3_dev_ab.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmjee_reasoner.analysis.devab import by_group, changed_rows, compare, pair  # noqa: E402
from mmjee_reasoner.config import CONFIG_DIR, load_config  # noqa: E402
from mmjee_reasoner.pipeline.common import candidates_path, load_candidates  # noqa: E402

ARMS = {
    "rag_v2": ("pot_train", "pot_rag_dev_v2"),
    "rag_v3": ("pot_train", "pot_rag_dev_v3"),
    "transcript": ("pot_train", "pot_tx_dev"),
    "extend_pot": ("pot_train", "pot_ext_dev"),
    "extend_cot": ("cot_train", "cot_ext_dev"),
}


def main() -> None:
    cfg = load_config(CONFIG_DIR / "v3.yaml")
    out: dict = {}
    for name, (base_exp, arm_exp) in ARMS.items():
        if not candidates_path(cfg, arm_exp).exists():
            continue
        base = load_candidates(cfg, base_exp)
        base = base[base["year"] == 2024]
        m = pair(base, load_candidates(cfg, arm_exp))
        out[name] = {"overall": compare(m), "changed": changed_rows(m),
                     "language": by_group(m, "language"),
                     "requires_image": by_group(m, "requires_image")}

    rules = {}
    if "transcript" in out:
        t = out["transcript"]
        worse = [g for grp in ("language", "requires_image") for g, r in t[grp].items()
                 if r["ci95"][1] < 0]
        rules["transcript"] = {"include": t["overall"]["diff"] > 0 and not worse,
                               "significantly_worse_groups": worse}
    if "extend_pot" in out:
        o = out["extend_pot"]["overall"]
        rules["extend"] = {"include": o["diff"] > 0 and o["ci95"][0] > 0}
    out["rules"] = rules

    path = Path(cfg["paths"]["artifacts"]) / "analysis" / "dev_ab.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {path}")
    for name, r in out.items():
        if name == "rules":
            continue
        o, c = r["overall"], r["changed"]
        print(f"{name:11s} {o['acc_base']:5.1f} -> {o['acc_arm']:5.1f}  diff {o['diff']:+.2f} "
              f"[{o['ci95'][0]:+.2f}, {o['ci95'][1]:+.2f}]  changed rows {c['n_changed']} "
              f"({c['acc_base']:.1f} -> {c['acc_arm']:.1f}, +{c['wrong_to_right']}/-"
              f"{c['right_to_wrong']}), tokens/row {c['tokens_per_row_base']:.0f} -> "
              f"{c['tokens_per_row_arm']:.0f}")
        for grp in ("language", "requires_image"):
            print("   ", grp, {g: f"{v['diff']:+.1f} [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}]"
                              for g, v in r[grp].items()})
    print("rules:", rules)


if __name__ == "__main__":
    main()
