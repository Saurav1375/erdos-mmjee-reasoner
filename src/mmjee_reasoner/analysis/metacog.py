"""Paper-style metacognition analysis of the correction loop (v3, base paper Sec. 4.2 / Table 4).

The base paper measures, on solutions that are wrong:

* **EP** (error presence): the model says the solution contains an error;
* **EC** (error correction): the corrected solution is right;
* **EP->EC** chained in one chat, with EC triggered only after a detection.

Our loop is richer: a step-localising critic, a sandboxed corrector, and up to R rounds.
To compare like with like we report round-1-only numbers (one detection + one correction,
like the paper's single pass) next to the final numbers after all rounds. We also report
**false alarms** (correct solutions flagged) and right->wrong flips, which the paper does
not measure.

``pass_at_k`` gives the unbiased pass@k estimator over the independent samples, the
reference point the paper compares EP/EC against.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from mmjee_reasoner.scoring import is_correct


def chain_outcomes(chains: list[dict], questions: pd.DataFrame) -> pd.DataFrame:
    """One row per chain: original / round-1 / final correctness and the round-1 critic flag."""
    qs = questions.set_index("uid")
    rows = []
    for c in chains:
        q = qs.loc[c["uid"]]
        sols, crits = c["solutions"], c["critiques"]
        orig = is_correct(sols[0]["answer"], q)
        flagged = bool(crits and crits[0]["has_error"])
        # Round-1 correction exists iff the first critique flagged an error.
        r1 = is_correct(sols[1]["answer"], q) if flagged and len(sols) > 1 else orig
        final = is_correct(sols[-1]["answer"], q)
        rows.append({"cid": c["cid"], "uid": c["uid"], "orig_correct": orig,
                     "flagged_r1": flagged, "r1_correct": r1, "final_correct": final,
                     "n_flags": sum(bool(x["has_error"]) for x in crits),
                     "rounds": len(crits)})
    return pd.DataFrame(rows)


def _pct(num: int, den: int) -> float | None:
    return 100.0 * num / den if den else None


def metacog_summary(df: pd.DataFrame) -> dict:
    """EP / EC / false-alarm numbers (percent) from :func:`chain_outcomes`."""
    wrong, right = df[~df["orig_correct"]], df[df["orig_correct"]]
    det = wrong[wrong["flagged_r1"]]
    return {
        "n_chains": len(df), "n_wrong": len(wrong), "n_right": len(right),
        "EP": _pct(int(wrong["flagged_r1"].sum()), len(wrong)),
        # Round 1 only (comparable to the paper's single EP->EC pass)
        "EC_r1_of_detected": _pct(int(det["r1_correct"].sum()), len(det)),
        "EP_to_EC_r1": _pct(int(wrong["r1_correct"].sum()), len(wrong)),
        # After all rounds
        "EC_final_of_detected": _pct(int(det["final_correct"].sum()), len(det)),
        "EP_to_EC_final": _pct(int(wrong["final_correct"].sum()), len(wrong)),
        # Costs the paper does not report
        "false_alarm": _pct(int(right["flagged_r1"].sum()), len(right)),
        "right_to_wrong_r1": int((right["flagged_r1"] & ~right["r1_correct"]).sum()),
        "right_to_wrong_final": int((~right["final_correct"]).sum()),
        "wrong_to_right_r1": int(wrong["r1_correct"].sum()),
        "wrong_to_right_final": int(wrong["final_correct"].sum()),
        "acc_orig": 100.0 * df["orig_correct"].mean(),
        "acc_r1": 100.0 * df["r1_correct"].mean(),
        "acc_final": 100.0 * df["final_correct"].mean(),
    }


def pass_at_k(cands: pd.DataFrame, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021) averaged over questions, in percent."""
    vals = []
    for _, g in cands.groupby("uid"):
        n, c = len(g), int(g["correct"].astype(bool).sum())
        if n < k:
            continue
        vals.append(1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0))
    return 100.0 * float(np.mean(vals)) if vals else float("nan")


def critic_gating(same: pd.DataFrame, cross: pd.DataFrame) -> dict:
    """What if a correction were accepted only when BOTH critics flag the original?

    ``same`` / ``cross``: candidates of pot_corr_same / pot_corr_cross (same source chains).
    The gate keeps the cross-critic corrected result where both flagged, else the original.
    """
    c = cross.set_index(["uid", "sample"])
    s = same.set_index(["uid", "sample"]).loc[c.index]
    fs = s["corr_critic_flagged_first"].astype(bool)
    fc = c["corr_critic_flagged_first"].astype(bool)
    orig = c["orig_correct"].astype(bool)

    def outcome(gate: pd.Series, corrected: pd.Series) -> dict:
        fin = corrected.astype(bool).where(gate, orig)
        return {"acc": 100.0 * fin.mean(), "wrong_to_right": int((~orig & fin).sum()),
                "right_to_wrong": int((orig & ~fin).sum()), "n_corrected": int(gate.sum()),
                "false_alarm": 100.0 * gate[orig].mean(), "detect": 100.0 * gate[~orig].mean()}

    return {"original_acc": 100.0 * orig.mean(),
            "same_critic": outcome(fs, s["correct"]),
            "cross_critic": outcome(fc, c["correct"]),
            "both_flag_cross_fix": outcome(fs & fc, c["correct"])}


def load_chains(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())
