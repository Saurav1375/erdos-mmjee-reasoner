"""v3 final report: metrics, one figure and the Markdown report, regenerated from artifacts only.

usage: .venv/bin/python scripts/final_report_v3.py
   (after `evaluate` with configs/v3_eval.yaml on 2026 and 2025, and configs/v3_eval_v2fw.yaml on 2026)
inputs : artifacts_v3/eval/results{.y2026.v3,.y2026.v2fw,.v3,.q25}.json + per_question CSVs,
         artifacts/eval/results.json + per_question.csv (frozen v2 run on 2025, read only),
         artifacts_v3/analysis/{analysis,dev_ab,rag_sweep_loyo}.json, artifacts_v3/verifier_v3/report.json,
         artifacts_v3/generations/*/{candidates,chains}*.json[l]
outputs: report/final_v3/metrics.json, report/final_v3/figures/table1_2026.{pdf,png},
         report/FINAL_REPORT_v3.md
No number in the Markdown report is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmjee_reasoner.analysis.metacog import (  # noqa: E402
    chain_outcomes,
    load_chains,
    metacog_summary,
    pass_at_k,
)
from mmjee_reasoner.config import CONFIG_DIR, load_config  # noqa: E402
from mmjee_reasoner.data.dataset import load_questions  # noqa: E402
from mmjee_reasoner.pipeline.common import candidates_path, load_candidates  # noqa: E402
from mmjee_reasoner.scoring.metrics import mcnemar, paired_bootstrap  # noqa: E402

V2_ART, ART = ROOT / "artifacts", ROOT / "artifacts_v3"
EVAL, GEN = ART / "eval", ART / "generations"
OUT = ROOT / "report" / "final_v3"
FIG = OUT / "figures"
MD = ROOT / "report" / "FINAL_REPORT_v3.md"

# Same validated categorical slots as the v2 report figures (scripts/final_report.py).
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

V3_FULL = "v3 framework (verifier v3)"
V2_FULL = "Full framework"
C = {y: load_config(*[CONFIG_DIR / f for f in files]) for y, files in {
    2026: ("v3.yaml", "v3_test2026.yaml", "v3_final.yaml"),
    2025: ("v3.yaml", "v3_final.yaml")}.items()}
TAG = {2026: ".y2026", 2025: ""}


# ----------------------------------------------------------------------------- loading
def results(name: str, v2: bool = False) -> dict:
    return json.loads(((V2_ART if v2 else ART) / "eval" / f"results{name}.json").read_text())


def per_q(name: str, v2: bool = False) -> pd.DataFrame:
    return pd.read_csv((V2_ART if v2 else ART) / "eval" / f"per_question{name}.csv",
                       index_col="uid")


def questions(year: int) -> pd.DataFrame:
    return load_questions(C[year], "test")


def compare(a: pd.Series, b: pd.Series, year: int) -> dict:
    clusters = questions(year).set_index("uid")["twin_key"]
    boot = C[year]["eval"]["bootstrap"]
    return {"bootstrap": paired_bootstrap(a, b, boot, C[year]["seed"], clusters),
            "mcnemar": mcnemar(a, b)}


def row(res: dict, name: str) -> dict:
    return next(r for r in res["table1"] if r["row"] == name)


# ----------------------------------------------------------------------------- analyses
def metacog(year: int, exp: str) -> dict | None:
    p = GEN / exp / f"chains{TAG[year]}.json"
    if not p.exists():
        return None
    return metacog_summary(chain_outcomes(load_chains(p), questions(year)))


def passk(year: int, exp: str) -> dict | None:
    if not candidates_path(C[year], exp).exists():
        return None
    c = load_candidates(C[year], exp)
    c = c[c["uid"].isin(set(questions(year)["uid"]))]
    return {f"pass@{k}": pass_at_k(c, k) for k in (1, 3, 5)}


def shift_stats(df: pd.DataFrame) -> dict:
    """Length / fallback statistics the verifier's features depend on (V3_FREEZE limitation 1)."""
    cut = df["finish"] == "length"
    return {"n": len(df), "acc": 100 * df["correct"].mean(),
            "truncated": 100 * cut.mean(), "fallback": 100 * df["used_fallback"].mean(),
            "tokens": float(df["completion_tokens"].mean()),
            "acc_truncated": 100 * df.loc[cut, "correct"].mean() if cut.any() else None,
            "acc_complete": 100 * df.loc[~cut, "correct"].mean() if (~cut).any() else None}


def collect() -> dict:
    m: dict = {"tables": {}, "comparisons": {}, "metacog": {}, "pass_at_k": {}, "shift": {}}
    tabs = {"2026_v3": results(".y2026.v3"), "2026_v2fw": results(".y2026.v2fw"),
            "2025_v3": results(".v3"), "2025_v2": results("", v2=True),
            "2025_q25": results(".q25")}
    m["tables"] = tabs
    pq = {"2026_v3": per_q(".y2026.v3"), "2026_v2fw": per_q(".y2026.v2fw"),
          "2025_v3": per_q(".v3"), "2025_v2": per_q("", v2=True)}
    pairs = [  # (name, year, (table, row), (table, row))
        ("2026: v3 framework vs v2 framework", 2026, ("2026_v3", V3_FULL), ("2026_v2fw", V2_FULL)),
        ("2026: v3 PoT vs v2 PoT", 2026, ("2026_v3", "+ Code sandbox + RAG v3 + transcript (v3 PoT)"),
         ("2026_v2fw", "+ Code sandbox")),
        ("2026: CoT adaptive budget vs CoT", 2026,
         ("2026_v3", "Baseline (single-pass CoT, adaptive budget)"),
         ("2026_v2fw", "Baseline (single-pass CoT)")),
        ("2026: gated vs ungated correction (v3)", 2026, ("2026_v3", "+ Correction (gated)"),
         ("2026_v3", "+ Correction (cross-model critic)")),
        ("2026: verifier v3 pick vs v3 PoT mean", 2026, ("2026_v3", V3_FULL),
         ("2026_v3", "+ Code sandbox + RAG v3 + transcript (v3 PoT)")),
        ("2025 (post-hoc): v3 framework vs v2 framework", 2025, ("2025_v3", V3_FULL),
         ("2025_v2", V2_FULL)),
        ("2025 (post-hoc): v3 PoT vs v2 PoT", 2025,
         ("2025_v3", "+ Code sandbox + RAG v3 + transcript (v3 PoT)"), ("2025_v2", "+ Code sandbox")),
    ]
    for name, year, (ta, ra), (tb, rb) in pairs:
        if ra in pq[ta] and rb in pq[tb]:
            m["comparisons"][name] = compare(pq[ta][ra], pq[tb][rb], year)
    for year, exps in {2026: ("pot_corr_cross", "full", "v3_full"), 2025: ("v3_full",)}.items():
        for e in exps:
            s = metacog(year, e)
            if s:
                m["metacog"][f"{year} {e}"] = s
    v2a = json.loads((ART / "analysis" / "analysis.json").read_text())
    for e in ("pot_corr_cross", "full"):
        m["metacog"][f"2025 {e} (v2)"] = v2a["metacog"][e]
    for year, exps in {2026: ("cot", "cot_ext", "pot", "pot_rag", "v3_pot"),
                       2025: ("cot_ext", "v3_pot")}.items():
        for e in exps:
            s = passk(year, e)
            if s:
                m["pass_at_k"][f"{year} {e}"] = s
    for e in ("cot", "pot", "pot_rag"):
        m["pass_at_k"][f"2025 {e} (v2)"] = v2a["pass_at_k"][e]
    m["shift"]["train 2019-2024 pot_train"] = shift_stats(load_candidates(C[2025], "pot_train"))
    for year in (2026, 2025):
        for e in ("pot", "v3_pot"):
            if candidates_path(C[year], e).exists():
                m["shift"][f"{year} {e}"] = shift_stats(load_candidates(C[year], e))
    m["probes"] = v2a.get("probes", {})
    m["dev_ab"] = json.loads((ART / "analysis" / "dev_ab.json").read_text())
    m["rag_sweep"] = json.loads((ART / "analysis" / "rag_sweep_loyo.json").read_text())
    vr = json.loads((ART / "verifier_v3" / "report.json").read_text())
    m["verifier_v3"] = {k: vr[k] for k in ("primary", "baselines_16pool", "primary_beats_majority_16pool")}
    m["verifier_v3"]["gate"] = {k: vr["gate"][k] for k in ("delta", "chosen", "nested_loyo")}
    m["primary"] = {
        "v3": vr["primary"],
        "v2": json.loads((ART / "verifier" / "report.json").read_text())["primary"]}
    m["verifier_v3"]["primary_loyo"] = next(
        c for c in vr["configs"] if (c["model"], c["feature_set"]) ==
        (vr["primary"]["model"], vr["primary"]["feature_set"]))
    return m


# ----------------------------------------------------------------------------- figure
def figure(m: dict) -> None:
    """2026 clean test: v2 vs v3 framework, matching rows, mean with 95% cluster-bootstrap CI."""
    t3, t2 = m["tables"]["2026_v3"], m["tables"]["2026_v2fw"]
    rows = [("Baseline CoT", "Baseline (single-pass CoT)",
             "Baseline (single-pass CoT, adaptive budget)"),
            ("Self-consistency (matched)", "Self-consistency (matched compute)",
             "Self-consistency (matched compute)"),
            ("PoT solver", "+ Code sandbox", "+ Code sandbox + RAG v3 + transcript (v3 PoT)"),
            ("+ Correction (cross critic)", "+ Agentic correction (cross-model critic)",
             "+ Correction (cross-model critic)"),
            ("Full framework", V2_FULL, V3_FULL)]
    fig, ax = plt.subplots(figsize=(5.6, 2.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (label, r2, r3) in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, name, tab, rname, color in ((0.14, "v2 framework", t2, r2, ORANGE),
                                            (-0.14, "v3 framework", t3, r3, BLUE)):
            r = row(tab, rname)
            lo, hi = r["ci95"]
            ax.plot([lo, hi], [y + dy, y + dy], color=color, lw=2, solid_capstyle="round")
            ax.plot(r["overall"], y + dy, "o", ms=8, color=color, mec=SURFACE, mew=2,
                    label=name if i == 0 else None)
            ax.text(hi + 0.8, y + dy, f"{r['overall']:.1f}", va="center", color=TEXT_2,
                    fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], color=TEXT)
    ax.set_xlabel("Accuracy on JEE Advanced 2026 (%), 95% cluster-bootstrap CI", color=TEXT_2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=7)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"table1_2026.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- markdown
def f1(x) -> str:
    return "–" if x is None else f"{x:.1f}"


def ci(c) -> str:
    return f"{c[0]:.1f}–{c[1]:.1f}"


def dci(b: dict) -> str:
    return f"{b['diff']:+.1f} [{b['ci95'][0]:+.1f}, {b['ci95'][1]:+.1f}]"


def table1_md(res: dict) -> str:
    out = ["| Row | Overall | Numerical | 95% CI | vs baseline | Tokens / q | Calls / q |",
           "|---|---|---|---|---|---|---|"]
    for r in res["table1"]:
        if not r.get("available"):
            out.append(f"| {r['row']} | n/a | | | | | |")
            continue
        name = r["row"]
        if r["kind"] == "sc_matched":
            name += f" (n = {r['n_samples']}" + (f"; needs {r['n_needed']}, capped" if
                                                  r.get("insufficient") else "") + ")"
        vb = dci(r["vs_baseline"]) if "vs_baseline" in r else "–"
        cost = r.get("cost") or {}
        out.append(f"| {name} | **{r['overall']:.1f}** | {r['numerical']:.1f} | {ci(r['ci95'])} | "
                   f"{vb} | {cost.get('tokens', 0) / 1000:.1f}k | {cost.get('calls', 0):.1f} |")
    return "\n".join(out)


def headline(res: dict, full: str) -> str:
    b, mc = res.get("full_vs_sc_matched"), res.get("full_vs_sc_matched_mcnemar")
    if not b:
        return ""
    return (f"Framework vs matched self-consistency: **{dci(b)}** points (cluster bootstrap), "
            f"McNemar p = {mc['p']:.3f} ({mc['a_only']} questions only the framework gets right, "
            f"{mc['b_only']} only self-consistency).")


def breakdown_md(pairs: list[tuple[str, dict]]) -> str:
    keys = [("language", "English"), ("language", "Hindi"), ("requires_image", "False"),
            ("requires_image", "True"), ("subject", "Physics"), ("subject", "Chemistry"),
            ("subject", "Mathematics"), ("question_type", "MCQ-Single"),
            ("question_type", "MCQ-Multiple"), ("question_type", "Numerical"),
            ("question_type", "Matching")]
    head = "| Group | " + " | ".join(n for n, _ in pairs) + " |"
    out = [head, "|---" * (len(pairs) + 1) + "|"]
    for g, v in keys:
        label = {"False": "Text only", "True": "Needs image"}.get(v, v)
        vals = [f1(r["breakdown"].get(g, {}).get(v)) for _, r in pairs]
        out.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def correction_md(m: dict) -> str:
    out = ["| Run | Chains | Acc. before → after | EP (detect) | EC r1 of detected | "
           "EP→EC final | False alarm | Wrong→right | Right→wrong |", "|---" * 9 + "|"]
    for k, s in m["metacog"].items():
        out.append(f"| {k} | {s['n_chains']} | {f1(s.get('acc_orig'))} → {f1(s.get('acc_final'))} | "
                   f"{f1(s['EP'])} | {f1(s['EC_r1_of_detected'])} | {f1(s['EP_to_EC_final'])} | "
                   f"{f1(s['false_alarm'])} | {s['wrong_to_right_final']} | "
                   f"{s['right_to_wrong_final']} |")
    return "\n".join(out)


def gate_md(m: dict) -> str:
    out = ["| Test set | Chains | Original | Corrected (ungated) | Gated | Accepted | "
           "Wrong→right (ungated → gated) | Right→wrong (ungated → gated) |", "|---" * 8 + "|"]
    g = m["verifier_v3"]["gate"]["chosen"]
    rows = [("Training, LOYO (selection of δ)", g)]
    for key, label in (("2026_v3", "2026 (clean)"), ("2025_v3", "2025 (post-hoc)")):
        r = next((x for x in m["tables"][key]["table1"] if x.get("kind") == "gated"
                  and x.get("available")), None)
        if r:
            rows.append((label, r["gate"]))
    for label, s in rows:
        out.append(f"| {label} | {s['n_chains']} | {s['acc_original']:.1f} | "
                   f"{s['acc_corrected']:.1f} | **{s['acc_gated']:.1f}** | {s['accepted']} | "
                   f"{s['wrong_to_right_ungated']} → {s['wrong_to_right_gated']} | "
                   f"{s['right_to_wrong_ungated']} → {s['right_to_wrong_gated']} |")
    return "\n".join(out)


def verifier_md(m: dict) -> str:
    """Primary verifier of each framework on its own 16-answer test pool vs pool baselines."""
    out = ["| Test pool | Primary verifier | AUC | Pick (max prob) | Weighted vote | Majority | "
           "Random | Oracle |", "|---|---|---|---|---|---|---|---|"]
    for key, full, exp, label in (("2026_v3", V3_FULL, "v3_full", "2026 v3"),
                                  ("2026_v2fw", V2_FULL, "full", "2026 v2fw"),
                                  ("2025_v3", V3_FULL, "v3_full", "2025 v3"),
                                  ("2025_v2", V2_FULL, "full", "2025 v2")):
        r = row(m["tables"][key], full)
        prim = m["primary"]["v3" if key.endswith("v3") else "v2"]
        name = f"{prim['model']}/{prim['feature_set']}"
        auc = m["tables"][key].get("verifier_test", {}).get(exp, {}).get(name, {}).get("auc")
        out.append(f"| {label} `{exp}` | {name} | {'–' if auc is None else f'{auc:.3f}'} | "
                   f"**{r['sel_max_prob']:.1f}** | {r['sel_weighted_vote']:.1f} | "
                   f"{r['sel_majority']:.1f} | {r['sel_random_expected']:.1f} | "
                   f"{r['sel_oracle']:.1f} |")
    return "\n".join(out)


def shift_md(m: dict) -> str:
    out = ["| Solutions | Rows | Accuracy | Cut off (`finish = length`) | CoT fallback | "
           "Tokens / sample | Acc. when cut off | Acc. when complete |", "|---" * 8 + "|"]
    for k, s in m["shift"].items():
        out.append(f"| {k} | {s['n']} | {s['acc']:.1f} | {s['truncated']:.1f} | "
                   f"{s['fallback']:.1f} | {s['tokens']:.0f} | {f1(s['acc_truncated'])} | "
                   f"{f1(s['acc_complete'])} |")
    return "\n".join(out)


def passk_md(m: dict) -> str:
    out = ["| Run | pass@1 | pass@3 | pass@5 |", "|---|---|---|---|"]
    for k, s in m["pass_at_k"].items():
        out.append(f"| {k} | {s['pass@1']:.1f} | {s['pass@3']:.1f} | {s['pass@5']:.1f} |")
    return "\n".join(out)


def comparisons_md(m: dict) -> str:
    out = ["| Comparison | Δ points [95% CI] | P(Δ ≤ 0) | McNemar (a only / b only, p) |",
           "|---|---|---|---|"]
    for k, c in m["comparisons"].items():
        b, mc = c["bootstrap"], c["mcnemar"]
        out.append(f"| {k} | {dci(b)} | {b['p_le_0']:.3f} | {mc['a_only']} / {mc['b_only']}, "
                   f"p = {mc['p']:.3f} |")
    return "\n".join(out)


def probes_md(m: dict) -> str:
    p = m["probes"]
    if "probe_answer_q3" not in p:
        return "_Probe outputs missing._"
    q3 = {r["year"]: r["correct"] for r in p["probe_answer_q3"]["all"]}
    q25 = {r["year"]: r["correct"] for r in p.get("probe_answer_q25", {}).get("all", [])}
    out = ["| Year | " + " | ".join(str(y) for y in sorted(q3)) + " |",
           "|---" * (len(q3) + 1) + "|",
           "| Qwen3-VL-8B | " + " | ".join(f1(q3[y]) for y in sorted(q3)) + " |"]
    if q25:
        out.append("| Qwen2.5-VL-7B | " + " | ".join(f1(q25.get(y)) for y in sorted(q3)) + " |")
    return "\n".join(out)


def dev_ab_md(m: dict) -> str:
    d = m["dev_ab"]
    names = {"transcript": "Transcript block (PoT)", "extend_pot": "Adaptive budget (PoT)",
             "extend_cot": "Adaptive budget (CoT)", "rag_v2": "RAG v2 recipe",
             "rag_v3": "RAG v3 (hybrid + JEEBench)"}
    out = ["| Arm (2024 dev, 204 questions) | Base | Arm | Δ [95% CI] | Rows changed |",
           "|---|---|---|---|---|"]
    for k, label in names.items():
        o, ch = d[k]["overall"], d[k]["changed"]
        out.append(f"| {label} | {o['acc_base']:.1f} | {o['acc_arm']:.1f} | {dci(o)} | "
                   f"{ch['n_changed']} / {o['n_rows']} |")
    return "\n".join(out)


def q25_md(m: dict) -> str:
    return table1_md(m["tables"]["2025_q25"]) + "\n\n" + headline(m["tables"]["2025_q25"], "")


def verdict(b: dict) -> str:
    lo, hi = b["ci95"]
    return ("significant gain" if lo > 0 else "significant drop" if hi < 0
            else "no significant difference")


def findings(m: dict) -> str:
    t26, t26v2, t25 = m["tables"]["2026_v3"], m["tables"]["2026_v2fw"], m["tables"]["2025_v3"]
    cmp_ = m["comparisons"]
    out = ["## Key findings", ""]
    for res, full, label in ((t26, V3_FULL, "2026 (clean), v3 framework"),
                             (t26v2, V2_FULL, "2026 (clean), v2 framework"),
                             (t25, V3_FULL, "2025 (post-hoc), v3 framework")):
        r, base = row(res, full), res["table1"][0]
        sc = next(x for x in res["table1"] if x.get("kind") == "sc_matched")
        b = res["full_vs_sc_matched"]
        out.append(f"- **{label}:** {r['overall']:.1f}% (CI {ci(r['ci95'])}), vs baseline CoT "
                   f"{base['overall']:.1f}% ({dci(r['vs_baseline'])}, "
                   f"{verdict(r['vs_baseline'])}). Vs self-consistency at matched compute "
                   f"({sc['overall']:.1f}%): {dci(b)}, {verdict(b)}.")
    for k in ("2026: v3 framework vs v2 framework", "2025 (post-hoc): v3 framework vs v2 framework",
              "2026: v3 PoT vs v2 PoT", "2026: gated vs ungated correction (v3)"):
        if k in cmp_:
            b = cmp_[k]["bootstrap"]
            out.append(f"- **{k}:** {dci(b)} points, {verdict(b)} "
                       f"(McNemar p = {cmp_[k]['mcnemar']['p']:.3f}).")
    for key, full, label in (("2026_v3", V3_FULL, "2026 v3"), ("2026_v2fw", V2_FULL, "2026 v2"),
                             ("2025_v3", V3_FULL, "2025 v3")):
        r = row(m["tables"][key], full)
        out.append(f"- **Verifier vs majority vote over the same pool, {label}:** "
                   f"{r['sel_max_prob']:.1f} vs {r['sel_majority']:.1f} "
                   f"({r['sel_max_prob'] - r['sel_majority']:+.1f}; oracle {r['sel_oracle']:.1f}).")
    return "\n".join(out)


def write_md(m: dict) -> None:
    t26, t26v2, t25 = m["tables"]["2026_v3"], m["tables"]["2026_v2fw"], m["tables"]["2025_v3"]
    t25v2 = m["tables"]["2025_v2"]
    cmp_ = m["comparisons"]
    vv = m["verifier_v3"]
    pl = vv["primary_loyo"]
    text = TEMPLATE.format(
        findings=findings(m),
        t26=table1_md(t26), h26=headline(t26, V3_FULL),
        t26v2=table1_md(t26v2), h26v2=headline(t26v2, V2_FULL),
        t25=table1_md(t25), h25=headline(t25, V3_FULL),
        t25v2_full=f"{row(t25v2, V2_FULL)['overall']:.1f}",
        cmp=comparisons_md(m),
        bd=breakdown_md([("v2 fw 2026", row(t26v2, V2_FULL)), ("v3 fw 2026", row(t26, V3_FULL)),
                         ("v2 fw 2025", row(t25v2, V2_FULL)), ("v3 fw 2025", row(t25, V3_FULL))]),
        corr=correction_md(m), gate=gate_md(m), ver=verifier_md(m), shift=shift_md(m),
        passk=passk_md(m), probes=probes_md(m), devab=dev_ab_md(m), q25=q25_md(m),
        v_prim=f"{vv['primary']['model']} / {vv['primary']['feature_set']} / "
               f"{vv['primary']['selection']}",
        v_loyo=f"{pl['sel_16pool']['max_prob']:.1f}" if "max_prob" in pl.get("sel_16pool", {})
        else "–",
        v_maj=f"{vv['baselines_16pool']['majority']:.1f}",
        v_orc=f"{vv['baselines_16pool']['oracle']:.1f}",
        delta=vv["gate"]["delta"],
        cmp_main=dci(cmp_["2026: v3 framework vs v2 framework"]["bootstrap"])
        if "2026: v3 framework vs v2 framework" in cmp_ else "n/a",
    )
    MD.write_text(text)


TEMPLATE = (Path(__file__).with_name("final_report_v3_template.md")).read_text() \
    if (Path(__file__).with_name("final_report_v3_template.md")).exists() else ""


def figure_compute(m: dict) -> None:
    """2026: accuracy vs generated tokens; SC over n CoT samples vs the framework rows."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for key, color, name, sc_label in (("2026_v2fw", ORANGE, "v2 framework", "CoT"),
                                       ("2026_v3", BLUE, "v3 framework", "CoT, adaptive budget")):
        res = m["tables"][key]
        base = res["table1"][0]
        tok = base["cost"]["tokens"]
        curve = sorted((int(n), a) for n, a in res["sc_curve"].items())
        ax.plot([n * tok / 1000 for n, _ in curve], [a for _, a in curve], color=color, lw=1.2,
                ls=(0, (3, 2)), label=f"SC over {sc_label}")
        for r in res["table1"]:
            if r.get("available") and r["kind"] != "sc_matched":
                full = r["row"] in (V2_FULL, V3_FULL)
                ax.plot(r["cost"]["tokens"] / 1000, r["overall"], "o", ms=8 if full else 5,
                        color=color, mec=SURFACE, mew=1.5,
                        label=name if full else None, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Generated tokens per question (thousands, log scale)", color=TEXT_2)
    ax.set_ylabel("Accuracy (%)", color=TEXT_2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"compute_2026.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- LaTeX
GEN_TEX = OUT / "generated"


def tex_escape(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("→", r"$\to$").replace("δ", r"$\delta$").replace("≤", r"$\le$"))


def dtex(b: dict) -> str:
    return f"{b['diff']:+.1f} [{b['ci95'][0]:+.1f}, {b['ci95'][1]:+.1f}]".replace("-", "$-$")


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def tex_table(path: str, caption: str, label: str, cols: str, head: list[str],
              body: list[list[str]], wide: bool = False, size: str = r"\footnotesize") -> None:
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[!t]", r"\centering", size, rf"\caption{{{caption}}}",
             rf"\label{{{label}}}",
             rf"\setlength{{\tabcolsep}}{{{'2.5pt' if 'script' in size else '3.5pt'}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for r in body:
        lines.append(r"\midrule" if r == ["MID"] else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    (GEN_TEX / path).write_text("\n".join(lines) + "\n")


def table1_rows(res: dict) -> list[list[str]]:
    out = []
    for r in res["table1"]:
        if not r.get("available"):
            continue
        name = tex_escape(r["row"].removeprefix("Qwen2.5: "))
        if r["kind"] == "sc_matched":
            name += f" ($n{{=}}{r['n_samples']}$" + (
                f", needs {r['n_needed']})" if r.get("insufficient") else ")")
        full = r["row"] in (V2_FULL, V3_FULL)
        acc = f"\\textbf{{{r['overall']:.1f}}}" if full else f"{r['overall']:.1f}"
        vb = dtex(r["vs_baseline"]) if "vs_baseline" in r else "--"
        cost = r.get("cost") or {}
        out.append([name, acc, f"{r['numerical']:.1f}",
                    f"{r['ci95'][0]:.1f}--{r['ci95'][1]:.1f}", vb,
                    f"{cost.get('tokens', 0) / 1000:.1f}k", f"{cost.get('calls', 0):.1f}"])
    return out


T1_HEAD = ["Configuration", "Acc.", "Num.", "95\\% CI", "$\\Delta$ vs.\\ baseline [CI]",
           "Tok./q", "Calls/q"]
T1_COLS = "lrrrrrr"


def macros(m: dict) -> dict[str, str]:
    t26, t26v2, t25 = m["tables"]["2026_v3"], m["tables"]["2026_v2fw"], m["tables"]["2025_v3"]
    t25v2, q25 = m["tables"]["2025_v2"], m["tables"]["2025_q25"]
    c = m["comparisons"]
    f = lambda x: f"{x:.1f}"                                        # noqa: E731
    sg = lambda x: f"{x:+.1f}".replace("-", "$-$")                  # noqa: E731
    mac: dict[str, str] = {"NTest": str(t26["n_test_questions"]),
                           "NPost": str(t25["n_test_questions"])}

    def fw(prefix: str, res: dict, full: str) -> None:
        r, base = row(res, full), res["table1"][0]
        sc = next(x for x in res["table1"] if x.get("kind") == "sc_matched")
        b, mc = res["full_vs_sc_matched"], res["full_vs_sc_matched_mcnemar"]
        mac.update({f"{prefix}Acc": f(r["overall"]), f"{prefix}Lo": f(r["ci95"][0]),
                    f"{prefix}Hi": f(r["ci95"][1]), f"{prefix}Base": f(base["overall"]),
                    f"{prefix}Gain": sg(r["vs_baseline"]["diff"]),
                    f"{prefix}Sc": f(sc["overall"]), f"{prefix}ScN": str(sc["n_samples"]),
                    f"{prefix}VsSc": sg(b["diff"]), f"{prefix}VsScLo": sg(b["ci95"][0]),
                    f"{prefix}VsScHi": sg(b["ci95"][1]), f"{prefix}VsScP": pval(mc["p"]),
                    f"{prefix}FwOnly": str(mc["a_only"]), f"{prefix}ScOnly": str(mc["b_only"]),
                    f"{prefix}Tok": f"{r['cost']['tokens'] / 1000:.0f}k",
                    f"{prefix}Pick": f(r["sel_max_prob"]), f"{prefix}Maj": f(r["sel_majority"]),
                    f"{prefix}Oracle": f(r["sel_oracle"]),
                    f"{prefix}PickVsMaj": sg(r["sel_max_prob"] - r["sel_majority"]),
                    f"{prefix}En": f(r["breakdown"]["language"]["English"]),
                    f"{prefix}Hin": f(r["breakdown"]["language"]["Hindi"]),
                    f"{prefix}Img": f(r["breakdown"]["requires_image"]["True"]),
                    f"{prefix}Txt": f(r["breakdown"]["requires_image"]["False"])})

    fw("Three", t26, V3_FULL)
    fw("Two", t26v2, V2_FULL)
    fw("Post", t25, V3_FULL)
    mac["PostTwoAcc"] = f(row(t25v2, V2_FULL)["overall"])
    mac["ThreePot"] = f(row(t26, "+ Code sandbox + RAG v3 + transcript (v3 PoT)")["overall"])
    mac["TwoPot"] = f(row(t26v2, "+ Code sandbox")["overall"])
    mac["ThreeCorr"] = f(row(t26, "+ Correction (cross-model critic)")["overall"])
    mac["ThreePotVer"] = f(row(t26, "+ Learned verifier v3 (PoT pool only)")["overall"])
    for key, prefix in (("2026: v3 framework vs v2 framework", "Vs"),
                        ("2026: v3 PoT vs v2 PoT", "PotVs"),
                        ("2026: CoT adaptive budget vs CoT", "ExtVs"),
                        ("2026: gated vs ungated correction (v3)", "GateVs"),
                        ("2025 (post-hoc): v3 framework vs v2 framework", "PostVs")):
        b, mc = c[key]["bootstrap"], c[key]["mcnemar"]
        mac.update({f"{prefix}Diff": sg(b["diff"]), f"{prefix}Lo": sg(b["ci95"][0]),
                    f"{prefix}Hi": sg(b["ci95"][1]), f"{prefix}P": pval(mc["p"])})
    g = row(t26, "+ Correction (gated)")["gate"]
    mac.update({"GateRtwU": str(g["right_to_wrong_ungated"]),
                "GateRtwG": str(g["right_to_wrong_gated"]),
                "GateWtrU": str(g["wrong_to_right_ungated"]),
                "GateWtrG": str(g["wrong_to_right_gated"]),
                "GateAccU": f(g["acc_corrected"]), "GateAccG": f(g["acc_gated"]),
                "GateAccO": f(g["acc_original"]), "GateDelta": f"{m['verifier_v3']['gate']['delta']:.2f}"
                .replace("-", "$-$")})
    mc3 = m["metacog"]["2026 v3_full"]
    mc2 = m["metacog"]["2026 full"]
    mac.update({"EpThree": f(mc3["EP"]), "EpTwo": f(mc2["EP"]),
                "FaThree": f(mc3["false_alarm"]), "FaTwo": f(mc2["false_alarm"]),
                "EcThree": f(mc3["EC_r1_of_detected"]), "EcTwo": f(mc2["EC_r1_of_detected"])})
    sh = m["shift"]
    mac.update({"TruncTrain": f(sh["train 2019-2024 pot_train"]["truncated"]),
                "TruncTest": f(sh["2026 v3_pot"]["truncated"]),
                "TokTrain": f"{sh['train 2019-2024 pot_train']['tokens'] / 1000:.1f}k",
                "TokTest": f"{sh['2026 v3_pot']['tokens'] / 1000:.1f}k"})
    vt = t26["verifier_test"]["v3_full"]["xgb_ranker/handcrafted"]
    pl = m["verifier_v3"]["primary_loyo"]
    mac.update({"VerAucTest": f"{vt['auc']:.2f}",
                "VerLoyo": f(pl["sel_16pool"]["max_prob"]),
                "VerLoyoMaj": f(m["verifier_v3"]["baselines_16pool"]["majority"]),
                "VerLoyoOracle": f(m["verifier_v3"]["baselines_16pool"]["oracle"])})
    d = m["dev_ab"]
    for k, prefix in (("transcript", "DevTx"), ("extend_pot", "DevExt"),
                      ("rag_v3", "DevRag")):
        o = d[k]["overall"]
        mac.update({f"{prefix}": sg(o["diff"]), f"{prefix}Lo": sg(o["ci95"][0]),
                    f"{prefix}Hi": sg(o["ci95"][1])})
    qb, qsc = q25["table1"][0], next(x for x in q25["table1"] if x.get("kind") == "sc_matched")
    qf = row(q25, "Qwen2.5: + Learned verifier (transfer)")
    mac.update({"QBase": f(qb["overall"]), "QSc": f(qsc["overall"]), "QFull": f(qf["overall"]),
                "QVsSc": sg(q25["full_vs_sc_matched"]["diff"]),
                "QVsScLo": sg(q25["full_vs_sc_matched"]["ci95"][0]),
                "QVsScHi": sg(q25["full_vs_sc_matched"]["ci95"][1])})
    pr = {r["year"]: r["correct"] for r in m["probes"]["probe_answer_q3"]["all"]}
    old = [v for y, v in pr.items() if int(y) <= 2024]
    mac.update({"ProbeOldLo": f(min(old)), "ProbeOldHi": f(max(old)),
                "ProbeTwentyFive": f(pr[2025] if 2025 in pr else pr["2025"]),
                "ProbeTwentySix": f(pr[2026] if 2026 in pr else pr["2026"])})
    pk = m["pass_at_k"]
    mac.update({"PassFiveThree": f(pk["2026 v3_pot"]["pass@5"])})
    rs = m["rag_sweep"]   # LOYO coverage at the same proxy precision: v3 recipe vs v2 recipe
    mac.update({"RagCovThree": f(100 * rs["ext=True w=0.9"]["coverage"]),
                "RagCovTwo": f(100 * rs["ext=False w=1.0"]["coverage"])})
    return mac


def write_tex(m: dict) -> None:
    GEN_TEX.mkdir(parents=True, exist_ok=True)
    mac = macros(m)
    (GEN_TEX / "macros.tex").write_text(
        "% generated by scripts/final_report_v3.py - do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    t26, t26v2, t25 = m["tables"]["2026_v3"], m["tables"]["2026_v2fw"], m["tables"]["2025_v3"]
    tex_table("t_table1_2026.tex",
              "Table~I on the clean 2026 test set (204 questions). Top: frozen v3 framework. "
              "Bottom: the frozen v2 framework re-run unchanged. Accuracy in \\%; "
              "95\\% cluster-bootstrap CIs over EN/HI twins; tokens are generated tokens.",
              "tab:t26", T1_COLS, T1_HEAD,
              [["\\multicolumn{7}{l}{\\emph{v3 framework}}"]] + table1_rows(t26) + [["MID"]]
              + [["\\multicolumn{7}{l}{\\emph{v2 framework (re-run)}}"]] + table1_rows(t26v2),
              wide=True)
    tex_table("t_table1_2025.tex",
              "v3 Table~I on 2025 (190 questions, \\textbf{post-hoc}: 2025 was used to pilot "
              "v1/v2).", "tab:t25", T1_COLS, T1_HEAD, table1_rows(t25), wide=True)
    body = []
    for k, cmp_ in m["comparisons"].items():
        b, mc = cmp_["bootstrap"], cmp_["mcnemar"]
        short = (k.replace("2025 (post-hoc): ", "2025$^\\dagger$ ").replace("2026: ", "2026 ")
                 .replace("framework", "fw").replace(" vs ", " vs.\\ "))
        body.append([tex_escape(short), dtex(b), f"{b['p_le_0']:.3f}",
                     f"{mc['a_only']}/{mc['b_only']}", pval(mc["p"])])
    tex_table("t_compare.tex", "Paired comparisons (cluster bootstrap; McNemar counts "
              "questions only $a$ / only $b$ gets right; it is meaningful for single-answer rows "
              "only, as Pass@1 rows are rounded per question). $^\\dagger$post-hoc.", "tab:cmp", "lrrrr",
              ["Comparison ($a$ vs.\\ $b$)", "$\\Delta$ [95\\% CI]", "P($\\Delta{\\le}0$)",
               "$a$/$b$", "$p$"], body, size=r"\scriptsize")
    keys = [("language", "English"), ("language", "Hindi"), ("requires_image", "False"),
            ("requires_image", "True"), ("subject", "Physics"), ("subject", "Chemistry"),
            ("subject", "Mathematics"), ("question_type", "MCQ-Single"),
            ("question_type", "MCQ-Multiple"), ("question_type", "Numerical"),
            ("question_type", "Matching")]
    cols = [row(t26v2, V2_FULL), row(t26, V3_FULL), row(m["tables"]["2025_v2"], V2_FULL),
            row(t25, V3_FULL)]
    body = [[{"False": "Text only", "True": "Needs image"}.get(v, v)]
            + [f1(r["breakdown"].get(g, {}).get(v)) for r in cols] for g, v in keys]
    tex_table("t_breakdown.tex", "Full-framework accuracy (\\%) by group. 2025 is post-hoc.",
              "tab:bd", "lrrrr", ["Group", "v2 '26", "v3 '26", "v2 '25", "v3 '25"], body)
    body = []
    for k, s in m["metacog"].items():
        body.append([tex_escape(k), str(s["n_chains"]),
                     f"{f1(s.get('acc_orig'))}$\\to${f1(s.get('acc_final'))}", f1(s["EP"]),
                     f1(s["EC_r1_of_detected"]), f1(s["false_alarm"]),
                     str(s["wrong_to_right_final"]), str(s["right_to_wrong_final"])])
    tex_table("t_correction.tex", "Correction loop, paper-style metacognition (\\%). EP: wrong "
              "solutions flagged in round 1; EC: round-1 fixes among detected; FA: correct "
              "solutions flagged. W$\\to$R / R$\\to$W: answer flips after all rounds.",
              "tab:corr", "lrrrrrrr",
              ["Run", "Chains", "Acc.", "EP", "EC", "FA", "W$\\to$R", "R$\\to$W"], body,
              size=r"\scriptsize")
    gch = m["verifier_v3"]["gate"]["chosen"]
    rows_ = [("Train (LOYO, picks $\\delta$)", gch)]
    for key, label in (("2026_v3", "2026 (clean)"), ("2025_v3", "2025 (post-hoc)")):
        rows_.append((label, row(m["tables"][key], "+ Correction (gated)")["gate"]))
    body = [[lab, str(s["n_chains"]), f"{s['acc_original']:.1f}", f"{s['acc_corrected']:.1f}",
             f"\\textbf{{{s['acc_gated']:.1f}}}",
             f"{s['wrong_to_right_ungated']}$\\to${s['wrong_to_right_gated']}",
             f"{s['right_to_wrong_ungated']}$\\to${s['right_to_wrong_gated']}"]
            for lab, s in rows_]
    tex_table("t_gate.tex", "Correction gate: per-chain accuracy (\\%) and flips, ungated "
              "$\\to$ gated.", "tab:gate", "lrrrrrr",
              ["Set", "Chains", "Orig.", "Corr.", "Gated", "W$\\to$R", "R$\\to$W"], body)
    body = []
    for key, full, exp, label in (("2026_v3", V3_FULL, "v3_full", "2026 v3"),
                                  ("2026_v2fw", V2_FULL, "full", "2026 v2"),
                                  ("2025_v3", V3_FULL, "v3_full", "2025 v3"),
                                  ("2025_v2", V2_FULL, "full", "2025 v2")):
        r = row(m["tables"][key], full)
        prim = m["primary"]["v3" if key.endswith("v3") else "v2"]
        name = f"{prim['model']}/{prim['feature_set']}"
        auc = m["tables"][key].get("verifier_test", {}).get(exp, {}).get(name, {}).get("auc")
        body.append([label, tex_escape(name), "--" if auc is None else f"{auc:.3f}",
                     f"\\textbf{{{r['sel_max_prob']:.1f}}}", f"{r['sel_majority']:.1f}",
                     f"{r['sel_random_expected']:.1f}", f"{r['sel_oracle']:.1f}"])
    tex_table("t_verifier.tex", "Primary verifier on its own 16-answer test pool: AUC and "
              "accuracy of the picked answer vs.\\ pool baselines (\\%).", "tab:ver",
              "llrrrrr", ["Pool", "Verifier", "AUC", "Pick", "Maj.", "Rand.", "Oracle"], body)
    body = [[tex_escape(k), str(s["n"]), f"{s['acc']:.1f}", f"{s['truncated']:.1f}",
             f"{s['tokens'] / 1000:.1f}k", f1(s["acc_truncated"]), f1(s["acc_complete"])]
            for k, s in m["shift"].items()]
    tex_table("t_shift.tex", "Shift between the verifier's training solutions and the test "
              "pools. Cut: \\texttt{finish = length} on the final call.", "tab:shift",
              "lrrrrrr", ["Solutions", "Rows", "Acc.", "Cut", "Tok.", "Acc.\\,cut",
                          "Acc.\\,compl."], body, size=r"\scriptsize")
    tex_table("t_q25.tex", "Bridge: the framework with the paper's Qwen2.5-VL-7B on 2025 "
              "(clean for that model; verifier transferred from Qwen3).", "tab:q25", T1_COLS,
              T1_HEAD, table1_rows(m["tables"]["2025_q25"]), wide=True)
    body = [[tex_escape(k), f"{s['pass@1']:.1f}", f"{s['pass@3']:.1f}", f"{s['pass@5']:.1f}"]
            for k, s in m["pass_at_k"].items()]
    tex_table("t_passk.tex", "Unbiased pass@$k$ (\\%) of the sampled runs.", "tab:passk",
              "lrrr", ["Run", "@1", "@3", "@5"], body)
    p = m["probes"]
    q3 = {int(r["year"]): r["correct"] for r in p["probe_answer_q3"]["all"]}
    q25p = {int(r["year"]): r["correct"] for r in p["probe_answer_q25"]["all"]}
    ys = sorted(q3)
    tex_table("t_probes.tex", "Memorisation probe: answer-only accuracy (\\%, 64 tokens) by "
              "exam year. 2026 is after both models' release.", "tab:probes",
              "l" + "r" * len(ys), ["Model"] + [f"'{str(y)[2:]}" for y in ys],
              [["Qwen3-VL-8B"] + [f1(q3[y]) for y in ys],
               ["Qwen2.5-VL-7B"] + [f1(q25p.get(y)) for y in ys]])
    names = {"transcript": "Transcript block (PoT)", "extend_pot": "Adaptive budget (PoT)",
             "extend_cot": "Adaptive budget (CoT)", "rag_v2": "RAG, v2 recipe",
             "rag_v3": "RAG v3 (hybrid + JEEBench)"}
    body = [[lab, f"{d['overall']['acc_base']:.1f}", f"{d['overall']['acc_arm']:.1f}",
             dtex(d["overall"])] for k, lab in names.items() for d in [m["dev_ab"][k]]]
    tex_table("t_devab.tex", "Dev A/B tests on 2024 (204 questions) that decided the freeze. "
              "Rule fixed in advance: include iff $\\Delta>0$ and the CI lower bound $>0$.",
              "tab:devab", "lrrr", ["Arm", "Base", "Arm", "$\\Delta$ [95\\% CI]"], body)


def main() -> None:
    m = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    figure(m)
    figure_compute(m)
    write_tex(m)
    if TEMPLATE:
        write_md(m)
    print(f"wrote {OUT / 'metrics.json'}, {FIG}/, {GEN_TEX}/"
          + (f", {MD}" if TEMPLATE else " (no template yet)"))


if __name__ == "__main__":
    main()
