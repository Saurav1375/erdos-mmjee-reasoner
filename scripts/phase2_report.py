"""Phase-2 figures, metrics and LaTeX tables (work done BEFORE the full run only).

Scope of Phase 2: reproduction of a base-paper number (Qwen2.5-VL-7B), the Qwen3-VL-8B baseline
on the 2025 test set (16 CoT samples), the end-to-end pilot of all modules (v1) and the
verification run of the fixes (v2). Full-run results belong to the final report and are NOT read.

usage: .venv/bin/python scripts/phase2_report.py
inputs : artifacts/eval/results.lab_baseline_only.json (lab evaluate before the full run:
           baseline 16 samples + reproduction),
         artifacts/generations/cot/candidates.jsonl (samples 0-15 = the baseline samples),
         artifacts/pilot_v1/ (v1 pilot), artifacts/generations/{pot,pot_corr_cross}/*.pilot.*
           (v2 re-run with the fixes), artifacts/verifier.pilot/, artifacts/rag/kb.pilot/
outputs: report/phase2/figures/*.pdf|png, report/phase2/generated/*.tex, report/phase2/metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from mmjee_reasoner.config import load_config
from mmjee_reasoner.pipeline.evaluate import Evaluator

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "report" / "phase2"
FIG, GEN = OUT / "figures", OUT / "generated"
N_BASE = 16  # CoT samples of the Phase-2 baseline (the full run later added 4 more)

# Palette: validated reference categorical slots + neutral for baselines (see report/figures.py).
SURFACE, TEXT, TEXT_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8d8c87"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PAPER_EC = (1.1, 5.2)  # base paper: single-pass error-correction success range (%)
# Answer-forcing A/B on the 480 pilot trajectories that hit the token cap (run 2026-10-03 on the
# lab, script output kept in the session log; no artifact file): accuracy of "forced" answers
# vs. the SOP CoT fallback on the same trajectories.
FORCING_AB = {"n": 480, "forced": 21.9, "fallback": 30.0,
              "by_type": {"forced": {"Numerical": 13.6, "MCQ-Single": 40.7,
                                     "MCQ-Multiple": 10.9, "Matching": 31.0},
                          "fallback": {"Numerical": 24.1, "MCQ-Single": 38.4,
                                       "MCQ-Multiple": 21.8, "Matching": 40.7}}}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


# ---------------------------------------------------------------- helpers
def style(ax, title: str, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=7.5)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, loc="left")
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def hbars(ax, labels, vals, cis, colors, xmax=100):
    y = np.arange(len(labels))[::-1]
    err = np.array([[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]])
    ax.barh(y, vals, color=colors, height=0.62, edgecolor=SURFACE, linewidth=1)
    ax.errorbar(vals, y, xerr=err, fmt="none", ecolor=TEXT_2, elinewidth=0.8, capsize=2)
    for yi, v, c in zip(y, vals, cis):
        ax.text(c[1] + 1, yi, f"{v:.1f}", va="center", color=TEXT, fontsize=7.5)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, xmax)


def f1(x) -> str:
    return "--" if x is None else f"{x:.1f}"


def tex(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def table(path: str, header: list[str], rows: list[list[str]], align: str, caption: str,
          label: str, wide: bool = False) -> None:
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[t]", r"\centering", r"\footnotesize",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{3.5pt}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / path).write_text("\n".join(lines) + "\n")


def anatomy(d: pd.DataFrame) -> dict:
    cat = np.select([d["used_fallback"].astype(bool), d["n_code_ok"] > 0,
                     d["n_code_blocks"] == 0],
                    ["CoT fallback", "Code executed", "Answered without code"],
                    "Code failed only")
    g = d.assign(cat=cat).groupby("cat")["correct"].agg(["size", "mean"])
    return {k: {"n": int(r["size"]), "share": float(r["size"] / len(d) * 100),
                "acc": float(r["mean"] * 100)} for k, r in g.iterrows()}


def critic_stats(path: Path) -> dict:
    chains = json.loads(path.read_text())
    crit = [c for ch in chains for c in ch["critiques"]]
    return {"calls": len(crit), "forced": sum(1 for c in crit if c.get("forced_verdict"))}


# ---------------------------------------------------------------- data
def collect() -> dict:
    cfg = load_config()
    base = json.loads((ART / "eval" / "results.lab_baseline_only.json").read_text())
    b0 = base["table1"][0]
    assert b0["pass_at_1"]["runs"] == N_BASE, "expected the 16-sample Phase-2 baseline"
    m: dict = {"baseline": {"overall": b0["overall"], "ci95": b0["ci95"],
                            "run_std": b0["pass_at_1"]["run_std"], "numerical": b0["numerical"],
                            "tokens": b0["cost"]["tokens"], "breakdown": b0["breakdown"],
                            "sc_curve": {int(k): v for k, v in base["sc_curve"].items()}},
               "repro": base["reproduction"]}
    cot = pd.read_json(ART / "generations" / "cot" / "candidates.jsonl", lines=True)
    cot = cot[cot["sample"] < N_BASE]
    assert abs(cot["correct"].mean() * 100 - b0["overall"]) < 1e-6  # same 16 samples
    m["baseline"]["truncated_pct"] = float((cot["finish"] == "length").mean() * 100)
    m["baseline"]["n_samples"] = len(cot)
    m["_cot_tokens"] = cot["completion_tokens"].tolist()

    pv1 = json.loads((ART / "pilot_v1" / "results.pilot.json").read_text())
    m["pilot"] = {"table1": [r for r in pv1["table1"] if r.get("available")],
                  "full_vs_sc": pv1.get("full_vs_sc_matched"),
                  "sc_curve": {int(k): v for k, v in pv1["sc_curve"].items()},
                  "correction": pv1["correction"], "verifier_test": pv1["verifier_test"]}
    v1pot = pd.read_json(ART / "pilot_v1" / "pot" / "candidates.pilot.jsonl", lines=True)
    m["pilot"]["pot_anatomy"] = anatomy(v1pot)
    ver = json.loads((ART / "verifier.pilot" / "report.json").read_text())
    for c in ver["configs"]:
        c.pop("full_oof", None)
    m["pilot"]["verifier"] = ver
    kb = json.loads((ART / "rag" / "kb.pilot" / "meta.json").read_text())
    m["pilot"]["kb"] = {"entries": kb["entries"], "threshold": kb["threshold"]}

    code = {}
    for tag, f in (("v1", ART / "pilot_v1" / "pot" / "candidates.pilot.jsonl"),
                   ("v2", ART / "generations" / "pot" / "candidates.pilot.jsonl")):
        d = pd.read_json(f, lines=True)
        errs = pd.Series([str(e) for es in d["exec_errors"] for e in es], dtype=str)
        code[tag] = {"ok": int(d["n_code_ok"].sum()), "blocked": int((errs == "blocked").sum()),
                     "NameError": int((errs == "NameError").sum()),
                     "acc": float(d["correct"].mean() * 100)}
    m["fixes"] = {"code": code,
                  "cross_v1": pv1["correction"]["pot_corr_cross"],
                  "cross_v2": Evaluator(cfg, ".pilot").correction_analysis("pot_corr_cross"),
                  "critic_v1": critic_stats(ART / "pilot_v1" / "pot_corr_cross" / "chains.pilot.json"),
                  "critic_v2": critic_stats(ART / "generations" / "pot_corr_cross" / "chains.pilot.json"),
                  "forcing_ab": FORCING_AB}
    return m


# ---------------------------------------------------------------- figures
def figures(m: dict) -> None:
    b, rep, p = m["baseline"], m["repro"], m["pilot"]

    # context: paper vs our harness
    ctx = [("Qwen2.5-VL-7B (paper)", 12.4, NEUTRAL),
           ("Qwen2.5-VL-7B (ours, reproduction)", rep["pass_at_1"]["pass1"], BLUE),
           ("Gemma-3-27B (paper)", 30.6, NEUTRAL),
           ("Qwen3-VL-8B (ours, baseline)", b["overall"], BLUE),
           ("Qwen3-VL-235B (paper)", 46.2, NEUTRAL)]
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    style(ax, "")
    y = np.arange(len(ctx))[::-1]
    ax.barh(y, [c[1] for c in ctx], color=[c[2] for c in ctx], height=0.62)
    for yi, c in zip(y, ctx):
        ax.text(c[1] + 0.8, yi, f"{c[1]:.1f}", va="center", fontsize=6.8, color=TEXT)
    ax.set_yticks(y, [c[0] for c in ctx], fontsize=6.6)
    ax.set_xlim(0, 55)
    ax.set_xlabel("Pass@1 on mmJEE-Eval 2025 (%)", color=TEXT_2)
    save(fig, "fig_context")

    # baseline breakdown: Qwen2.5-VL-7B vs Qwen3-VL-8B
    groups = [("question_type", ["Numerical", "MCQ-Single", "MCQ-Multiple", "Matching"],
               ["Num.", "MCQ-S", "MCQ-M", "Match."], "Question type"),
              ("language", ["English", "Hindi"], ["EN", "HI"], "Language"),
              ("subject", ["Mathematics", "Chemistry", "Physics"], ["Math", "Chem", "Phys"],
               "Subject"),
              ("requires_image", ["False", "True"], ["No", "Yes"], "Needs diagram")]
    series = [("Qwen2.5-VL-7B (reproduction)", rep["breakdown"], NEUTRAL),
              ("Qwen3-VL-8B (baseline)", b["breakdown"], BLUE)]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.8), sharey=True,
                             gridspec_kw={"width_ratios": [4, 2, 3, 2]})
    for ax, (key, keys, names, title) in zip(axes, groups):
        style(ax, title, "y")
        for i, (lab, bd, c) in enumerate(series):
            vals = [bd[key][k] for k in keys]
            ax.bar(np.arange(len(keys)) + (i - 0.5) * 0.38, vals, 0.38, color=c,
                   edgecolor=SURFACE, lw=1, label=lab)
        ax.set_xticks(range(len(keys)), names)
        ax.set_ylim(0, 65)
    axes[0].set_ylabel("Accuracy (%)", color=TEXT_2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=6.8, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.08))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.83, wspace=0.12)
    save(fig, "fig_baseline_breakdown")

    # baseline self-consistency curve + token histogram
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 1.9))
    style(a1, "Self-consistency (majority vote), Qwen3-VL-8B", "both")
    sc = b["sc_curve"]
    a1.plot(list(sc), list(sc.values()), color=BLUE, lw=1.6, marker="o", ms=3)
    a1.axhline(b["overall"], color=NEUTRAL, lw=1, ls="--")
    a1.text(16, b["overall"] - 1.8, f"Pass@1 {b['overall']:.1f}", ha="right", fontsize=6.5,
            color=TEXT_2)
    a1.set_xlabel("Samples voted (n)", color=TEXT_2)
    a1.set_ylabel("Accuracy (%)", color=TEXT_2)
    a1.set_xticks([1, 4, 8, 12, 16])
    a1.set_ylim(38, 58)
    style(a2, f"CoT length ({b['n_samples']:,} samples)", "y")
    a2.hist(np.clip(m["_cot_tokens"], None, 4200), bins=np.arange(0, 4300, 200), color=BLUE,
            edgecolor=SURFACE, lw=0.8)
    a2.axvline(4096, color=TEXT_2, lw=1, ls="--")
    a2.text(4000, a2.get_ylim()[1] * 0.85, f"hit 4,096 cap: {b['truncated_pct']:.0f}%",
            ha="right", fontsize=6.8, color=TEXT)
    a2.set_xlabel("Completion tokens", color=TEXT_2)
    a2.set_ylabel("Count", color=TEXT_2)
    fig.tight_layout(w_pad=2)
    save(fig, "fig_baseline_sc_tokens")

    # pilot table I
    short = {"Baseline (single-pass CoT)": "Baseline CoT",
             "Self-consistency (matched compute)": "Self-consistency (matched)",
             "+ Code sandbox": "+ Code sandbox", "+ Agentic correction (same-model critic)":
             "+ Correction (same)", "+ Agentic correction (cross-model critic)":
             "+ Correction (cross)", "+ Learned verifier": "+ Learned verifier",
             "+ Retrieval (RAG)": "+ RAG = Full framework"}
    pt = [r for r in p["table1"] if r["row"] in short]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    style(ax, "")
    hbars(ax, [short[r["row"]] for r in pt], [r["overall"] for r in pt], [r["ci95"] for r in pt],
          [NEUTRAL if i < 2 else BLUE for i in range(len(pt))], 80)
    ax.set_xlabel("Pilot accuracy (%), 40 questions, 95% CI", color=TEXT_2)
    save(fig, "fig_pilot")

    # pilot accuracy vs compute
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    style(ax, "", "both")
    tok = pt[0]["cost"]["tokens"]
    ax.plot([n * tok / 1000 for n in p["sc_curve"]], list(p["sc_curve"].values()), color=NEUTRAL,
            lw=1.6, marker="o", ms=3, label="Self-consistency (CoT, n = 1..16)")
    names = {"+ Code sandbox": ("PoT", (5, -3)),
             "+ Agentic correction (same-model critic)": ("+corr. same", (5, 2)),
             "+ Agentic correction (cross-model critic)": ("+corr. cross", (5, -8)),
             "+ Learned verifier": ("+verifier", (5, -9)), "+ Retrieval (RAG)": ("+RAG = full", (5, 3))}
    for r in pt[2:]:
        x, yv = r["cost"]["tokens"] / 1000, r["overall"]
        ax.scatter(x, yv, s=40, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=3)
        lab, off = names[r["row"]]
        ax.annotate(lab, (x, yv), textcoords="offset points", xytext=off, fontsize=6.3, color=TEXT)
    ax.scatter([], [], s=40, color=BLUE, label="Framework rows (additive)")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([2, 5, 10, 20, 50], ["2k", "5k", "10k", "20k", "50k"])
    ax.set_xlabel("Generated tokens per question (log scale)", color=TEXT_2)
    ax.set_ylabel("Pilot accuracy (%)", color=TEXT_2)
    ax.set_ylim(15, 65)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    save(fig, "fig_pilot_compute")

    # pilot correction stats (v1)
    corr = p["correction"]
    crit = [("pot_corr_same", "Same-model", ORANGE), ("pot_corr_cross", "Cross-model", AQUA)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.0), gridspec_kw={"width_ratios": [3, 2]})
    style(a1, "Critic and corrector rates (pilot v1, 320 chains each)", "y")
    mets = [("detect_rate", "Detects error\n(wrong sols.)"),
            ("false_alarm_rate", "False alarm\n(correct sols.)"),
            ("fix_rate_of_detected", "Fixed |\ndetected")]
    for i, (k, lab, c) in enumerate(crit):
        vals = [corr[k][mk] for mk, _ in mets]
        bars = a1.bar(np.arange(3) + (i - 0.5) * 0.36, vals, 0.36, color=c, edgecolor=SURFACE,
                      lw=1, label=lab)
        for bb, v in zip(bars, vals):
            a1.text(bb.get_x() + bb.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center",
                    fontsize=6.5, color=TEXT)
    a1.fill_between([1.55, 2.45], *PAPER_EC, color=NEUTRAL, alpha=0.45, lw=0, zorder=0,
                    label="Base paper, single-pass fix rate (1.1–5.2)")
    a1.set_xticks(range(3), [lab for _, lab in mets])
    a1.set_ylabel("%", color=TEXT_2)
    a1.set_ylim(0, 90)
    a1.set_title("Critic and corrector rates (pilot v1, 320 chains each)", color=TEXT,
                 fontsize=9, loc="left", pad=26)
    a1.legend(fontsize=6.3, frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0, 0.98))
    style(a2, "", "x")
    a2.set_title("Net answer flips", color=TEXT, fontsize=9, loc="left", pad=16)
    y = np.arange(2)[::-1]
    w2r = [corr[k]["wrong_to_right"] for k, _, _ in crit]
    r2w = [corr[k]["right_to_wrong"] for k, _, _ in crit]
    a2.barh(y + 0.17, w2r, 0.32, color=BLUE, edgecolor=SURFACE, label="wrong → right")
    a2.barh(y - 0.17, r2w, 0.32, color=NEUTRAL, edgecolor=SURFACE, label="right → wrong")
    for yi, a, bv in zip(y, w2r, r2w):
        a2.text(a + 0.5, yi + 0.17, str(a), va="center", fontsize=6.5, color=TEXT)
        a2.text(bv + 0.5, yi - 0.17, str(bv), va="center", fontsize=6.5, color=TEXT)
    a2.set_yticks(y, [lab for _, lab, _ in crit])
    a2.set_xlim(0, 32)
    a2.set_xlabel("Chains", color=TEXT_2)
    a2.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0, 0.98))
    fig.tight_layout(w_pad=2)
    save(fig, "fig_pilot_correction")

    # pilot PoT anatomy (v1) -- motivates the fallback analysis
    an = p["pot_anatomy"]
    order = [k for k in ("Code executed", "Answered without code", "Code failed only",
                         "CoT fallback") if k in an]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 1.5), sharey=True)
    style(a1, "Share of pilot PoT trajectories (%)")
    style(a2, "Accuracy within group (%)")
    y = np.arange(len(order))[::-1]
    a1.barh(y, [an[k]["share"] for k in order], color=BLUE, height=0.6)
    a2.barh(y, [an[k]["acc"] for k in order], color=BLUE, height=0.6)
    for yi, k in zip(y, order):
        a1.text(an[k]["share"] + 0.8, yi, f"{an[k]['share']:.1f} (n={an[k]['n']})", va="center",
                fontsize=6.5, color=TEXT)
        a2.text(an[k]["acc"] + 0.8, yi, f"{an[k]['acc']:.1f}", va="center", fontsize=6.5,
                color=TEXT)
    a1.set_yticks(y, order)
    a1.set_xlim(0, 75)
    a2.set_xlim(0, 75)
    fig.tight_layout(w_pad=2)
    save(fig, "fig_pilot_anatomy")

    # pilot verifier AUC
    cf = p["verifier"]["configs"]
    vt = p["verifier_test"]["full"]
    labs = [f"{'LR' if c['model'] == 'logreg' else 'XGB'} / {c['feature_set'][:5]}" for c in cf]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    style(ax, "", "y")
    x = np.arange(len(cf))
    for off, vals, col, lab in (
            (-0.27, [c["cv"]["auc"] for c in cf], BLUE, "CV 2019–23"),
            (0.0, [c["dev"]["auc"] for c in cf], ORANGE, "Dev 2024"),
            (0.27, [vt[f"{c['model']}/{c['feature_set']}"]["auc"] for c in cf], AQUA,
             "2025 pilot pool")):
        ax.bar(x + off, vals, 0.27, color=col, edgecolor=SURFACE, label=lab)
    ax.axhline(0.5, color=TEXT_2, lw=0.8, ls=":")
    ax.set_xticks(x, labs, rotation=25, ha="right", fontsize=6.6)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC AUC", color=TEXT_2)
    ax.legend(fontsize=6.3, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_verifier_pilot")

    # fixes: v1 -> v2
    fx = m["fixes"]
    c1, c2 = fx["code"]["v1"], fx["code"]["v2"]
    items = [("Code blocks blocked", c1["blocked"], c2["blocked"]),
             ("NameError (lost state)", c1["NameError"], c2["NameError"]),
             ("Critic replies unparsed", fx["cross_v1"]["critic_unparsed"],
              fx["cross_v2"]["critic_unparsed"])]
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    style(ax, "")
    y = np.arange(len(items))[::-1]
    ax.barh(y + 0.17, [i[1] for i in items], 0.32, color=NEUTRAL, label="v1 pilot")
    ax.barh(y - 0.17, [i[2] for i in items], 0.32, color=BLUE, label="v2 (after fixes)")
    for yi, it in zip(y, items):
        ax.text(it[1] + 1, yi + 0.17, str(it[1]), va="center", fontsize=6.5, color=TEXT)
        ax.text(it[2] + 1, yi - 0.17, str(it[2]), va="center", fontsize=6.5, color=TEXT)
    ax.set_yticks(y, [i[0] for i in items])
    ax.set_xlabel("Count (40-question pilot)", color=TEXT_2)
    ax.set_xlim(0, 85)
    ax.legend(fontsize=6.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_fixes")

    # answer-forcing A/B
    ab = fx["forcing_ab"]
    types = ["Numerical", "MCQ-Single", "MCQ-Multiple", "Matching"]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    style(ax, "", "y")
    x = np.arange(len(types) + 1)
    lab_x = ["All"] + [t.replace("MCQ-", "MCQ-\n") for t in types]
    for off, key, col, lab in ((-0.19, "fallback", BLUE, "SOP CoT fallback (kept)"),
                               (0.19, "forced", NEUTRAL, "Answer forcing (rejected)")):
        vals = [ab[key]] + [ab["by_type"][key][t] for t in types]
        ax.bar(x + off, vals, 0.38, color=col, edgecolor=SURFACE, label=lab)
        ax.text(off, vals[0] + 1.2, f"{vals[0]:.1f}", ha="center", fontsize=6.5, color=TEXT)
    ax.set_xticks(x, lab_x)
    ax.set_ylabel("Accuracy (%)", color=TEXT_2)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=6.3, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_forcing_ab")


# ---------------------------------------------------------------- tables + macros
def tables(m: dict) -> None:
    b, rep, p, fx = m["baseline"], m["repro"], m["pilot"], m["fixes"]

    bd, rb = b["breakdown"], rep["breakdown"]
    keys = [("question_type", "Numerical", "Numerical"), ("question_type", "MCQ-Single", "MCQ-Single"),
            ("question_type", "MCQ-Multiple", "MCQ-Multiple"), ("question_type", "Matching", "Matching"),
            ("language", "English", "English"), ("language", "Hindi", "Hindi"),
            ("subject", "Mathematics", "Mathematics"), ("subject", "Chemistry", "Chemistry"),
            ("subject", "Physics", "Physics"), ("requires_image", "False", "Diagram not needed"),
            ("requires_image", "True", "Diagram needed")]
    rows = [["\\textbf{Pass@1 (all 190)}", f"\\textbf{{{rep['pass_at_1']['pass1']:.1f}}}",
             f"\\textbf{{{b['overall']:.1f}}}"],
            ["95\\% CI (cluster bootstrap)", f"[{rep['ci95'][0]:.1f}, {rep['ci95'][1]:.1f}]",
             f"[{b['ci95'][0]:.1f}, {b['ci95'][1]:.1f}]"],
            ["Std over runs", f"{rep['pass_at_1']['run_std']:.1f} ({rep['pass_at_1']['runs']} runs)",
             f"{b['run_std']:.1f} ({N_BASE} runs)"],
            ["Paper (Table 3)", "12.4 $\\pm$ 1.9", "--"]]
    rows += [[lab, f1(rb[g][k]), f1(bd[g][k])] for g, k, lab in keys]
    rows += [["Self-consistency, 16 votes", "--", f1(b["sc_curve"][16])],
             ["Mean generated tokens / sample", "--", f"{b['tokens']:,.0f}"],
             ["Samples hitting the 4{,}096 cap", "--", f"{b['truncated_pct']:.0f}\\%"]]
    table("t_baseline.tex", ["Metric (2025 test set)", "Qwen2.5-VL-7B", "Qwen3-VL-8B"], rows,
          "lrr", "Baselines on the 2025 held-out set with the paper's prompts (single-pass CoT). "
          "Qwen2.5-VL-7B reproduces the paper's number; Qwen3-VL-8B is our substrate.",
          "tab:baseline")

    pt = p["table1"]
    rows = []
    for r in pt:
        if r["row"] == "Full framework":
            continue
        rows.append([tex(r["row"].replace(" (matched compute)", " (matched)")
                         .replace("Agentic correction", "Correction").replace("-model critic", "")),
                     f"{r['overall']:.1f}", f"[{r['ci95'][0]:.1f}, {r['ci95'][1]:.1f}]",
                     f1(r.get("numerical")), f"{(r.get('cost') or {}).get('tokens', 0) / 1000:.1f}k"])
    fv = p["full_vs_sc"]
    table("t_pilot.tex", ["Configuration (additive)", "Acc.", "95\\% CI", "Num.", "Tok/q"], rows,
          "lrcrr", "End-to-end pilot (v1): 40 test questions (20 EN/HI pairs, stratified by type), "
          "verifier and KB from 150 training questions. Full framework vs.\\ matched-compute SC: "
          f"{fv['diff']:+.1f} points, paired 95\\% CI [{fv['ci95'][0]:+.1f}, {fv['ci95'][1]:+.1f}].",
          "tab:pilot")

    rows = []
    for k, lab in (("pot_corr_same", "Same-model (Qwen3-VL)"),
                   ("pot_corr_cross", "Cross-model (InternVL3.5)"), ("full", "Full (RAG, cross)")):
        c = p["correction"][k]
        rows.append([lab, f"{c['acc_before']:.1f} $\\to$ {c['acc_after']:.1f}", f1(c["detect_rate"]),
                     f1(c["false_alarm_rate"]), f1(c["fix_rate_of_detected"]),
                     f"{c['wrong_to_right']} / {c['right_to_wrong']}", str(c["critic_unparsed"])])
    rows.append(["Base paper (single pass)", "", "", "", "1.1--5.2", "", ""])
    table("t_correction.tex", ["Critic", "Before $\\to$ after", "Detect", "False al.",
                               "Fix$|$det.", "W$\\to$R / R$\\to$W", "Unparsed"], rows, "lrrrrrr",
          "Correction loop in the pilot (v1, 320 chains each). Detect = critic flags an originally "
          "wrong solution; False al. = flags a correct one; Fix$|$det. = detected wrong solutions "
          "that end correct.", "tab:corr", wide=True)

    v = p["verifier"]
    vt = p["verifier_test"]["full"]
    rows = []
    for c in v["configs"]:
        key = f"{c['model']}/{c['feature_set']}"
        rows.append([("LR" if c["model"] == "logreg" else "XGB") + " / " + c["feature_set"],
                     f"{c['cv']['auc']:.3f}", f"{c['dev']['auc']:.3f}", f"{c['dev']['f1']:.3f}",
                     f"{vt[key]['auc']:.3f}", f"{vt[key]['sel_weighted_vote']:.1f}"])
    rows.append(["Majority vote", "", "", "", "", f"{vt['majority']:.1f}"])
    rows.append(["Oracle (any correct)", "", "", "", "", f"{vt['oracle']:.1f}"])
    table("t_verifier.tex", ["Model / features", "CV AUC", "Dev AUC", "F1", "Test AUC",
                             "Test sel."], rows, "lrrrrr",
          f"Learned verifier in the pilot ({v['n_solutions']:,} auto-labelled solutions, "
          f"{v['pos_rate'] * 100:.1f}\\% correct). CV: GroupKFold by twin pair on 2019--2023; "
          "dev: 2024; test: the 2025 pilot pool (16 candidates per question). Test sel.\\ = "
          "selection accuracy (weighted vote).", "tab:verifier")

    c1, c2 = fx["code"]["v1"], fx["code"]["v2"]
    k1, k2 = fx["cross_v1"], fx["cross_v2"]
    rows = [["Code blocks executed OK", str(c1["ok"]), str(c2["ok"])],
            ["Blocked by sandbox whitelist", str(c1["blocked"]), str(c2["blocked"])],
            ["NameError (state lost)", str(c1["NameError"]), str(c2["NameError"])],
            ["PoT accuracy (320 traj.)", f1(c1["acc"]), f1(c2["acc"])],
            ["Critic calls / unparsed", f"{fx['critic_v1']['calls']} / {k1['critic_unparsed']}",
             f"{fx['critic_v2']['calls']} / {k2['critic_unparsed']}"],
            ["Verdicts via verdict forcing", "--", str(fx["critic_v2"]["forced"])],
            ["Critic detection rate", f1(k1["detect_rate"]), f1(k2["detect_rate"])],
            ["Critic false-alarm rate", f1(k1["false_alarm_rate"]), f1(k2["false_alarm_rate"])],
            ["Acc.\\ before $\\to$ after",
             f"{k1['acc_before']:.1f} $\\to$ {k1['acc_after']:.1f}",
             f"{k2['acc_before']:.1f} $\\to$ {k2['acc_after']:.1f}"]]
    table("t_fixes.tex", ["Pilot check (40 q, cross critic)", "v1", "v2"], rows, "lrr",
          "Verification run of the fixes on the same pilot questions.", "tab:fixes")

    pl = {r["row"]: r for r in p["table1"]}
    an = p["pot_anatomy"]
    mac = {
        "BaseAcc": f"{b['overall']:.1f}", "BaseLo": f"{b['ci95'][0]:.1f}",
        "BaseHi": f"{b['ci95'][1]:.1f}", "BaseStd": f"{b['run_std']:.1f}",
        "BaseTrunc": f"{b['truncated_pct']:.0f}", "BaseScSixteen": f"{b['sc_curve'][16]:.1f}",
        "BaseScMax": f"{max(b['sc_curve'].values()):.1f}",
        "ReproAcc": f"{rep['pass_at_1']['pass1']:.1f}", "ReproStd": f"{rep['pass_at_1']['run_std']:.1f}",
        "ReproLo": f"{rep['ci95'][0]:.1f}", "ReproHi": f"{rep['ci95'][1]:.1f}",
        "PilotBase": f"{pl['Baseline (single-pass CoT)']['overall']:.1f}",
        "PilotSc": f"{pl['Self-consistency (matched compute)']['overall']:.1f}",
        "PilotPot": f"{pl['+ Code sandbox']['overall']:.1f}",
        "PilotFull": f"{pl['Full framework']['overall']:.1f}",
        "PilotVsSc": f"{p['full_vs_sc']['diff']:+.1f}",
        "PilotVsScLo": f"{p['full_vs_sc']['ci95'][0]:+.1f}",
        "PilotVsScHi": f"{p['full_vs_sc']['ci95'][1]:+.1f}",
        "PilotFallback": f"{an.get('CoT fallback', {}).get('share', 0):.0f}",
        "PilotCodeRan": f"{an.get('Code executed', {}).get('share', 0):.0f}",
        "PilotKb": str(p["kb"]["entries"]),
        "CriticVTwoCalls": str(fx["critic_v2"]["calls"]),
        "CriticVTwoForced": str(fx["critic_v2"]["forced"]),
    }
    (GEN / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n"
                                            for k, v in mac.items()))


def main() -> None:
    m = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (FIG, GEN):  # drop outputs of earlier versions of this script
        if d.exists():
            for f in d.iterdir():
                f.unlink()
    figures(m)
    tables(m)
    (OUT / "metrics.json").write_text(json.dumps({k: v for k, v in m.items() if k[0] != "_"},
                                                 indent=2, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
