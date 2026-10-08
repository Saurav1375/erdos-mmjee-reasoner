"""Final-report figures, metrics and LaTeX tables, regenerated from the full-run artifacts only.

usage: .venv/bin/python scripts/final_report.py   (after scripts/lab/sync_from_lab.sh)
inputs : artifacts/eval/results.json + per_question.csv (mmjee evaluate on the full run),
         artifacts/generations/*/candidates.jsonl, artifacts/verifier/, artifacts/rag/,
         artifacts/logs/full.supervisor.out (stage times)
outputs: report/final/figures/*.pdf|png, report/final/generated/*.tex, report/final/metrics.json
No number in the final LaTeX report is typed by hand: tables and \\Macros come from here.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from mmjee_reasoner.config import load_config
from mmjee_reasoner.pipeline.evaluate import Evaluator
from mmjee_reasoner.scoring.metrics import mcnemar, paired_bootstrap
from mmjee_reasoner.verifier.model import selection_accuracy

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "report" / "final"
FIG, GEN = OUT / "figures", OUT / "generated"

# Palette: validated reference categorical slots + neutral for baselines (see report/figures.py).
SURFACE, TEXT, TEXT_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8d8c87"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PAPER_EC = (1.1, 5.2)  # base paper: single-pass error-correction success range (%)
BASE = "Baseline (single-pass CoT)"
SC = "Self-consistency (matched compute)"
SHORT = {BASE: "Baseline CoT", SC: "Self-consistency (matched)",
         "+ Code sandbox": "+ Code sandbox (PoT)",
         "+ Agentic correction (same-model critic)": "+ Correction, same-model critic",
         "+ Agentic correction (cross-model critic)": "+ Correction, cross-model critic",
         "+ Learned verifier": "+ Learned verifier", "+ Retrieval (RAG)": "+ RAG = Full framework"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


# ---------------------------------------------------------------- helpers
def style(ax, title: str = "", grid_axis: str = "x", pad: float = 6) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=7.5)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, loc="left", pad=pad)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def f1(x) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def tex(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def ci(c) -> str:
    return f"[{c[0]:.1f}, {c[1]:.1f}]"


def dci(d) -> str:
    return f"{d['diff']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]"


def table(path: str, header: list[str], rows: list[list[str]], align: str, caption: str,
          label: str, wide: bool = False) -> None:
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[t]", r"\centering", r"\footnotesize",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{3.5pt}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [(" & ".join(r) + r" \\") if r != ["MID"] else r"\midrule" for r in rows]
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


def stage_times() -> list[dict]:
    """Stage durations from the supervisor log (START/OK lines)."""
    out, start = [], {}
    pat = re.compile(r"^(\w{3} \w{3} +\d+ [\d:]+ [AP]M) IST (\d{4}) (START \[\d\]|OK) (.+)$")
    for line in (ART / "logs" / "full.supervisor.out").read_text().splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        t = datetime.strptime(f"{m.group(1)} {m.group(2)}".replace("  ", " "),
                              "%a %b %d %I:%M:%S %p %Y")
        stage = m.group(4)
        if m.group(3).startswith("START"):
            start[stage] = t
        elif stage in start:
            out.append({"stage": stage, "hours": (t - start[stage]).total_seconds() / 3600})
    return out


# ---------------------------------------------------------------- data
def collect() -> dict:
    cfg = load_config()
    nb, seed = cfg["eval"]["bootstrap"], cfg["seed"]
    ev = Evaluator(cfg, "")
    res = json.loads((ART / "eval" / "results.json").read_text())
    perq = pd.read_csv(ART / "eval" / "per_question.csv", index_col="uid")
    rows = [r for r in res["table1"] if r.get("available") and r["row"] != "Full framework"]
    m: dict = {"rows": rows, "sc_curve": {int(k): v for k, v in res["sc_curve"].items()},
               "correction": res["correction"], "contamination": res["contamination"],
               "reproduction": res["reproduction"], "verifier_test": res["verifier_test"],
               "full_vs_sc": res["full_vs_sc_matched"],
               "full_vs_sc_mcnemar": res["full_vs_sc_matched_mcnemar"],
               "pot_rag_mean": res["extra"]["pot_rag"]["overall"]}
    cot = ev.cands("cot")
    m["cot_tokens_per_sample"] = float(cot["completion_tokens"].mean())
    m["cot_truncated_pct"] = float((cot["finish"] == "length").mean() * 100)
    m["_cot_tok"] = cot["completion_tokens"].tolist()

    # significance of key steps (paired cluster bootstrap; McNemar for single-answer rows)
    cl = ev.clusters
    pairs = [("+ Code sandbox", BASE), ("+ Agentic correction (cross-model critic)", "+ Code sandbox"),
             ("+ Learned verifier", "+ Agentic correction (cross-model critic)"),
             ("+ Retrieval (RAG)", "+ Learned verifier"), ("+ Retrieval (RAG)", SC),
             ("+ Retrieval (RAG)", BASE)]
    binary = {SC, "+ Learned verifier", "+ Retrieval (RAG)"}
    sig = []
    for a, b in pairs:
        d = paired_bootstrap(perq[a], perq[b], nb, seed, cl)
        mc = mcnemar(perq[a], perq[b]) if a in binary and b in binary else None
        sig.append({"a": a, "b": b, "boot": d, "mcnemar": mc})
    m["significance"] = sig

    # PoT anatomy + RAG effect split by whether retrieval fired
    pot, rag = ev.cands("pot"), ev.cands("pot_rag")
    m["pot_anatomy"] = anatomy(pot)
    m["rag_anatomy"] = anatomy(rag)
    m["pot_truncated_pct"] = float((pot["finish"] == "length").mean() * 100)
    m["_pot_tok"] = pot["completion_tokens"].tolist()
    retr = pd.read_json(ART / "rag" / "retrieval_test.jsonl", lines=True)
    got = set(retr.loc[retr["kb_ids"].map(len) > 0, "uid"])
    pq_pot = pot.groupby("uid")["correct"].mean() * 100
    pq_rag = rag.groupby("uid")["correct"].mean() * 100
    split = {}
    for name, mask in (("with exemplars", pq_pot.index.isin(got)),
                       ("without exemplars", ~pq_pot.index.isin(got))):
        u = pq_pot.index[mask]
        split[name] = {"n": int(len(u)), "pot": float(pq_pot[u].mean()),
                       "rag": float(pq_rag[u].mean())}
    m["rag_split"] = split
    kb = json.loads((ART / "rag" / "kb" / "meta.json").read_text())
    m["kb"] = {k: kb[k] for k in ("twins", "no_solution", "entries")} | {
        "dedup_dropped": len(kb["dedup_dropped"]), "threshold": kb["threshold"],
        "retrieved": len(got)}

    # verifier: training report, ROC / calibration on the 2025 full pool, selection methods
    ver = json.loads((ART / "verifier" / "report.json").read_text())
    m["verifier"] = {k: ver[k] for k in ("n_solutions", "n_questions", "pos_rate", "configs",
                                         "dev_baselines", "primary",
                                         "primary_beats_majority_on_dev")}
    for c in m["verifier"]["configs"]:
        c.pop("full_oof", None)
    pool = ev.pool("full")
    y = pool["correct"].astype(int).to_numpy()
    vs = ev.verifiers()["all"]
    scores = {k: ev.verifier_scores(pool, k) for k in (("logreg", "all"), ("xgboost", "all"),
                                                      ("logreg", "handcrafted"))}
    m["_roc"] = {f"{a}/{b}": roc_curve(y, s)[:2] for (a, b), s in scores.items()}
    p = scores[("logreg", "all")]
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    calib = [{"p": float(p[idx == i].mean()), "obs": float(y[idx == i].mean()),
              "n": int((idx == i).sum())} for i in range(10) if (idx == i).sum() >= 15]
    m["calibration"] = calib
    m["ece"] = float(sum(abs(c["p"] - c["obs"]) * c["n"] for c in calib) / len(p))
    sel = {"Random pick (expected)": float(pool.groupby("uid")["correct"].mean().mean() * 100),
           "Majority vote": selection_accuracy(pool, None, "majority")[0] * 100}
    for (a, b), s in scores.items():
        for meth in ("max_prob", "weighted_vote"):
            sel[f"{'LogReg' if a == 'logreg' else 'XGBoost'}/{b}, {meth.replace('_', '-')}"] = \
                selection_accuracy(pool, s, meth)[0] * 100
    sel["Oracle (any correct)"] = selection_accuracy(pool, None, "oracle")[0] * 100
    m["selection_full_pool"] = sel
    lr_h = vs[("logreg", "handcrafted")]
    coef = lr_h.pipeline.named_steps["clf"].coef_[0]
    m["lr_hand_coef"] = sorted(zip(lr_h.hand_columns, map(float, coef)), key=lambda t: -abs(t[1]))
    m["stage_times"] = stage_times()
    return m


# ---------------------------------------------------------------- figures
def figures(m: dict) -> None:
    rows = m["rows"]

    # F1 Table I bars
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    style(ax)
    lab = [SHORT[r["row"]] for r in rows]
    vals = [r["overall"] for r in rows]
    cis = [r["ci95"] for r in rows]
    col = [NEUTRAL if r["row"] in (BASE, SC) else BLUE for r in rows]
    y = np.arange(len(rows))[::-1]
    err = np.array([[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]])
    ax.barh(y, vals, color=col, height=0.62, edgecolor=SURFACE)
    ax.errorbar(vals, y, xerr=err, fmt="none", ecolor=TEXT_2, elinewidth=0.8, capsize=2)
    for yi, v, c in zip(y, vals, cis):
        ax.text(c[1] + 1, yi, f"{v:.1f}", va="center", fontsize=7.3, color=TEXT)
    ax.set_yticks(y, lab, fontsize=7)
    ax.set_xlim(0, 85)
    ax.set_xlabel("Pass@1 on 2025 (%), 95% cluster-bootstrap CI", color=TEXT_2)
    save(fig, "fig_table1")

    # F2 accuracy vs compute
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    style(ax, grid_axis="both")
    tok = m["cot_tokens_per_sample"]
    sc = m["sc_curve"]
    ax.plot([n * tok / 1000 for n in sc], list(sc.values()), color=NEUTRAL, lw=1.6, marker="o",
            ms=3, label="Self-consistency (CoT, n = 1..20)")
    lbl = {"+ Code sandbox": ("PoT", (5, -8)),
           "+ Agentic correction (same-model critic)": ("+corr. same", (-4, 6)),
           "+ Agentic correction (cross-model critic)": ("+corr. cross", (-12, -13)),
           "+ Learned verifier": ("+verifier", (-34, 4)), "+ Retrieval (RAG)": ("+RAG = full", (-46, 5))}
    for r in rows:
        if r["row"] not in lbl:
            continue
        x, yv = r["cost"]["tokens"] / 1000, r["overall"]
        ax.scatter(x, yv, s=40, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=3)
        t, off = lbl[r["row"]]
        ax.annotate(t, (x, yv), textcoords="offset points", xytext=off, fontsize=6.3, color=TEXT)
    ax.scatter([], [], s=40, color=BLUE, label="Framework rows (additive)")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([2, 5, 10, 20, 50], ["2k", "5k", "10k", "20k", "50k"])
    ax.set_xlabel("Generated tokens per question (log scale)", color=TEXT_2)
    ax.set_ylabel("Accuracy (%)", color=TEXT_2)
    ax.set_ylim(38, 70)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    save(fig, "fig_compute")

    # F3 breakdown: baseline vs SC (matched) vs full
    by = {r["row"]: r["breakdown"] for r in rows}
    series = [(BASE, "Baseline CoT", NEUTRAL), (SC, "Self-consistency (matched)", ORANGE),
              ("+ Retrieval (RAG)", "Full framework", BLUE)]
    groups = [("question_type", ["Numerical", "MCQ-Single", "MCQ-Multiple", "Matching"],
               ["Num.", "MCQ-S", "MCQ-M", "Match."], "Question type"),
              ("language", ["English", "Hindi"], ["EN", "HI"], "Language"),
              ("subject", ["Mathematics", "Chemistry", "Physics"], ["Math", "Chem", "Phys"], "Subject"),
              ("requires_image", ["False", "True"], ["No", "Yes"], "Needs diagram")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True,
                             gridspec_kw={"width_ratios": [4, 2, 3, 2]})
    w = 0.27
    for ax, (key, keys, names, title) in zip(axes, groups):
        style(ax, title, "y")
        for i, (row, lab_, c) in enumerate(series):
            ax.bar(np.arange(len(keys)) + (i - 1) * w, [by[row][key][k] for k in keys], w,
                   color=c, edgecolor=SURFACE, lw=0.8, label=lab_)
        ax.set_xticks(range(len(keys)), names)
        ax.set_ylim(0, 80)
    axes[0].set_ylabel("Accuracy (%)", color=TEXT_2)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, fontsize=6.8, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.08))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.83, wspace=0.12)
    save(fig, "fig_breakdown")

    # F4 correction loop
    corr = m["correction"]
    crit = [("pot_corr_same", "Same-model (PoT)", ORANGE), ("pot_corr_cross", "Cross-model (PoT)", AQUA),
            ("full", "Cross-model (RAG-PoT)", BLUE)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [3, 2]})
    style(a1, "Critic and corrector rates (1,520 chains each)", "y", pad=26)
    mets = [("detect_rate", "Detects error\n(wrong sols.)"),
            ("false_alarm_rate", "False alarm\n(correct sols.)"),
            ("fix_rate_of_detected", "Fixed |\ndetected")]
    bw = 0.26
    for i, (k, lab_, c) in enumerate(crit):
        vals = [corr[k][mk] for mk, _ in mets]
        bars = a1.bar(np.arange(3) + (i - 1) * bw, vals, bw, color=c, edgecolor=SURFACE, lw=0.8,
                      label=lab_)
        for b, v in zip(bars, vals):
            a1.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", fontsize=6,
                    color=TEXT)
    a1.fill_between([1.55, 2.45], *PAPER_EC, color=NEUTRAL, alpha=0.45, lw=0, zorder=0,
                    label="Base paper single-pass fix rate")
    a1.set_xticks(range(3), [x for _, x in mets])
    a1.set_ylabel("%", color=TEXT_2)
    a1.set_ylim(0, 72)
    a1.legend(fontsize=6.1, frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0, 0.98))
    style(a2, "Net answer flips", "x", pad=16)
    y = np.arange(3)[::-1]
    w2r = [corr[k]["wrong_to_right"] for k, _, _ in crit]
    r2w = [corr[k]["right_to_wrong"] for k, _, _ in crit]
    a2.barh(y + 0.17, w2r, 0.32, color=BLUE, edgecolor=SURFACE, label="wrong → right")
    a2.barh(y - 0.17, r2w, 0.32, color=NEUTRAL, edgecolor=SURFACE, label="right → wrong")
    for yi, a, b in zip(y, w2r, r2w):
        a2.text(a + 1, yi + 0.17, str(a), va="center", fontsize=6.3, color=TEXT)
        a2.text(b + 1, yi - 0.17, str(b), va="center", fontsize=6.3, color=TEXT)
    a2.set_yticks(y, [x for _, x, _ in crit], fontsize=6.8)
    a2.set_xlim(0, 100)
    a2.set_xlabel("Chains", color=TEXT_2)
    a2.legend(fontsize=6.3, frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0, 0.98))
    fig.tight_layout(w_pad=2)
    save(fig, "fig_correction")

    # F5 verifier AUC: CV / dev / 2025 full pool
    cf = m["verifier"]["configs"]
    vt = m["verifier_test"]["full"]
    labs = [f"{'LR' if c['model'] == 'logreg' else 'XGB'} / {c['feature_set'][:5]}" for c in cf]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    style(ax, grid_axis="y")
    x = np.arange(len(cf))
    for off, vals, col, lab_ in (
            (-0.27, [c["cv"]["auc"] for c in cf], BLUE, "CV 2019–23"),
            (0.0, [c["dev"]["auc"] for c in cf], ORANGE, "Dev 2024"),
            (0.27, [vt[f"{c['model']}/{c['feature_set']}"]["auc"] for c in cf], AQUA, "Test 2025 (full pool)")):
        ax.bar(x + off, vals, 0.27, color=col, edgecolor=SURFACE, label=lab_)
    ax.axhline(0.5, color=TEXT_2, lw=0.8, ls=":")
    ax.set_xticks(x, labs, rotation=25, ha="right", fontsize=6.6)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC AUC", color=TEXT_2)
    ax.legend(fontsize=6.2, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_verifier_auc")

    # F6 verifier: ROC + reliability (2025 full pool)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    style(a1, "ROC on the 2025 full pool (3,040 candidates)", "both")
    for (k, (fpr, tpr)), c in zip(m["_roc"].items(), (BLUE, ORANGE, AQUA)):
        auc = m["verifier_test"]["full"][k]["auc"]
        a1.plot(fpr, tpr, color=c, lw=1.6, label=f"{k.replace('logreg', 'LogReg').replace('xgboost', 'XGBoost')} (AUC {auc:.3f})")
    a1.plot([0, 1], [0, 1], color=TEXT_2, lw=0.8, ls=":")
    a1.set_xlabel("False-positive rate", color=TEXT_2)
    a1.set_ylabel("True-positive rate", color=TEXT_2)
    a1.legend(fontsize=6.3, frameon=False, loc="lower right")
    style(a2, f"Calibration, LogReg/all (ECE {m['ece']:.3f})", "both")
    cal = m["calibration"]
    a2.plot([0, 1], [0, 1], color=TEXT_2, lw=0.8, ls=":")
    a2.plot([c["p"] for c in cal], [c["obs"] for c in cal], color=BLUE, lw=1.6, marker="o", ms=4)
    a2.set_xlabel("Predicted probability correct", color=TEXT_2)
    a2.set_ylabel("Observed fraction correct", color=TEXT_2)
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1)
    fig.tight_layout(w_pad=2)
    save(fig, "fig_verifier_roc")

    # F7 selection accuracy on the 2025 full pool
    sel = m["selection_full_pool"]
    keys = ["Random pick (expected)", "Majority vote", "LogReg/all, max-prob",
            "LogReg/all, weighted-vote", "LogReg/handcrafted, max-prob", "XGBoost/all, max-prob",
            "XGBoost/all, weighted-vote", "Oracle (any correct)"]
    names = [k.replace("LogReg/all, max-prob", "LogReg/all, max-prob (primary)") for k in keys]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    style(ax)
    y = np.arange(len(keys))[::-1]
    cols = [NEUTRAL, NEUTRAL] + [BLUE] * 5 + [NEUTRAL]
    ax.barh(y, [sel[k] for k in keys], color=cols, height=0.62)
    for yi, k in zip(y, keys):
        ax.text(sel[k] + 0.8, yi, f"{sel[k]:.1f}", va="center", fontsize=6.8, color=TEXT)
    ax.set_yticks(y, names, fontsize=6.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy of the selected answer (%)", color=TEXT_2)
    save(fig, "fig_selection")

    # F8 LogReg handcrafted coefficients (standardised features)
    co = m["lr_hand_coef"][:12][::-1]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    style(ax)
    y = np.arange(len(co))
    ax.barh(y, [c for _, c in co], color=[BLUE if c > 0 else ORANGE for _, c in co], height=0.62)
    ax.axvline(0, color=TEXT_2, lw=0.8)
    ax.set_yticks(y, [n for n, _ in co], fontsize=6.6)
    ax.set_xlabel("Coefficient (standardised feature); + = more likely correct", color=TEXT_2,
                  fontsize=6.8)
    save(fig, "fig_verifier_coef")

    # F9 RAG effect split by retrieval
    sp = m["rag_split"]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    style(ax, grid_axis="y")
    groups_ = list(sp)
    x = np.arange(len(groups_))
    for off, key, col, lab_ in ((-0.19, "pot", NEUTRAL, "PoT"), (0.19, "rag", BLUE, "RAG-PoT")):
        vals = [sp[g][key] for g in groups_]
        bars = ax.bar(x + off, vals, 0.38, color=col, edgecolor=SURFACE, label=lab_)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=6.8,
                    color=TEXT)
    ax.set_xticks(x, [f"{g}\n(n = {sp[g]['n']})" for g in groups_])
    ax.set_ylabel("Accuracy (%)", color=TEXT_2)
    ax.set_ylim(0, 75)
    ax.legend(fontsize=6.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_rag")

    # F10 contamination: CoT 1-sample accuracy by year
    c = m["contamination"]
    yrs = sorted(c["train_by_year"]) + ["2025"]
    vals = [c["train_by_year"][y_] for y_ in yrs[:-1]] + [c["test_2025_sample0"]]
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    style(ax, grid_axis="y")
    ax.bar(yrs, vals, color=[NEUTRAL] * 6 + [BLUE], width=0.62, edgecolor=SURFACE)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=6.8, color=TEXT)
    ax.axhline(c["train_2019_2024"], color=TEXT_2, lw=0.8, ls="--")
    ax.set_ylabel("CoT accuracy (%)", color=TEXT_2)
    ax.set_ylim(0, 65)
    save(fig, "fig_contamination")

    # F11 PoT anatomy (PoT vs RAG-PoT)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 1.6), sharey=True)
    style(a1, "Share of trajectories (%)")
    style(a2, "Accuracy within group (%)")
    order = ["Code executed", "Answered without code", "CoT fallback"]
    y = np.arange(len(order))[::-1]
    for off, an, col, lab_ in ((0.17, m["pot_anatomy"], NEUTRAL, "PoT"),
                               (-0.17, m["rag_anatomy"], BLUE, "RAG-PoT")):
        a1.barh(y + off, [an.get(k, {}).get("share", 0) for k in order], 0.32, color=col, label=lab_)
        a2.barh(y + off, [an.get(k, {}).get("acc", 0) for k in order], 0.32, color=col)
        for yi, k in zip(y, order):
            a1.text(an[k]["share"] + 0.8, yi + off, f"{an[k]['share']:.0f}", va="center",
                    fontsize=6.2, color=TEXT)
            a2.text(an[k]["acc"] + 0.8, yi + off, f"{an[k]['acc']:.0f}", va="center",
                    fontsize=6.2, color=TEXT)
    a1.set_yticks(y, order)
    a1.set_xlim(0, 75)
    a2.set_xlim(0, 75)
    a1.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout(w_pad=2)
    save(fig, "fig_pot_anatomy")

    # F12 token lengths
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.6), sharey=True)
    for ax, t, title, share in ((axes[0], m["_cot_tok"], "Baseline CoT (3,800 samples)",
                                 m["cot_truncated_pct"]),
                                (axes[1], m["_pot_tok"], "PoT (1,520 trajectories)",
                                 m["pot_truncated_pct"])):
        style(ax, title, "y")
        ax.hist(np.clip(t, None, 4200), bins=np.arange(0, 4300, 200), color=BLUE,
                edgecolor=SURFACE, lw=0.8)
        ax.axvline(4096, color=TEXT_2, lw=1, ls="--")
        ax.text(4000, ax.get_ylim()[1] * 0.85, f"hit cap: {share:.0f}%", ha="right",
                fontsize=6.8, color=TEXT)
        ax.set_xlabel("Completion tokens", color=TEXT_2)
    axes[0].set_ylabel("Count", color=TEXT_2)
    fig.tight_layout(w_pad=2)
    save(fig, "fig_tokens")


# ---------------------------------------------------------------- tables + macros
def tables(m: dict) -> None:
    rows = m["rows"]
    t = []
    for r in rows:
        name = SHORT[r["row"]] + (f" (n={r['n_samples']})" if r["row"] == SC else "")
        t.append([tex(name), f1(r["numerical"]), f"\\textbf{{{r['overall']:.1f}}}", ci(r["ci95"]),
                  dci(r["vs_baseline"]) if r.get("vs_baseline") else "--",
                  f"{r['cost']['tokens'] / 1000:.1f}k", f"{r['cost']['calls']:.1f}"])
    table("t_table1.tex", ["Configuration", "Numerical", "Overall", "95\\% CI",
                           "$\\Delta$ vs baseline [CI]", "Tok/q", "Calls/q"], t, "lrrcrrr",
          "SOP Table I: additive evaluation on the 2025 held-out set (190 questions, 95 EN + 95 "
          "HI). Pass@1 = mean over samples for single-candidate rows; the verifier rows select one "
          "answer per question. CIs: cluster bootstrap over EN/HI twins (2{,}000 resamples). "
          "Self-consistency uses the number of CoT samples whose tokens match the full framework.",
          "tab:table1", wide=True)

    cols = [("question_type", "Numerical", "Num."), ("question_type", "MCQ-Single", "MCQ-S"),
            ("question_type", "MCQ-Multiple", "MCQ-M"), ("question_type", "Matching", "Match."),
            ("language", "English", "EN"), ("language", "Hindi", "HI"),
            ("subject", "Mathematics", "Math"), ("subject", "Chemistry", "Chem"),
            ("subject", "Physics", "Phys"), ("requires_image", "False", "No diag."),
            ("requires_image", "True", "Diag.")]
    t = [[tex(SHORT[r["row"]])] + [f1(r["breakdown"][g][k]) for g, k, _ in cols] for r in rows]
    table("t_breakdown.tex", ["Configuration"] + [c for _, _, c in cols], t, "l" + "r" * 11,
          "Accuracy (\\%) by question type (84 / 46 / 42 / 18 questions), language, subject and "
          "diagram dependence.", "tab:breakdown", wide=True)

    t = []
    for k, lab in (("pot_corr_same", "Same-model critic, PoT source"),
                   ("pot_corr_cross", "Cross-model critic, PoT source"),
                   ("full", "Cross-model critic, RAG-PoT source")):
        c = m["correction"][k]
        stops = c["stop_reasons"]
        t.append([lab, f"{c['acc_before']:.1f} $\\to$ {c['acc_after']:.1f}", f1(c["detect_rate"]),
                  f1(c["false_alarm_rate"]), f1(c["fix_rate_of_detected"]),
                  f"{c['wrong_to_right']} / {c['right_to_wrong']}", f"{c['mean_rounds']:.2f}",
                  f"{stops.get('critic_ok', 0)} / {stops.get('unchanged', 0)} / {stops.get('max_rounds', 0)}"])
    t.append(["Base paper (single pass, EP$\\to$EC)", "", "", "", "1.1--5.2", "", "1", ""])
    table("t_correction.tex", ["Run (1{,}520 chains)", "Acc. before $\\to$ after", "Detect",
                               "False al.", "Fix$|$det.", "W$\\to$R / R$\\to$W", "Rounds",
                               "Stop: ok / unch. / max"], t, "lrrrrrrr",
          "Agentic correction. Detect = critic flags an originally wrong solution; False al. = "
          "flags a correct one; Fix$|$det. = detected wrong solutions that end correct.",
          "tab:correction", wide=True)

    v = m["verifier"]
    vt = m["verifier_test"]["full"]
    t = []
    for c in v["configs"]:
        k = f"{c['model']}/{c['feature_set']}"
        prim = (c["model"], c["feature_set"]) == (v["primary"]["model"], v["primary"]["feature_set"])
        t.append([("LR" if c["model"] == "logreg" else "XGB") + " / " + c["feature_set"]
                  + (" $^\\star$" if prim else ""),
                  f"{c['cv']['auc']:.3f}", f"{c['cv']['f1']:.3f}", f"{c['dev']['auc']:.3f}",
                  f"{c['dev']['f1']:.3f}", f"{c['dev_selection']['max_prob'] * 100:.1f}",
                  f"{vt[k]['auc']:.3f}", f"{vt[k]['f1']:.3f}", f"{vt[k]['sel_max_prob']:.1f}"])
    b = v["dev_baselines"]
    t.append(["MID"])
    t.append(["Majority vote", "", "", "", "", f"{b['majority'] * 100:.1f}", "", "",
              f"{vt['majority']:.1f}"])
    t.append(["Oracle (any correct)", "", "", "", "", f"{b['oracle'] * 100:.1f}", "", "",
              f"{vt['oracle']:.1f}"])
    table("t_verifier.tex", ["Model / features", "CV AUC", "CV F1", "Dev AUC", "Dev F1",
                             "Dev sel.", "Test AUC", "Test F1", "Test sel."], t, "lrrrrrrrr",
          f"Learned verifier trained on {v['n_solutions']:,} auto-labelled solutions of "
          f"{v['n_questions']:,} training questions ({v['pos_rate'] * 100:.1f}\\% correct). CV: "
          "GroupKFold by twin pair on 2019--2023; dev: 2024; test: the 2025 full pool (16 "
          "candidates per question). Sel.\\ = accuracy of the max-probability candidate. "
          "$^\\star$ = primary model, chosen on dev only.", "tab:verifier", wide=True)

    t = []
    for s in m["significance"]:
        mc = s["mcnemar"]
        t.append([tex(SHORT[s["a"]].replace("+ ", "")) + " vs " + tex(SHORT[s["b"]].replace("+ ", "")),
                  dci(s["boot"]), f"{s['boot']['p_le_0']:.4f}",
                  "--" if not mc else f"{mc['a_only']} / {mc['b_only']}, $p$={mc['p']:.4f}"])
    table("t_significance.tex", ["Comparison", "$\\Delta$ [95\\% CI]", "$P(\\Delta\\le0)$",
                                 "McNemar (a only / b only)"], t, "lrrr",
          "Paired cluster-bootstrap tests of key steps; McNemar where both rows give one answer "
          "per question.", "tab:sig", wide=True)

    t = []
    for pool, lab in (("pot", "8 PoT samples"), ("pot_corr_cross", "8 PoT + 8 corrected (cross)"),
                      ("full", "8 RAG-PoT + 8 corrected (full)")):
        q = m["verifier_test"][pool]
        t.append([lab, f"{q['majority']:.1f}", f"{q['logreg/all']['sel_max_prob']:.1f}",
                  f"{q['logreg/all']['sel_weighted_vote']:.1f}", f"{q['xgboost/all']['sel_max_prob']:.1f}",
                  f"{q['oracle']:.1f}", f"{q['logreg/all']['auc']:.3f}"])
    table("t_selection.tex", ["2025 candidate pool", "Majority", "LR max-p.$^\\star$",
                              "LR w.-vote", "XGB max-p.", "Oracle", "LR AUC"], t, "lrrrrrr",
          "Answer selection on the 2025 pools (accuracy \\%). $^\\star$ = primary verifier "
          "chosen on 2024. Reported only; no choice was made on 2025.", "tab:selection", wide=True)

    kb, sp = m["kb"], m["rag_split"]
    th = kb["threshold"]
    t = [["Training twin pairs (2019--2024)", str(kb["twins"])],
         ["No verified-correct solution", str(kb["no_solution"])],
         ["Near-duplicate of a 2025 question", str(kb["dedup_dropped"])],
         ["KB entries", str(kb["entries"])],
         ["Threshold $\\tau$ (tuned on 2024)", f"{th['value']:.3f}"],
         ["Dev proxy precision (base rate)", f"{th['proxy_precision'] * 100:.1f}\\% ({th['base_precision'] * 100:.1f}\\%)"],
         ["2025 questions with exemplars", f"{kb['retrieved']} / 190"],
         ["PoT $\\to$ RAG-PoT, with exemplars",
          f"{sp['with exemplars']['pot']:.1f} $\\to$ {sp['with exemplars']['rag']:.1f}"],
         ["PoT $\\to$ RAG-PoT, without exemplars",
          f"{sp['without exemplars']['pot']:.1f} $\\to$ {sp['without exemplars']['rag']:.1f}"]]
    table("t_rag.tex", ["Retrieval (full run)", "Value"], t, "lr",
          "RAG knowledge base and the effect of retrieval on PoT accuracy.", "tab:rag")

    st = m["stage_times"]
    names = {"solve --exp cot": "CoT, 20 $\\times$ 190 (16 cached)",
             "solve --exp pot": "PoT, 8 $\\times$ 190",
             "correct --exp pot_corr_same": "Correction, same-model critic",
             "correct --exp pot_corr_cross": "Correction, cross-model critic",
             "solve --exp pot_train": "PoT on train, 8 $\\times$ 1{,}270",
             "build-kb": "Build KB", "solve --exp pot_rag": "RAG-PoT, 8 $\\times$ 190",
             "correct --exp full": "Full correction (RAG-PoT)", "train-verifier": "Train verifiers",
             "solve --exp cot_train": "CoT on train, 1 $\\times$ 1{,}270", "evaluate": "Evaluate",
             "report": "Report"}
    seen, t, total = set(), [], 0.0
    for s in st:
        lab = names.get(s["stage"], tex(s["stage"]))
        key = lab + ("" if s["stage"] not in seen else " (re-run)")
        seen.add(s["stage"])
        total += s["hours"]
        if s["hours"] >= 0.05:
            t.append([key, f"{s['hours']:.1f}"])
    t.append(["\\textbf{Total}", f"\\textbf{{{total:.1f}}}"])
    table("t_compute.tex", ["Stage (one RTX 3060, 12\\,GB)", "Hours"], t, "lr",
          "Wall-clock time of the full run (supervisor log); no stage needed a retry.",
          "tab:compute")
    m["total_hours"] = total

    r = {x["row"]: x for x in rows}
    full = r["+ Retrieval (RAG)"]
    fs = m["full_vs_sc"]
    mc = m["full_vs_sc_mcnemar"]
    con = m["contamination"]
    sel = m["selection_full_pool"]
    cv = m["correction"]
    vt = m["verifier_test"]["full"]
    rep = m["reproduction"]
    mac = {
        "BaseAcc": f"{r[BASE]['overall']:.1f}", "BaseNum": f"{r[BASE]['numerical']:.1f}",
        "ScAcc": f"{r[SC]['overall']:.1f}", "ScN": str(r[SC]["n_samples"]),
        "PotAcc": f"{r['+ Code sandbox']['overall']:.1f}", "PotNum": f"{r['+ Code sandbox']['numerical']:.1f}",
        "CsAcc": f"{r['+ Agentic correction (same-model critic)']['overall']:.1f}",
        "CcAcc": f"{r['+ Agentic correction (cross-model critic)']['overall']:.1f}",
        "VerAcc": f"{r['+ Learned verifier']['overall']:.1f}",
        "FullAcc": f"{full['overall']:.1f}", "FullNum": f"{full['numerical']:.1f}",
        "FullLo": f"{full['ci95'][0]:.1f}", "FullHi": f"{full['ci95'][1]:.1f}",
        "FullDelta": f"{full['vs_baseline']['diff']:+.1f}",
        "FullTok": f"{full['cost']['tokens'] / 1000:.1f}k",
        "FullVsSc": f"{fs['diff']:+.1f}", "FullVsScLo": f"{fs['ci95'][0]:+.1f}",
        "FullVsScHi": f"{fs['ci95'][1]:+.1f}", "FullVsScP": f"{mc['p']:.3f}",
        "FullOnly": str(mc["a_only"]), "ScOnly": str(mc["b_only"]),
        "ScMax": f"{max(m['sc_curve'].values()):.1f}",
        "CotTrunc": f"{m['cot_truncated_pct']:.0f}", "PotTrunc": f"{m['pot_truncated_pct']:.0f}",
        "ReproAcc": f"{rep['pass_at_1']['pass1']:.1f}", "ReproStd": f"{rep['pass_at_1']['run_std']:.1f}",
        "ConTrain": f"{con['train_2019_2024']:.1f}", "ConTest": f"{con['test_2025_sample0']:.1f}",
        "ConDelta": f"{con['delta_2025_minus_train']:+.1f}",
        "VerN": f"{m['verifier']['n_solutions']:,}".replace(",", "{,}"),
        "VerDevAuc": f"{[c for c in m['verifier']['configs'] if c['model'] == 'logreg' and c['feature_set'] == 'all'][0]['dev']['auc']:.3f}",
        "VerTestAuc": f"{vt['logreg/all']['auc']:.3f}", "XgbTestAuc": f"{vt['xgboost/all']['auc']:.3f}",
        "SelMaj": f"{sel['Majority vote']:.1f}", "SelPrim": f"{sel['LogReg/all, max-prob']:.1f}",
        "SelXgb": f"{sel['XGBoost/all, max-prob']:.1f}", "SelOracle": f"{sel['Oracle (any correct)']:.1f}",
        "SelRandom": f"{sel['Random pick (expected)']:.1f}", "Ece": f"{m['ece']:.3f}",
        "RagMean": f"{m['pot_rag_mean']:.1f}", "KbEntries": str(m["kb"]["entries"]),
        "RagN": str(m["kb"]["retrieved"]),
        "RagWithPot": f"{m['rag_split']['with exemplars']['pot']:.1f}",
        "RagWithRag": f"{m['rag_split']['with exemplars']['rag']:.1f}",
        "RagWoPot": f"{m['rag_split']['without exemplars']['pot']:.1f}",
        "RagWoRag": f"{m['rag_split']['without exemplars']['rag']:.1f}",
        "FixSame": f"{cv['pot_corr_same']['fix_rate_of_detected']:.1f}",
        "FixCross": f"{cv['pot_corr_cross']['fix_rate_of_detected']:.1f}",
        "FixFull": f"{cv['full']['fix_rate_of_detected']:.1f}",
        "PotCodeRan": f"{m['pot_anatomy']['Code executed']['share']:.0f}",
        "PotFallback": f"{m['pot_anatomy']['CoT fallback']['share']:.0f}",
        "TotalHours": f"{m['total_hours']:.0f}",
    }
    from mmjee_reasoner.report.build import latex_prompts

    (GEN / "prompts.tex").write_text(latex_prompts(width=46))  # one IEEE column
    (GEN / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n"
                                            for k, v in mac.items()))


def main() -> None:
    m = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (FIG, GEN):
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
