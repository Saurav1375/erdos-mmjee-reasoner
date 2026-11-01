"""Figures for the final paper and the final deck (grayscale, plain labels).

Reads only existing result files (no new experiment):
  results/phase2/results.json (in the repo copy), report/final_v3/metrics.json,
  report/final_v4b/generated/macros*.tex, results/v4/w1_diagnostics.json,
  artifacts_v4b/analysis/v4b.json.
Writes report/final_paper/figures/*.{pdf,png}.

usage: .venv/bin/python scripts/final_paper_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
OUT = ROOT / "report" / "final_paper" / "figures"

K, D, M, L, XL = "#111111", "#4a4a4a", "#7f7f7f", "#b0b0b0", "#dcdcdc"
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": D,
    "axes.linewidth": 0.8, "xtick.color": K, "ytick.color": K, "axes.labelcolor": K,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42,
})


def load_macros() -> dict[str, str]:
    d = {}
    for f in sorted((ROOT / "report/final_v4b/generated").glob("macros*.tex")):
        for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", f.read_text()):
            d[k] = v.replace("$-$", "-").replace("$<$", "<").replace("{,}", ",")
    return d


def num(s: str) -> float:
    return float(re.match(r"[+-]?[\d.]+", s.strip()).group())


# deck versions: drawn at the exact size they take on the slide, so text is shown at 1:1
DECK_SIZE = {"phase2_ladder": (5.4, 3.2), "main_2026": (6.0, 2.75), "compute_2026": (5.0, 3.6),
             "breakdown_2026": (6.2, 3.0), "correction": (5.35, 2.05), "errors_2026": (5.9, 2.9),
             "selectors": (5.4, 3.0), "decomposition": (4.9, 3.1), "diagram_first": (5.4, 1.95),
             "memorisation_probe": (4.6, 2.4), "pilot_fixes": (3.05, 1.65), "dev_ab": (5.5, 2.75)}


# paper versions of one-column figures: drawn at the column width (3.5 in), so text is 1:1
PAPER_SIZE = {"phase2_ladder": (3.5, 2.3), "compute_2026": (3.5, 2.7),
              "decomposition": (3.5, 2.5), "memorisation_probe": (3.5, 2.1)}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if name in PAPER_SIZE:
        fig.set_size_inches(*PAPER_SIZE[name])
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    (OUT / "deck").mkdir(exist_ok=True)
    fig.set_size_inches(*DECK_SIZE[name])
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "deck" / f"{name}.png", dpi=250)
    plt.close(fig)


def hbars(ax, labels, vals, cis=None, colors=None, fmt="{:.1f}", xmax=None):
    y = np.arange(len(labels))[::-1]
    colors = colors or [M] * len(vals)
    err = None
    if cis:
        err = np.array([[v - c[0], c[1] - v] for v, c in zip(vals, cis)]).T
    ax.barh(y, vals, 0.62, color=colors, edgecolor=K, linewidth=0.6, xerr=err,
            error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": K})
    for yi, v, c in zip(y, vals, cis or [None] * len(vals)):
        x = (c[1] if c else v) + 1.0
        ax.text(x, yi, fmt.format(v), va="center", fontsize=8.5, color=K)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    if xmax:
        ax.set_xlim(0, xmax)


# ---------------------------------------------------------------------------------------------
def fig_phase2_ladder(p2: dict) -> None:
    rows = {r["row"]: r for r in p2["table1"]}
    order = [("Baseline (single-pass CoT)", "Single answer (CoT)"),
             ("Self-consistency (matched compute)", "Majority vote, same compute"),
             ("+ Code sandbox", "+ Code sandbox"),
             ("+ Agentic correction (cross-model critic)", "+ Agentic correction"),
             ("+ Learned verifier", "+ Learned verifier"),
             ("Full framework", "+ Retrieval = full framework")]
    vals = [rows[k]["overall"] for k, _ in order]
    cis = [rows[k]["ci95"] for k, _ in order]
    cols = [L, XL, M, M, D, K]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    hbars(ax, [n for _, n in order], vals, cis, cols, xmax=85)
    ax.set_xlabel("Accuracy on JEE Advanced 2025 (%)")
    save(fig, "phase2_ladder")


def fig_main_2026(m3: dict) -> None:
    f = {r["row"]: r for r in m3["tables"]["2026_v3"]["table1"]}
    p = {r["row"]: r for r in m3["tables"]["2026_v2fw"]["table1"]}
    groups = ["Single answer\n(CoT)", "Majority vote,\nsame compute", "Code sandbox\n(8 samples)",
              "+ Correction", "Full\nframework"]
    pk = ["Baseline (single-pass CoT)", "Self-consistency (matched compute)", "+ Code sandbox",
          "+ Agentic correction (cross-model critic)", "Full framework"]
    fk = ["Baseline (single-pass CoT, adaptive budget)", "Self-consistency (matched compute)",
          "+ Code sandbox + RAG v3 + transcript (v3 PoT)", "+ Correction (cross-model critic)",
          "v3 framework (verifier v3)"]
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    x = np.arange(len(groups))
    w = 0.38
    for off, src, keys, col, lab in [(-w / 2, p, pk, L, "Phase-2 framework (re-run unchanged)"),
                                     (w / 2, f, fk, D, "Final framework")]:
        v = [src[k]["overall"] for k in keys]
        c = [src[k]["ci95"] for k in keys]
        err = np.array([[a - b[0], b[1] - a] for a, b in zip(v, c)]).T
        ax.bar(x + off, v, w, color=col, edgecolor=K, linewidth=0.6, label=lab, yerr=err,
               error_kw={"elinewidth": 0.7, "capsize": 2, "ecolor": K})
        for xi, vi in zip(x + off, v):
            ax.text(xi, 2, f"{vi:.1f}", ha="center", va="bottom", fontsize=8,
                    color="white" if col == D else K, fontweight="bold")
    ax.set_xticks(x, groups)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 75)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.set_title("JEE Advanced 2026 (clean test, 204 questions, English + Hindi)", loc="left")
    save(fig, "main_2026")


def fig_compute_2026(m3: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for key, col, ls, lab in [("2026_v2fw", L, (0, (3, 2)), "Majority vote over n answers (Phase-2 setup)"),
                              ("2026_v3", D, (0, (5, 2)), "Majority vote over n answers (final setup)")]:
        t = m3["tables"][key]
        sc = t["sc_curve"]
        base_tok = t["table1"][0]["cost"]["tokens"]
        n = sorted(int(k) for k in sc)
        ax.plot([base_tok * i / 1000 for i in n], [sc[str(i)] for i in n], ls=ls, color=col,
                lw=1.4, label=lab)
    for key, col, full_row, lab in [("2026_v3", K, "v3 framework (verifier v3)", "Final framework"),
                                    ("2026_v2fw", M, "Full framework", "Phase-2 framework")]:
        for r in m3["tables"][key]["table1"]:
            if not r.get("available") or r["kind"] == "sc_matched":
                continue
            full = r["row"] == full_row
            ax.scatter(r["cost"]["tokens"] / 1000, r["overall"], s=70 if full else 18, color=col,
                       edgecolor=K if full else col, zorder=3, label=lab if full else None)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Generated tokens per question (thousands, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    save(fig, "compute_2026")


def fig_breakdown_2026(m3: dict) -> None:
    f = {r["row"]: r for r in m3["tables"]["2026_v3"]["table1"]}
    base = f["Baseline (single-pass CoT, adaptive budget)"]["breakdown"]
    full = f["v3 framework (verifier v3)"]["breakdown"]
    groups = [("language", "English", "English"), ("language", "Hindi", "Hindi"),
              ("requires_image", "False", "Text only"), ("requires_image", "True", "Needs diagram"),
              ("subject", "Mathematics", "Maths"), ("subject", "Chemistry", "Chemistry"),
              ("subject", "Physics", "Physics")]
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    x = np.arange(len(groups))
    w = 0.38
    for off, src, col, lab in [(-w / 2, base, XL, "Single answer (CoT)"),
                               (w / 2, full, D, "Final framework")]:
        v = [src[a][b] for a, b, _ in groups]
        ax.bar(x + off, v, w, color=col, edgecolor=K, linewidth=0.6, label=lab)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [g for _, _, g in groups])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 95)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    save(fig, "breakdown_2026")


def fig_correction(p2: dict, mc: dict) -> None:
    runs = [("2025, same critic", p2["correction"]["pot_corr_same"]),
            ("2025, cross critic", p2["correction"]["pot_corr_cross"]),
            ("2026, final", mc)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(runs))
    a1.axhspan(1.1, 5.2, color=XL, zorder=0)
    a1.text(-0.4, 21.5, "grey band: base paper, single pass (1.1-5.2%)", fontsize=7.5, color=D)
    v = [r["fix_rate_of_detected"] for _, r in runs]
    a1.bar(x, v, 0.55, color=D, edgecolor=K, linewidth=0.6)
    for xi, vi in zip(x, v):
        a1.text(xi, vi + 0.6, f"{vi:.1f}%", ha="center", fontsize=8)
    a1.set_xticks(x, [n.replace(", ", ",\n") for n, _ in runs], fontsize=7.5)
    a1.set_ylabel("Detected errors fixed (%)")
    a1.set_ylim(0, 24)
    a1.grid(axis="x", visible=False)
    w = 0.36
    a2.bar(x - w / 2, [r["wrong_to_right"] for _, r in runs], w, color=D, edgecolor=K,
           linewidth=0.6, label="wrong to right")
    a2.bar(x + w / 2, [r["right_to_wrong"] for _, r in runs], w, color=XL, edgecolor=K,
           linewidth=0.6, label="right to wrong")
    a2.set_xticks(x, [n.replace(", ", ",\n") for n, _ in runs], fontsize=7.5)
    a2.set_ylabel("Answers changed (count)")
    a2.grid(axis="x", visible=False)
    a2.legend(frameon=False, fontsize=7.5)
    save(fig, "correction")


def fig_errors_2026(w1: dict) -> None:
    e = w1["errors_2026"]
    groups = [("All", None), ("Text only", ("requires_image", "False")),
              ("Needs diagram", ("requires_image", "True")), ("English", ("language", "English")),
              ("Hindi", ("language", "Hindi"))]
    tot = sum(e["counts"].values())
    rows = []
    for name, key in groups:
        if key is None:
            c = e["counts"]
            rows.append([100 * c["correct"] / tot, 100 * c["selector_miss"] / tot,
                         100 * c["never_in_pool"] / tot])
        else:
            g = e[key[0]][key[1]]
            rows.append([g["correct"], g["selector_miss"], g["never_in_pool"]])
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    y = np.arange(len(groups))[::-1]
    left = np.zeros(len(groups))
    for j, (lab, col) in enumerate([("Correct", D), ("Right answer in pool, not picked", L),
                                    ("No right answer in pool", "white")]):
        ax.barh(y, rows[:, j], 0.6, left=left, color=col, edgecolor=K, linewidth=0.6, label=lab,
                hatch="///" if col == "white" else None)
        for yi, l0, v in zip(y, left, rows[:, j]):
            if v > 7:
                ax.text(l0 + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if col == D else K,
                        bbox=dict(fc="white", ec="none", pad=0.5) if col == "white" else None)
        left += rows[:, j]
    ax.set_yticks(y, [g for g, _ in groups])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of 2026 questions (%), final framework")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=7.5)
    save(fig, "errors_2026")


def fig_selectors(mac: dict) -> None:
    rows = [("Ranker in the final framework", num(mac["VfRef"]), K),
            ("Ranker without length features", num(mac["VfNoLen"]), M),
            ("Fine-tuned neural verifier", num(mac["VfNeural"]), M),
            ("Ranker + neural score", num(mac["VfStacked"]), M),
            ("Ranker, more matched training data", num(mac["VfXRef"]), M),
            ("Answer-level verifier", num(mac["VbAMean"]), M),
            ("Majority vote", num(mac["VfMaj"]), "white")]
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    hbars(ax, [r[0] for r in rows], [r[1] for r in rows], colors=[r[2] for r in rows], xmax=80)
    ax.axvline(num(mac["VfRef"]), color=K, ls=":", lw=1)
    ax.set_xlim(40, 72)
    ax.set_xlabel("Pick accuracy, leave-one-year-out 2019-24 (%)")
    save(fig, "selectors")


def fig_decomp(v4b: dict) -> None:
    dec = v4b["decomposition"]
    yrs = ["2025", "2026"]
    steps = [("matched_sc", "Majority vote over CoT\n(same compute)", XL),
             ("majority_v3_pool", "Majority vote over\nour candidate pool", M),
             ("v3_pick", "Learned verifier pick\nfrom our pool", K)]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    w = 0.26
    for i, (k, lab, col) in enumerate(steps):
        xs = np.arange(2) + (i - 1) * w
        v = [dec[y][k] for y in yrs]
        ax.bar(xs, v, w, color=col, edgecolor=K, linewidth=0.6, label=lab)
        for xi, vi in zip(xs, v):
            ax.text(xi, vi + 0.6, f"{vi:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(2), ["2025 (seen during design)", "2026 (clean)"])
    ax.set_ylim(45, 80)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, "decomposition")


def fig_diagram_first(v4b: dict) -> None:
    b = v4b["dev_b"]
    blocks = [("all", "All\n(204)"), ("diagram_questions", "Needs diagram\n(59)"),
              ("text_questions", "Text only\n(145)")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.4))
    x = np.arange(3)
    w = 0.36
    for ax, key, cur, new, ylab in [
            (a1, "per_sample_s4_s7", "current", "diagram", "Accuracy per answer (%)"),
            (a2, "truncated_s4_s7", "current", "diagram", "Answers cut off (%)")]:
        v1 = [b[k][key][cur] for k, _ in blocks]
        v2 = [b[k][key][new] for k, _ in blocks]
        ax.bar(x - w / 2, v1, w, color=L, edgecolor=K, linewidth=0.6, label="Current transcript")
        ax.bar(x + w / 2, v2, w, color=D, edgecolor=K, linewidth=0.6,
               label="Diagram-first transcript")
        for xi, vi in zip(np.r_[x - w / 2, x + w / 2], v1 + v2):
            ax.text(xi, vi + 1, f"{vi:.1f}", ha="center", fontsize=7.5)
        ax.set_xticks(x, [n for _, n in blocks], fontsize=7.5)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 80)
        ax.grid(axis="x", visible=False)
    a1.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "diagram_first")


def fig_probe(m3: dict) -> None:
    pr = m3["probes"]
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for key, col, mk, lab in [("probe_answer_q3", K, "o", "Qwen3-VL-8B (our solver)"),
                              ("probe_answer_q25", M, "s", "Qwen2.5-VL-7B (paper's model)")]:
        d = pr[key]["all"]
        ax.plot([r["year"] for r in d], [r["correct"] for r in d], color=col, marker=mk, ms=4,
                lw=1.2, label=lab)
    ax.axvspan(2025.5, 2026.5, color=XL, zorder=0)
    ax.text(2025.55, 23.2, "after model\nrelease", fontsize=7, color=D)
    ax.set_xlabel("Exam year")
    ax.set_ylabel("Answer-only accuracy (%)")
    ax.set_ylim(0, 27)
    ax.set_xticks(range(2019, 2027))
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    save(fig, "memorisation_probe")


def fig_pilot_fixes() -> None:
    # counts stated in the Phase-2 report (Table: method-level problems found in the pilot traces)
    items = [("NameError\nin code", 47, 0), ("Code blocked\n(scipy)", 86, 0),
             ("Critic gave\nno verdict", 76, 1)]
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    x = np.arange(len(items))
    w = 0.36
    ax.bar(x - w / 2, [i[1] for i in items], w, color=L, edgecolor=K, linewidth=0.6,
           label="Pilot, before fixes")
    ax.bar(x + w / 2, [i[2] for i in items], w, color=D, edgecolor=K, linewidth=0.6,
           label="After fixes")
    for xi, vi in zip(np.r_[x - w / 2, x + w / 2], [i[1] for i in items] + [i[2] for i in items]):
        ax.text(xi, vi + 1.5, str(vi), ha="center", fontsize=8)
    ax.set_xticks(x, [i[0] for i in items], fontsize=7.5)
    ax.set_ylabel("Count")
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 25))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=7)
    save(fig, "pilot_fixes")


def fig_devab(m3: dict) -> None:
    da = m3["dev_ab"]
    rows = [("transcript", "Transcript in the prompt"), ("extend_pot", "Continue cut-off answers (code)"),
            ("extend_cot", "Continue cut-off answers (CoT)"), ("rag_v2", "Retrieval, Phase-2 recipe"),
            ("rag_v3", "Retrieval, hybrid + JEEBench")]
    v = [da[k]["overall"]["diff"] for k, _ in rows]
    c = [da[k]["overall"]["ci95"] for k, _ in rows]
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    y = np.arange(len(rows))[::-1]
    for yi, vi, ci in zip(y, v, c):
        col = K if ci[0] > 0 else M
        ax.plot(ci, [yi, yi], color=col, lw=1.6)
        ax.plot(vi, yi, "o", color=col, ms=5)
        ax.text(ci[1] + 0.3, yi, f"{vi:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]", va="center",
                fontsize=7.5)
    ax.axvline(0, color=D, lw=0.8)
    ax.set_yticks(y, [n for _, n in rows])
    ax.set_xlim(-4, 17)
    ax.set_xlabel("Change on the 2024 dev year (points, 95% CI)")
    ax.grid(axis="y", visible=False)
    save(fig, "dev_ab")


def main() -> None:
    p2 = json.loads((REPO / "results/phase2/results.json").read_text())
    m3 = json.loads((ROOT / "report/final_v3/metrics.json").read_text())
    w1 = json.loads((REPO / "results/v4/w1_diagnostics.json").read_text())
    v4b = json.loads((ROOT / "artifacts_v4b/analysis/v4b.json").read_text())
    mac = load_macros()
    mc = m3["tables"]["2026_v3"]["correction"]["v3_full"]
    fig_phase2_ladder(p2)
    fig_main_2026(m3)
    fig_compute_2026(m3)
    fig_breakdown_2026(m3)
    fig_correction(p2, mc)
    fig_errors_2026(w1)
    fig_selectors(mac)
    fig_decomp(v4b)
    fig_diagram_first(v4b)
    fig_probe(m3)
    fig_pilot_fixes()
    fig_devab(m3)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
