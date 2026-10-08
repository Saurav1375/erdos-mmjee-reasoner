"""Static report figures (matplotlib, PDF + PNG).

Palette: validated reference categorical slots 1-3 (blue, orange, aqua) on a
near-white surface; baseline-style references in neutral gray. Thin marks,
recessive grid, single y-axis, legends for >= 2 series plus direct labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8d8c87"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, out_dir: Path, name: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        paths.append(str(p))
    plt.close(fig)
    return paths


def table1_bars(rows: list[dict], out_dir: Path) -> list[str]:
    rows = [r for r in rows if r.get("available")]
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 0.9))
    labels = [r["row"] for r in rows][::-1]
    vals = [r["overall"] for r in rows][::-1]
    lo = [r["overall"] - r["ci95"][0] for r in rows][::-1]
    hi = [r["ci95"][1] - r["overall"] for r in rows][::-1]
    colors = [NEUTRAL if r["kind"] in ("sc_matched",) or i == 0 else SERIES[0]
              for i, r in enumerate(rows)][::-1]
    y = range(len(rows))
    ax.barh(y, vals, height=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=TEXT_2, elinewidth=0.8, capsize=2)
    for yi, v, h in zip(y, vals, hi):
        ax.text(v + h + 0.6, yi, f"{v:.1f}", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(list(y), labels, fontsize=8, color=TEXT)
    ax.set_xlabel("Pass@1 on 2025 held-out (%), 95% bootstrap CI", color=TEXT_2, fontsize=8)
    _style(ax, "Additive evaluation (SOP Table I)")
    return _save(fig, out_dir, "table1")


def per_type_bars(rows: list[dict], out_dir: Path) -> list[str]:
    pick = [r for r in rows if r.get("available") and r["row"] in (
        "Baseline (single-pass CoT)", "Self-consistency (matched compute)", "Full framework")]
    if not pick:
        return []
    types = ["Numerical", "MCQ-Single", "MCQ-Multiple", "Matching"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    width = 0.8 / len(pick)
    for i, r in enumerate(pick):
        vals = [r["breakdown"]["question_type"].get(t, 0.0) for t in types]
        xs = [j + (i - (len(pick) - 1) / 2) * width for j in range(len(types))]
        ax.bar(xs, vals, width=width * 0.92, color=SERIES[i], label=r["row"],
               edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.8, f"{v:.0f}", ha="center", fontsize=7, color=TEXT)
    ax.set_xticks(range(len(types)), types, fontsize=8, color=TEXT)
    ax.set_ylabel("Accuracy (%)", color=TEXT_2, fontsize=8)
    _style(ax, "Accuracy by question type (2025)")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    return _save(fig, out_dir, "per_type")


def sc_curve(curve: dict, refs: dict[str, float], out_dir: Path) -> list[str]:
    if not curve:
        return []
    ns = sorted(int(k) for k in curve)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(ns, [curve[n] if n in curve else curve[str(n)] for n in ns], color=SERIES[0],
            linewidth=2, marker="o", markersize=4, label="Self-consistency (CoT, majority vote)")
    for (name, val), color in zip(refs.items(), (SERIES[1], NEUTRAL)):
        ax.axhline(val, color=color, linewidth=1.2, linestyle="--", label=name)
        ax.text(ns[-1], val + 0.5, f"{name} {val:.1f}", ha="right", fontsize=7, color=TEXT)
    ax.set_xlabel("CoT samples per question (n)", color=TEXT_2, fontsize=8)
    ax.set_ylabel("Accuracy (%)", color=TEXT_2, fontsize=8)
    _style(ax, "Self-consistency scaling vs. full framework")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    return _save(fig, out_dir, "sc_curve")


def correction_bars(correction: dict, out_dir: Path) -> list[str]:
    if not correction:
        return []
    names = list(correction)
    fixes = [correction[n]["wrong_to_right"] for n in names]
    breaks = [correction[n]["right_to_wrong"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    xs = range(len(names))
    ax.bar([x - 0.2 for x in xs], fixes, width=0.36, color=SERIES[2], label="wrong -> right",
           edgecolor=SURFACE, linewidth=2)
    ax.bar([x + 0.2 for x in xs], breaks, width=0.36, color=SERIES[1], label="right -> wrong",
           edgecolor=SURFACE, linewidth=2)
    for x, f, b in zip(xs, fixes, breaks):
        ax.text(x - 0.2, f + 0.5, str(f), ha="center", fontsize=7, color=TEXT)
        ax.text(x + 0.2, b + 0.5, str(b), ha="center", fontsize=7, color=TEXT)
    ax.set_xticks(list(xs), names, fontsize=8, color=TEXT)
    ax.set_ylabel("Solutions", color=TEXT_2, fontsize=8)
    _style(ax, "Answer transitions caused by agentic correction")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, out_dir, "correction")


def verifier_bars(configs: list[dict], out_dir: Path) -> list[str]:
    if not configs:
        return []
    labels = [f"{c['model']} / {c['feature_set']}" for c in configs]
    cv = [c["cv"]["auc"] for c in configs]
    dev = [c["dev"]["auc"] for c in configs]
    fig, ax = plt.subplots(figsize=(6.4, 0.4 * len(configs) + 1.0))
    y = list(range(len(configs)))
    ax.barh([v - 0.18 for v in y], cv, height=0.34, color=SERIES[0], label="CV 2019-2023",
            edgecolor=SURFACE, linewidth=2)
    ax.barh([v + 0.18 for v in y], dev, height=0.34, color=SERIES[1], label="Dev 2024",
            edgecolor=SURFACE, linewidth=2)
    for yi, a, b in zip(y, cv, dev):
        ax.text(a + 0.005, yi - 0.18, f"{a:.3f}", va="center", fontsize=7, color=TEXT)
        ax.text(b + 0.005, yi + 0.18, f"{b:.3f}", va="center", fontsize=7, color=TEXT)
    ax.set_yticks(y, labels, fontsize=8, color=TEXT)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC AUC", color=TEXT_2, fontsize=8)
    _style(ax, "Learned verifier: classifier comparison")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    return _save(fig, out_dir, "verifier_auc")
