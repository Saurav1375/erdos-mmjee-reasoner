"""Final report v4 = the frozen v3 report + a post-hoc "Phase 4" section (V4_PREREG.md).

Reads Phase-4 artifacts only (artifacts_v4/: analysis/w1.json, analysis/rag_sweep_kb4.json,
verifier_v4/selectors_{base,extra}/report.json, verifier_v4/neural_*/oof_*.parquet,
eval/v4_results.json) and writes report/final_v4/: metrics_v4.json, generated/macros_v4.tex,
generated/t_v4_*.tex and figures/v4_selectors.pdf. The v3 generated tables, macros and figures
are copied unchanged from report/final_v3/, so the v3 sections of the report stay identical.

usage: .venv/bin/python scripts/final_report_v3.py   (once, already done for v3)
       .venv/bin/python scripts/final_report_v4.py && cd report/final_v4 && tectonic final_report_v4.tex
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from final_report_v3 import dtex, tex_escape  # noqa: E402

A4 = ROOT / "artifacts_v4"
V3 = ROOT / "report" / "final_v3"
OUT = ROOT / "report" / "final_v4"
FIG, GEN = OUT / "figures", OUT / "generated"
VARIANTS = {"v3_ref": "v3 ranker (refit)", "no_length": "S2 no length",
            "neural": "S3 neural", "stacked": "S4 stacked"}
METHOD = {"max_prob": "max-prob", "weighted_vote": "weighted", "log_odds_vote": "log-odds"}


def load(p: str) -> dict:
    return json.loads((A4 / p).read_text())


def neural_auc(tag: str, with_extra: bool) -> dict:
    from sklearn.metrics import roc_auc_score

    from mmjee_reasoner.config import CONFIG_DIR, load_config
    from mmjee_reasoner.verifier.neural import load_oof

    cfg = load_config(*[CONFIG_DIR / f for f in ("v3.yaml", "v3_final.yaml", "v4.yaml")])
    df, s = load_oof(cfg, tag, with_extra)
    y = df["correct"].astype(int).to_numpy()
    by = {int(yr): roc_auc_score(y[df["year"] == yr], s[df["year"] == yr])
          for yr in sorted(df["year"].unique())}
    return {"all": roc_auc_score(y, s), "min": min(by.values()), "max": max(by.values())}


def collect() -> dict:
    ev = load("eval/v4_results.json")
    return {"w1": load("analysis/w1.json"), "sweep": load("analysis/rag_sweep_kb4.json"),
            "sel": {t: load(f"verifier_v4/selectors_{t}/report.json") for t in ("base", "extra")},
            "auc": {"base": neural_auc("base", False), "extra": neural_auc("extra", True)},
            "eval": ev}


def best(tab: dict, kind: str = "v3_16pool") -> float:
    return tab[kind][tab["best_method"]]


def macros(m: dict) -> dict[str, str]:
    w, sel, ev = m["w1"], m["sel"], m["eval"]
    vm = w["verifier_vs_majority"]
    d = {}
    for y, k in ((2025, "TwentyFive"), (2026, "TwentySix")):
        b = vm[str(y)]["verifier_minus_majority"]
        d[f"VfVerMaj{k}"] = dtex(b)
        d[f"VfOracle{k}"] = f"{vm[str(y)]['oracle']:.1f}"
        d[f"VfTok{k}"] = f"{w['shift'][str(y)]['mean_tokens'] / 1000:.1f}k"
        d[f"VfSmd{k}"] = f"{w['shift'][f'smd_train_to_{y}']['log_tokens']:.2f}"
    d["VfTokTrain"] = f"{w['shift']['train']['mean_tokens'] / 1000:.1f}k"
    e = w["errors_2026"]
    d["VfErrMiss"], d["VfErrNever"] = str(e["counts"]["selector_miss"]), str(e["counts"]["never_in_pool"])
    d["VfErrShare"] = f"{100 * e['selector_miss_right_answer_share']:.0f}"
    d["VfImgMiss"] = f"{e['requires_image']['True']['selector_miss']:.1f}"
    d["VfTxtMiss"] = f"{e['requires_image']['False']['selector_miss']:.1f}"
    b26 = w["by_truncation"]["2026"]["0-25"]
    d["VfFinVer"], d["VfFinMaj"] = f"{b26['verifier']:.1f}", f"{b26['majority']:.1f}"
    mp = w["mixed_pools"]["2026"]
    d["VfMixed"], d["VfMixRag"], d["VfMixPot"] = (f"{mp[k]:.1f}" for k in ("mixed_4_4", "pot_rag_8",
                                                                         "pot_8"))
    base, extra = sel["base"]["variants"], sel["extra"]["variants"]
    d["VfRef"] = f"{best(base['v3_ref']):.1f}"
    d["VfNoLen"], d["VfNeural"], d["VfStacked"] = (f"{best(base[v]):.1f}"
                                                   for v in ("no_length", "neural", "stacked"))
    d["VfMaj"] = f"{sel['base']['baselines']['v3_16pool']['majority']:.1f}"
    d["VfOracle"] = f"{sel['base']['baselines']['v3_16pool']['oracle']:.1f}"
    d["VfXRef"] = f"{best(extra['v3_ref']):.1f}"
    d["VfXNeural"] = f"{best(extra['neural']):.1f}"
    d["VfXNoLen"] = f"{best(extra['no_length']):.1f}"
    d["VfXRows"] = f"{sel['extra']['pools']['v4']:,}".replace(",", "{,}")
    d["VfAuc"] = f"{m['auc']['base']['all']:.3f}"
    d["VfAucLo"], d["VfAucHi"] = f"{m['auc']['base']['min']:.3f}", f"{m['auc']['base']['max']:.3f}"
    rd = ev["rag_dev"]
    d["VfFired"], d["VfDevQ"] = str(rd["fired"]), str(rd["questions"])
    d["VfCardSol"] = dtex(rd["card_vs_sol"]["fired"])
    d["VfCardSolP"] = f"{rd['card_vs_sol']['fired']['p_le_0']:.2f}"
    sw = m["sweep"]
    d["VfCovThree"], d["VfCovFour"] = (f"{100 * sw[k]['coverage']:.1f}" for k in ("kb_v3", "kb_v4"))
    d["VfOlyHits"] = str(sw["kb_v4"]["hits_by_source"].get("olympiadbench", 0))
    d["VfOlyEntries"] = str(m["sweep"]["kb_v4"]["entries"] - m["sweep"]["kb_v3"]["entries"])
    k4 = ev["kb4_2026"]
    d["VfKbChanged"] = str(k4["changed"])
    d["VfKbAll"] = dtex(k4["all"])
    d["VfKbAcc"] = f"{k4['all']['a']:.1f}"
    s26 = ev["selectors"]["2026"]
    d["VfSZero"], d["VfSOne"] = f"{s26['S0_v3']['acc']:.1f}", f"{s26['S1_majority']['acc']:.1f}"
    return d


def tex_table(path: str, caption: str, label: str, cols: str, head: list[str],
              body: list[list[str]]) -> None:
    lines = [r"\begin{table}[!t]", r"\centering", r"\scriptsize", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\setlength{\tabcolsep}{2.5pt}", rf"\begin{{tabular}}{{{cols}}}",
             r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    lines += [r"\midrule" if r == ["MID"] else " & ".join(r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / path).write_text("\n".join(lines) + "\n")


def write_tex(m: dict) -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "macros_v4.tex").write_text(
        "% generated by scripts/final_report_v4.py - do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros(m).items()))
    base, extra = m["sel"]["base"], m["sel"]["extra"]
    body = []
    for v, name in VARIANTS.items():
        tb, tx = base["variants"][v], extra["variants"][v]
        gate = "ref." if v == "v3_ref" else ("pass" if tb["gate_pass"] or tx["gate_pass"] else "fail")
        body.append([tex_escape(name), f"{best(tb):.1f}", f"{best(tx):.1f}",
                     f"{best(tx, 'v4_pool'):.1f}", METHOD[tb["best_method"]], gate])
    body += [["MID"], ["Majority vote", f"{base['baselines']['v3_16pool']['majority']:.1f}",
                       f"{extra['baselines']['v3_16pool']['majority']:.1f}",
                       f"{extra['baselines']['v4_pool']['majority']:.1f}", "", ""],
             ["Oracle", f"{base['baselines']['v3_16pool']['oracle']:.1f}",
              f"{extra['baselines']['v3_16pool']['oracle']:.1f}",
              f"{extra['baselines']['v4_pool']['oracle']:.1f}", "", ""]]
    tex_table("t_v4_selectors.tex",
              "Phase~4 selectors, LOYO selection accuracy (\\%) on 2019--2024 (no test data). "
              "\\emph{v3 pools}: the 16-answer training pools of verifier v3; \\emph{+W2}: "
              "retrained with the new v3-style pools (S5), evaluated on the v3 pools and on the "
              "new pools. Each run uses its best of max-prob / weighted / log-odds vote on the v3 "
              "pools (Method: base run). Gate (fixed in advance): beat the v3 ranker on every "
              "pool kind.",
              "tab:v4sel", "lrrrlr",
              ["Selector", "v3 pools", "+W2: v3", "+W2: new", "Method", "Gate"], body)
    rd, sw, k4 = m["eval"]["rag_dev"], m["sweep"], m["eval"]["kb4_2026"]
    rows = [["Solutions $-$ no RAG", dtex(rd["sol_vs_pot"]["all"]),
             dtex(rd["sol_vs_pot"]["fired"])],
            ["Cards $-$ no RAG", dtex(rd["card_vs_pot"]["all"]),
             dtex(rd["card_vs_pot"]["fired"])],
            ["\\textbf{Cards $-$ solutions}", dtex(rd["card_vs_sol"]["all"]),
             dtex(rd["card_vs_sol"]["fired"])],
            ["MID"],
            ["kb\\_v4 $-$ kb\\_v3, 2026$^\\dagger$",
             dtex(k4["all"]), dtex(k4["changed_only"])]]
    tex_table("t_v4_rag.tex",
              f"RAG in Phase~4, paired differences (points, 95\\% cluster-bootstrap CI). Top: "
              f"2024 dev year with matched seeds, {rd['questions']} questions, RAG fires on "
              f"{rd['fired']}. Bottom: post-hoc 2026 PoT with the extended KB ($^\\dagger$right "
              f"column: questions whose retrieval changed). LOYO coverage at proxy precision "
              f"0.70: kb\\_v4 {100 * sw['kb_v4']['coverage']:.1f}\\% vs.\\ kb\\_v3 "
              f"{100 * sw['kb_v3']['coverage']:.1f}\\%.",
              "tab:v4rag", "lrr", ["Comparison", "All", "RAG fires"], rows)


def figure(m: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    base, extra = m["sel"]["base"], m["sel"]["extra"]
    names = list(VARIANTS)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.bar(x - 0.18, [best(base["variants"][v]) for v in names], 0.36, label="v3 pools",
           color="#4c72b0")
    ax.bar(x + 0.18, [best(extra["variants"][v]) for v in names], 0.36, label="+ W2 pools",
           color="#dd8452")
    bl = base["baselines"]["v3_16pool"]
    ax.axhline(bl["majority"], ls="--", c="grey", lw=1)
    ax.axhline(best(base["variants"]["v3_ref"]), ls=":", c="k", lw=1)
    ax.set_xticks(x, ["v3 ranker", "no length", "neural", "stacked"], fontsize=7)
    ax.set_ylim(48, 66)
    ax.set_ylabel("LOYO pick accuracy (%)", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"v4_selectors.{ext}", dpi=200)
    plt.close(fig)


def copy_v3() -> None:
    for sub in ("generated", "figures"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for f in (V3 / sub).iterdir():
            shutil.copy2(f, OUT / sub / f.name)


def main() -> None:
    copy_v3()
    m = collect()
    (OUT / "metrics_v4.json").write_text(json.dumps(m, indent=2, default=float))
    write_tex(m)
    figure(m)
    print(f"wrote {OUT / 'metrics_v4.json'}, {GEN}/, {FIG}/")


if __name__ == "__main__":
    main()
