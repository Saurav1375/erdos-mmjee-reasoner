"""Build ``report/REPORT.md``, ``report/latex/*.tex`` and figures from evaluation artifacts.

All numbers are read from ``artifacts/eval/results*.json``, the verifier report
and the KB metadata; nothing is typed by hand. Rows that have not been run yet
are shown as "--".
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from mmjee_reasoner.config import PROJECT_ROOT, artifacts_dir
from mmjee_reasoner.prompts import templates as T
from mmjee_reasoner.report import figures

log = logging.getLogger(__name__)


def _fmt(x, nd=1) -> str:
    if x is None:
        return "--"
    try:
        if x != x:  # NaN
            return "--"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}"


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def collect(cfg: dict, suffix: str) -> dict:
    a = Path(cfg["paths"]["artifacts"])
    return {
        "results": _load_json(a / "eval" / f"results{suffix}.json"),
        "verifier": _load_json(a / f"verifier{suffix}" / "report.json"),
        "kb": _load_json(a / "rag" / f"kb{suffix}" / "meta.json"),
        "data": _load_json(a / "data" / "prepare_report.json"),
    }


# ----------------------------------------------------------------------------- markdown
def md_table1(rows: list[dict]) -> str:
    lines = ["| Configuration | Numerical Acc. | Overall Pass@1 | 95% CI | Δ vs baseline "
             "(pts) | Gen. tokens / Q | Calls / Q |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("available"):
            lines.append(f"| {r['row']} | -- | -- | -- | -- | -- | -- |")
            continue
        vs = r.get("vs_baseline")
        delta = f"{vs['diff']:+.1f} [{vs['ci95'][0]:+.1f}, {vs['ci95'][1]:+.1f}]" if vs else "--"
        name = r["row"]
        if r.get("kind") == "sc_matched":
            name += f" (n={r['n_samples']}" + (", INSUFFICIENT samples" if r.get(
                "insufficient") else "") + ")"
        lines.append(
            f"| {name} | {_fmt(r['numerical'])} | {_fmt(r['overall'])} | "
            f"[{_fmt(r['ci95'][0])}, {_fmt(r['ci95'][1])}] | {delta} | "
            f"{_fmt(r['cost']['tokens'], 0) if r.get('cost') else '--'} | "
            f"{_fmt(r['cost']['calls'], 1) if r.get('cost') else '--'} |")
    return "\n".join(lines)


def md_breakdown(rows: list[dict], col: str) -> str:
    rows = [r for r in rows if r.get("available")]
    if not rows:
        return "_not available_"
    cats = sorted({k for r in rows for k in r["breakdown"][col]})
    lines = ["| Configuration | " + " | ".join(cats) + " |",
             "|---|" + "---|" * len(cats)]
    for r in rows:
        lines.append(f"| {r['row']} | " + " | ".join(
            _fmt(r["breakdown"][col].get(c)) for c in cats) + " |")
    return "\n".join(lines)


def md_verifier(ver: dict | None, results: dict | None) -> str:
    if not ver:
        return "_Verifier not trained yet._"
    out = [f"Training data: {ver['n_solutions']} solutions of {ver['n_questions']} questions "
           f"(2019–2024), positive rate {ver['pos_rate']:.3f}.", "",
           "| Model | Features | CV AUC | CV AP | CV F1 | Dev AUC | Dev F1 | Dev sel. "
           "(max-prob) | Dev sel. (weighted vote) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c in ver["configs"]:
        out.append(f"| {c['model']} | {c['feature_set']} | {_fmt(c['cv']['auc'], 3)} | "
                   f"{_fmt(c['cv']['ap'], 3)} | {_fmt(c['cv']['f1'], 3)} | "
                   f"{_fmt(c['dev']['auc'], 3)} | {_fmt(c['dev']['f1'], 3)} | "
                   f"{_fmt(c['dev_selection']['max_prob'] * 100)} | "
                   f"{_fmt(c['dev_selection']['weighted_vote'] * 100)} |")
    b = ver["dev_baselines"]
    out += ["", f"Dev (2024) selection baselines: majority vote {_fmt(b['majority'] * 100)}%, "
                f"random pick {_fmt(b['random_expected'] * 100)}%, oracle (any correct) "
                f"{_fmt(b['oracle'] * 100)}%.",
            f"Primary configuration (chosen on dev only): **{ver['primary']['model']} / "
            f"{ver['primary']['feature_set']} / {ver['primary']['selection']}**.", "",
            "Top features of the primary model: " + ", ".join(
                f"`{n}` ({v:.3f})" for n, v in ver.get("primary_importance", [])[:10])]
    vt = (results or {}).get("verifier_test") or {}
    for exp, d in vt.items():
        out += ["", f"**2025 pool `{exp}`** (reported only; not used for any choice): "
                    f"majority vote {_fmt(d['majority'])}%, oracle {_fmt(d['oracle'])}%.", "",
                "| Variant | AUC | AP | F1 | Sel. max-prob | Sel. weighted vote |",
                "|---|---|---|---|---|---|"]
        for k, m in d.items():
            if isinstance(m, dict):
                out.append(f"| {k} | {_fmt(m['auc'], 3)} | {_fmt(m['ap'], 3)} | "
                           f"{_fmt(m['f1'], 3)} | {_fmt(m['sel_max_prob'])} | "
                           f"{_fmt(m['sel_weighted_vote'])} |")
    return "\n".join(out)


def md_correction(results: dict | None) -> str:
    corr = (results or {}).get("correction") or {}
    if not corr:
        return "_Correction runs not available yet._"
    lines = ["| Run | Acc. before | Acc. after | Detect (wrong flagged) | False alarm | "
             "Fix rate of detected | wrong→right | right→wrong | Mean rounds |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, c in corr.items():
        lines.append(f"| {k} | {_fmt(c['acc_before'])} | {_fmt(c['acc_after'])} | "
                     f"{_fmt(c['detect_rate'])} | {_fmt(c['false_alarm_rate'])} | "
                     f"{_fmt(c['fix_rate_of_detected'])} | {c['wrong_to_right']} | "
                     f"{c['right_to_wrong']} | {_fmt(c['mean_rounds'], 2)} |")
    return "\n".join(lines)


def md_kb(kb: dict | None) -> str:
    if not kb:
        return "_Knowledge base not built yet._"
    t = kb["threshold"]
    return (f"KB entries: {kb['entries']} (of {kb['twins']} training questions; "
            f"{kb['no_solution']} had no verified-correct solution; "
            f"{len(kb['dedup_dropped'])} removed as near-duplicates of 2025 questions). "
            f"Similarity threshold τ = {t['value']:.3f} ({t['method']}; dev coverage "
            f"{_fmt(t.get('coverage', 0) * 100)}%, proxy precision "
            f"{_fmt(t.get('proxy_precision', 0) * 100)}% vs. base "
            f"{_fmt(t.get('base_precision', 0) * 100)}%).")


def build_markdown(cfg: dict, d: dict, figs: list[str]) -> str:
    res = d["results"] or {}
    rows = res.get("table1", [])
    m = cfg["models"]
    full_sc = res.get("full_vs_sc_matched")
    parts = [
        "# mmJEE-Reasoner — results report",
        "",
        "_Auto-generated by `mmjee report` from `artifacts/`. All accuracies are measured on "
        "the 2025 held-out subset (190 questions: 95 English + 95 Hindi) with the base "
        "paper's verbatim scoring functions._",
        "",
        "## Setup",
        f"- Solver / Corrector / tagger: `{m['solver']['hf_id']}` (frozen, vLLM).",
        f"- Cross-model critic: `{m['critic_cross']['hf_id']}`.",
        f"- Sampling: temperature {cfg['sampling']['temperature']}, top-p "
        f"{cfg['sampling']['top_p']}, top-k {cfg['sampling']['top_k']}, max "
        f"{cfg['sampling']['max_tokens']} generated tokens per trajectory.",
        f"- Correction: ≤{cfg['correction']['max_rounds']} rounds; stop when the critic "
        f"finds no error or the answer is unchanged for "
        f"{cfg['correction']['unchanged_patience']} rounds.",
        f"- Verifier pool: 8 PoT samples + their corrected versions; RAG top-"
        f"{cfg['rag']['top_k']} exemplars.",
        "",
        "## Table I — additive evaluation (2025 held-out)",
        md_table1(rows) if rows else "_Not evaluated yet._",
        "",
    ]
    if full_sc:
        parts += [f"Full framework vs. compute-matched self-consistency: "
                  f"{full_sc['diff']:+.1f} points (95% CI [{full_sc['ci95'][0]:+.1f}, "
                  f"{full_sc['ci95'][1]:+.1f}]; McNemar p = "
                  f"{res['full_vs_sc_matched_mcnemar']['p']:.3f}).", ""]
    parts += ["## Breakdown by question type", md_breakdown(rows, "question_type"), "",
              "## Breakdown by language", md_breakdown(rows, "language"), "",
              "## Breakdown by subject", md_breakdown(rows, "subject"), "",
              "## Self-consistency curve (CoT)",
              ", ".join(f"n={k}: {_fmt(v)}" for k, v in (res.get("sc_curve") or {}).items())
              or "_n/a_", "",
              "## Agentic correction analysis", md_correction(res), "",
              "## Learned verifier", md_verifier(d["verifier"], res), "",
              "## Retrieval knowledge base", md_kb(d["kb"]), ""]
    con = res.get("contamination")
    if con:
        parts += ["## Contamination check (paper Table 7 protocol)",
                  f"Baseline CoT, 1 sample: 2019–2024 = {con['train_2019_2024']:.1f}% "
                  f"(n={con['n_train']}), 2025 = {con['test_2025_sample0']:.1f}% "
                  f"(n={con['n_test']}); Δ = {con['delta_2025_minus_train']:+.1f} points. "
                  "By year: " + ", ".join(f"{y}: {v:.1f}" for y, v in
                                          con["train_by_year"].items()), ""]
    rep = res.get("reproduction")
    if rep:
        p = rep["pass_at_1"]
        parts += ["## Reproduction check (base paper protocol)",
                  f"`{rep['model']}` with the paper's verbatim prompt and scorer: Pass@1 "
                  f"**{p['run_mean']:.1f} ± {p['run_std']:.1f}%** over {p['runs']} runs "
                  f"(95% CI [{rep['ci95'][0]:.1f}, {rep['ci95'][1]:.1f}]); the paper reports "
                  f"{rep['paper_pass1']} ± {rep['paper_std']}% (Table 3, 2025 set). Differences can "
                  f"come from 4-bit AWQ weights and unreported sampling settings.", ""]
    extra = res.get("extra", {})
    if "pot_rag" in extra:
        parts += [f"RAG-PoT without correction/verifier (mean over samples): "
                  f"{_fmt(extra['pot_rag']['overall'])}%.", ""]
    if d.get("data"):
        dd = d["data"]
        parts += ["## Data notes",
                  f"- Split sizes: {dd['counts']}.",
                  f"- Numerical golds expanded for the scorer (ranges / alternatives): "
                  f"{dd['expanded_numerical_golds']}.",
                  f"- Retyped MCQ-Single → MCQ-Multiple (metadata error): "
                  f"{len(dd['retyped_mcq_single_to_multiple'])} rows.",
                  f"- Duplicate question ids disambiguated: {dd['duplicate_question_ids']}.", ""]
    if figs:
        parts += ["## Figures"] + [f"![{Path(f).stem}]({Path(f).parent.name}/{Path(f).name})"
                                   for f in figs if f.endswith(".png")]
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------------- latex
def _tex(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(ch, ch) for ch in s)


def latex_table1(rows: list[dict]) -> str:
    body = []
    for r in rows:
        name = _tex(r["row"])
        if not r.get("available"):
            body.append(f"{name} & -- & -- & -- \\\\")
            continue
        if r.get("kind") == "sc_matched":
            name += f" ($n{{=}}{r['n_samples']}$)"
        vs = r.get("vs_baseline")
        delta = f"{vs['diff']:+.1f}" if vs else "--"
        body.append(f"{name} & {_fmt(r['numerical'])} & {_fmt(r['overall'])} & {delta} \\\\")
    return ("\\begin{table}[t]\\centering\\caption{Additive evaluation on the 2025 held-out "
            "subset (Pass@1, \\%). $\\Delta$: points vs.\\ single-pass CoT.}"
            "\\label{tab:main}\n\\begin{tabular}{lccc}\\toprule\n"
            "Configuration & Num. & Overall & $\\Delta$ \\\\\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\\end{tabular}\\end{table}\n")


def latex_verifier(ver: dict | None) -> str:
    if not ver:
        return "% verifier not trained\n"
    body = [f"{c['model']} & {c['feature_set']} & {_fmt(c['cv']['auc'], 3)} & "
            f"{_fmt(c['dev']['auc'], 3)} & {_fmt(c['dev']['f1'], 3)} & "
            f"{_fmt(max(c['dev_selection'].values()) * 100)} \\\\" for c in ver["configs"]]
    b = ver["dev_baselines"]
    return ("\\begin{table}[t]\\centering\\caption{Verifier comparison (CV on 2019--2023, "
            f"dev = 2024). Dev majority vote: {_fmt(b['majority'] * 100)}\\%.}}"
            "\\label{tab:verifier}\n\\begin{tabular}{llcccc}\\toprule\n"
            "Model & Feat. & CV AUC & Dev AUC & Dev F1 & Dev sel. \\\\\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\\end{tabular}\\end{table}\n")


def latex_prompts(width: int = 70) -> str:
    items = [
        ("Baseline (base paper, English)", T.baseline_prompt("English", "Numerical")),
        ("PoT solver instructions", T.POT_INSTRUCTIONS.format(max_blocks=3)),
        ("Critic", T.CRITIC_PROMPT),
        ("Corrector feedback", T.CORRECTOR_FEEDBACK),
        ("Concept tagger", T.TAGGER_PROMPT),
        ("Transcriber", T.TRANSCRIBE_PROMPT),
        ("Exemplar header", T.EXEMPLAR_HEADER),
    ]
    out = []
    for title, text in items:
        out.append(f"\\paragraph{{{_tex(title)}}}\n\\begin{{small}}\\begin{{verbatim}}\n"
                   + _wrap(text, width) + "\n\\end{verbatim}\\end{small}\n")
    return "\n".join(out)


def _wrap(text: str, width: int = 70) -> str:
    import textwrap

    lines = []
    for para in text.splitlines():
        lines += textwrap.wrap(para, width) or [""]
    return "\n".join(lines)


def latex_results_macros(d: dict) -> str:
    res = d["results"] or {}
    rows = {r["row"]: r for r in res.get("table1", []) if r.get("available")}

    def val(row, key="overall"):
        return _fmt(rows[row][key]) if row in rows else "--"

    macros = {
        "BaselineAcc": val("Baseline (single-pass CoT)"),
        "SCAcc": val("Self-consistency (matched compute)"),
        "PoTAcc": val("+ Code sandbox"),
        "FullAcc": val("Full framework"),
        "BaselineNum": val("Baseline (single-pass CoT)", "numerical"),
        "PoTNum": val("+ Code sandbox", "numerical"),
        "FullNum": val("Full framework", "numerical"),
        "SCn": str(rows["Self-consistency (matched compute)"]["n_samples"])
        if "Self-consistency (matched compute)" in rows else "--",
    }
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n"


def build_latex(cfg: dict, d: dict, latex_dir: Path) -> None:
    latex_dir.mkdir(parents=True, exist_ok=True)
    res = d["results"] or {}
    (latex_dir / "generated_tables.tex").write_text(
        latex_table1(res.get("table1", [])) + "\n" + latex_verifier(d["verifier"]))
    (latex_dir / "generated_macros.tex").write_text(latex_results_macros(d))
    (latex_dir / "generated_prompts.tex").write_text(latex_prompts())
    template = Path(__file__).with_name("main_template.tex")
    target = latex_dir / "main.tex"
    if not target.exists():  # never overwrite hand-edited prose
        shutil.copy(template, target)


def build_report(cfg: dict, limit: int | None = None) -> Path:
    from mmjee_reasoner.pipeline.common import set_subset, subset_suffix

    if limit is not None:
        set_subset(cfg, limit=limit)
    suffix = subset_suffix(cfg)
    d = collect(cfg, suffix)
    report_dir = Path(cfg["paths"]["report"])
    fig_dir = report_dir / f"figures{suffix}"
    figs: list[str] = []
    res = d["results"] or {}
    rows = res.get("table1", [])
    figs += figures.table1_bars(rows, fig_dir)
    figs += figures.per_type_bars(rows, fig_dir)
    refs = {}
    for r in rows:
        if r.get("available") and r["row"] in ("Full framework", "Baseline (single-pass CoT)"):
            refs[r["row"]] = r["overall"]
    figs += figures.sc_curve(res.get("sc_curve") or {}, refs, fig_dir)
    figs += figures.correction_bars(res.get("correction") or {}, fig_dir)
    figs += figures.verifier_bars((d["verifier"] or {}).get("configs", []), fig_dir)
    md_path = report_dir / f"REPORT{suffix}.md"
    md_path.write_text(build_markdown(cfg, d, figs), encoding="utf-8")
    if not suffix:  # the LaTeX report is only built from full runs
        build_latex(cfg, d, report_dir / "latex")
    log.info("report written: %s (+ LaTeX in %s)", md_path, report_dir / "latex")
    _ = artifacts_dir, PROJECT_ROOT  # (kept for API symmetry)
    return md_path
