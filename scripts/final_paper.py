"""Final paper: tables and macros with plain labels (no internal version names).

Copies the generated macros and tables of report/final_v4b/generated (which come from the result
files only) into report/final_paper/generated, renaming internal labels, and writes the Phase-2
tables from results/phase2 and report/phase2/metrics.json. Figures: scripts/final_paper_figures.py.

usage: .venv/bin/python scripts/final_paper_figures.py && .venv/bin/python scripts/final_paper.py
       && cd report/final_paper && tectonic final_paper.tex
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
SRC = ROOT / "report" / "final_v4b" / "generated"
GEN = ROOT / "report" / "final_paper" / "generated"

RENAME = {
    "t_table1_2026.tex": [
        ("Top: frozen v3 framework. Bottom: the frozen v2 framework re-run unchanged.",
         "Top: the frozen final framework. Bottom: the frozen Phase-2 framework, re-run unchanged."),
        (r"\emph{v3 framework}", r"\emph{Final framework}"),
        (r"\emph{v2 framework (re-run)}", r"\emph{Phase-2 framework (re-run unchanged)}"),
        ("+ Code sandbox + RAG v3 + transcript (v3 PoT)", "+ Code sandbox + retrieval + transcript"),
        ("+ Learned verifier v3 (PoT pool only)", "+ Learned verifier (8 code answers only)"),
        ("v3 framework (verifier v3)", "Final framework (16-answer pool)"),
        (r"\label{tab:t26}", r"\label{tab:t26}"),
    ],
    "t_table1_2025.tex": [
        (r"v3 Table~I on 2025 (190 questions, \textbf{post-hoc}: 2025 was used to pilot v1/v2).",
         r"Final framework on 2025 (190 questions, \textbf{post-hoc}: 2025 was the Phase-2 test "
         r"year and was used for the pilot)."),
        ("+ Code sandbox + RAG v3 + transcript (v3 PoT)", "+ Code sandbox + retrieval + transcript"),
        ("+ Learned verifier v3 (PoT pool only)", "+ Learned verifier (8 code answers only)"),
        ("v3 framework (verifier v3)", "Final framework (16-answer pool)"),
    ],
    "t_compare.tex": [
        (r"2026 v3 fw vs.\ v2 fw", r"2026 final vs.\ Phase-2 framework"),
        (r"2026 v3 PoT vs.\ v2 PoT", r"2026 final vs.\ Phase-2 PoT"),
        (r"2026 CoT adaptive budget vs.\ CoT", r"2026 CoT with continuation vs.\ CoT"),
        (r"2026 gated vs.\ ungated correction (v3)", r"2026 gated vs.\ ungated correction"),
        (r"2026 verifier v3 pick vs.\ v3 PoT mean", r"2026 verifier pick vs.\ mean PoT answer"),
        (r"2025$^\dagger$ v3 fw vs.\ v2 fw", r"2025$^\dagger$ final vs.\ Phase-2 framework"),
        (r"2025$^\dagger$ v3 PoT vs.\ v2 PoT", r"2025$^\dagger$ final vs.\ Phase-2 PoT"),
    ],
    "t_correction.tex": [
        (r"2026 pot\_corr\_cross", r"2026 Ph.\,2, PoT + corr."),
        ("2026 full &", r"2026 Ph.\,2, full &"),
        (r"2026 v3\_full", "2026 final"),
        (r"2025 v3\_full", r"2025$^\dagger$ final"),
        (r"2025 pot\_corr\_cross (v2)", r"2025 Ph.\,2, PoT + corr."),
        ("2025 full (v2)", r"2025 Ph.\,2, full"),
    ],
    "t_verifier.tex": [
        ("2026 v3 &", "2026 final &"), ("2026 v2 &", "2026 Phase-2 &"),
        ("2025 v3 &", r"2025$^\dagger$ final &"), ("2025 v2 &", "2025 Phase-2 &"),
        (r"xgb\_ranker/handcrafted", "XGBoost ranker"), (r"logreg/all", "LogReg, all feat."),
    ],
    "t_shift.tex": [
        ("train 2019-2024 pot\\_train", "Training, 2019--2024"),
        ("2026 v3\\_pot", "2026, final solver"), ("2026 pot", "2026, Phase-2 solver"),
        ("2025 v3\\_pot", "2025, final solver"), ("2025 pot", "2025, Phase-2 solver"),
    ],
    "t_devab.tex": [
        ("RAG, v2 recipe", "Retrieval, Phase-2 recipe"),
        ("RAG v3 (hybrid + JEEBench)", "Retrieval, hybrid + JEEBench"),
        ("Adaptive budget (PoT)", "Continuation (PoT)"), ("Adaptive budget (CoT)", "Continuation (CoT)"),
    ],
    "t_breakdown.tex": [
        ("Group & v2 '26 & v3 '26 & v2 '25 & v3 '25",
         r"Group & Ph.\,2 '26 & Final '26 & Ph.\,2 '25 & Final '25"),
        ("2025 is post-hoc.", "Ph.\\,2: Phase-2 framework. 2025 is post-hoc."),
    ],
    "t_gate.tex": [],
    "t_q25.tex": [],
    "t_probes.tex": [],
    "t_v4b_decomp.tex": [
        ("Where the v3 gain over", "Where the final framework's gain over"),
        ("Majority vote on the v3 pool", "Majority vote on our pool"),
        ("v3 pick (verifier v3)", "Verifier pick"),
        ("majority vote on the v3 pool", "majority vote on our pool"),
        ("selection gain: v3 pick", "selection gain: verifier pick"),
        ("Frozen v3 verifier", "Frozen verifier"),
    ],
    "t_v4b_diag.tex": [("PoT with the v3 transcript", "PoT with the current transcript")],
    "t_v4_rag.tex": [
        (r"RAG in Phase~4, paired differences", "Retrieval variants, paired differences"),
        (r"kb\_v4 $-$ kb\_v3, 2026$^\dagger$", r"Extended KB $-$ current KB, 2026$^\dagger$"),
        (r"kb\_v4 13.5\% vs.\ kb\_v3 15.0\%", r"extended KB 13.5\% vs.\ current KB 15.0\%"),
    ],
}
FORBIDDEN = re.compile(r"\bv[1-4]b?\b|\bv[234]_|Phase~4|kb\\_v|pot\\_", re.I)


def relabel() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("macros*.tex"):
        shutil.copy2(f, GEN / f.name)
    for name, pairs in RENAME.items():
        s = (SRC / name).read_text()
        for a, b in pairs:
            if a not in s:
                raise SystemExit(f"{name}: '{a}' not found")
            s = s.replace(a, b)
        body = "\n".join(l for l in s.splitlines() if not l.startswith("%"))
        if FORBIDDEN.search(body):
            raise SystemExit(f"{name}: internal label left: {FORBIDDEN.search(body).group()}")
        (GEN / name).write_text(s)


def ci(c) -> str:
    return f"{c[0]:.1f}--{c[1]:.1f}"


def table(path: str, caption: str, label: str, cols: str, head: list[str], body: list[list[str]],
          star: bool = False) -> None:
    env = "table*" if star else "table"
    lines = [rf"\begin{{{env}}}[!t]", r"\centering", r"\footnotesize", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{cols}}}",
             r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    lines += [r"\midrule" if r == ["MID"] else " & ".join(r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    (GEN / path).write_text("\n".join(lines) + "\n")


def phase2_tables() -> None:
    res = json.loads((REPO / "results/phase2/results.json").read_text())
    names = {"Baseline (single-pass CoT)": "Baseline (single-pass CoT)",
             "Self-consistency (matched compute)": r"Self-consistency (matched, $n{=}20$)",
             "+ Code sandbox": "+ Code sandbox (PoT)",
             "+ Agentic correction (same-model critic)": "+ Correction, same-model critic",
             "+ Agentic correction (cross-model critic)": "+ Correction, cross-model critic",
             "+ Learned verifier": "+ Learned verifier",
             "Full framework": r"\textbf{+ Retrieval = full framework}"}
    body = []
    for r in res["table1"]:
        if r["row"] not in names:
            continue
        acc = f"{r['overall']:.1f}"
        if r["row"] == "Full framework":
            acc = rf"\textbf{{{acc}}}"
        body.append([names[r["row"]], acc, f"{r['numerical']:.1f}", ci(r["ci95"]),
                     f"{r['cost']['tokens'] / 1000:.1f}k"])
    table("t_phase2_2025.tex", "Phase-2 framework on JEE Advanced 2025 (190 questions; the Phase-2 "
          "test year). Rows add one module at a time from the code sandbox. Accuracy in \\%, 95\\% "
          "cluster-bootstrap CI; tokens are generated tokens per question.", "tab:p2",
          "lrrcr", ["Configuration", "Acc.", "Num.", "95\\% CI", "Tok./q"], body)

    pil = json.loads((ROOT / "report/phase2/metrics.json").read_text())["pilot"]["table1"]
    pnames = {"Baseline (single-pass CoT)": "Baseline (single-pass CoT)",
              "Self-consistency (matched compute)": "Self-consistency (matched)",
              "+ Code sandbox": "+ Code sandbox (PoT)",
              "+ Agentic correction (same-model critic)": "+ Correction, same-model critic",
              "+ Agentic correction (cross-model critic)": "+ Correction, cross-model critic",
              "+ Learned verifier": "+ Learned verifier",
              "Full framework": "Full framework"}
    body = [[pnames[r["row"]], f"{r['overall']:.1f}", ci(r["ci95"])]
            for r in pil if r["row"] in pnames and r.get("available")]
    table("t_pilot.tex", "Pilot study before the fixes (40 test questions, 20 EN/HI pairs; verifier "
          "and knowledge base from 150 training questions). Accuracy in \\%.", "tab:pilot", "lrc",
          ["Configuration", "Acc.", "95\\% CI"], body)


def prompts() -> None:
    """Appendix: the prompts, taken verbatim from the code (prompts/templates.py, rag/cards.py)."""
    import sys
    import textwrap

    sys.path.insert(0, str(ROOT / "src"))
    from mmjee_reasoner.prompts import templates as T
    from mmjee_reasoner.rag.cards import CARD_PROMPT

    items = [
        ("Baseline CoT (base paper, verbatim; English, numerical question)",
         T.baseline_prompt("English", "Numerical")),
        ("Code-sandbox solver instructions (appended to the baseline prompt)",
         T.POT_INSTRUCTIONS.format(max_blocks=3)),
        ("Transcript block (final framework)", T.TRANSCRIPT_BLOCK),
        ("Transcriber", T.TRANSCRIBE_PROMPT),
        ("Translator (Hindi transcripts)", T.TRANSLATE_PROMPT),
        ("Concept tagger (retrieval query)", T.TAGGER_PROMPT),
        ("Exemplar header (retrieval)", T.EXEMPLAR_HEADER + "\n\n" + T.EXEMPLAR_TEMPLATE),
        ("Critic", T.CRITIC_PROMPT),
        ("Critic verdict cue (verdict forcing)", T.CRITIC_VERDICT_CUE.strip()),
        ("Corrector feedback (appended to the solver prompt)", T.CORRECTOR_FEEDBACK),
        ("Method card writer (Step 4)", CARD_PROMPT),
        ("Diagram-first transcriber (Step 5)", T.TRANSCRIBE_DIAGRAM_PROMPT),
        ("Memorisation probe (answer only)", T.ANSWER_ONLY_INSTRUCTION),
    ]
    out = []
    for title, text in items:
        lines = []
        for para in text.splitlines():
            lines += textwrap.wrap(para, 58) or [""]
        out.append(f"\\paragraph{{{title}}}\n\\begin{{scriptsize}}\\begin{{verbatim}}\n"
                   + "\n".join(lines) + "\n\\end{verbatim}\\end{scriptsize}\n")
    (GEN / "prompts.tex").write_text("\n".join(out))


def main() -> None:
    relabel()
    phase2_tables()
    prompts()
    print(f"wrote {GEN}")


if __name__ == "__main__":
    main()
