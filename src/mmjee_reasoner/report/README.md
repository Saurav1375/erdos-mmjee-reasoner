# Report builder (`report/`)

`mmjee report` builds `report/REPORT.md`, `report/latex/*.tex` and `report/figures/*` from the
evaluation artifacts. Every number is read from `artifacts/eval/results*.json`, the verifier
report and the KB metadata; nothing is typed by hand. Rows that have not been run yet show
"--".

| File | Contents |
|---|---|
| `build.py` | Markdown tables (Table I, breakdowns, verifier, correction, KB) and LaTeX tables, macros and prompt listings |
| `figures.py` | Matplotlib figures: Table I bars, per-type accuracy, SC curve, verifier AUC, correction |
| `main_template.tex` | LaTeX skeleton for the generated report |

The longer Phase-2 and final reports are built by
[`scripts/phase2_report.py`](../../../scripts/phase2_report.py) and
[`scripts/final_report.py`](../../../scripts/final_report.py). They write to `report/phase2/`
and `report/final/`, and compile to PDF with `tectonic`.
