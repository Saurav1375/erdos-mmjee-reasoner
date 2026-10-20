"""Phase 4 (W4c): OlympiadBench loader filters (no download; synthetic records)."""

from __future__ import annotations

import pandas as pd

from mmjee_reasoner.data.external import olympiad_rows, plain_number


def _rec(i, ans, multiple="False", modality="Text-only", ctx="nan"):
    return {"id": i, "question": f"Q{i}", "final_answer": [ans] if ans is not None else [],
            "context": ctx, "is_multiple_answer": multiple, "modality": modality}


def test_plain_number():
    assert plain_number("$2022.2$") == "2022.2" and plain_number("12") == "12"
    assert plain_number("$1.5 \\times 10^{-3}$") is None and plain_number("$x^2$") is None
    assert plain_number("0.001") is None and plain_number("0") == "0"


def test_olympiad_rows_filters_and_caps():
    phys = pd.DataFrame([_rec(1, "$3.5$", ctx="Constants: g"), _rec(2, "$x$"),
                         _rec(3, "4", multiple="True"), _rec(4, "5", modality="Multimodal")])
    maths = pd.DataFrame([_rec(10 + i, str(i + 1)) for i in range(5)])
    df = olympiad_rows({"Physics": phys, "Mathematics": maths}, cap=3, seed=0)
    assert len(df) == 3 and (df["subject"] == "Physics").sum() == 1   # physics first, then maths
    p = df[df["subject"] == "Physics"].iloc[0]
    assert p["answer"] == "3.5" and p["question_text"].startswith("Constants: g")
    assert set(df["split"]) == {"external"} and set(df["year"]) == {0}
    assert set(df["source_dataset"]) == {"olympiadbench"} and set(df["question_type"]) == {"Numerical"}
