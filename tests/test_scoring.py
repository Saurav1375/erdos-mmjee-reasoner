"""Verbatim reuse of the upstream scorer + adapter behaviour."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from mmjee_reasoner.config import load_config
from mmjee_reasoner.data.adapter import adapter_columns, expand_intervals, parse_numeric_gold
from mmjee_reasoner.scoring import calculate_statistics, extract_answer, is_correct, upstream

UPSTREAM = [
    ("eval/self-improvement/gemma3_27b_epec.ipynb", 4, ["extract_answer"]),
    ("eval/eval_test_1_acc/acc_test.ipynb", 3, ["is_answer_correct", "calculate_statistics"]),
]


def _functions(src: str) -> dict[str, str]:
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n, padded=True)
            for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_vendored_functions_are_byte_identical_to_notebooks():
    repo = Path(load_config()["paths"]["upstream_repo"])
    if not repo.exists():
        pytest.skip("upstream repo not available")
    vendored = _functions(Path(upstream.__file__).read_text(encoding="utf-8"))
    for rel, cell, names in UPSTREAM:
        nb = json.loads((repo / rel).read_text(encoding="utf-8"))
        original = _functions("".join(nb["cells"][cell]["source"]))
        for name in names:
            assert vendored[name] == original[name], f"{name} differs from {rel}"


def q(qtype: str, answer: str) -> dict:
    return {"question_type": qtype, "answer": answer, **adapter_columns(qtype, answer)}


@pytest.mark.parametrize("text,qtype,expected", [
    ("so \\boxed{B}", "MCQ-Single", "B"),
    ("\\boxed{C, A}", "MCQ-Multiple", "AC"),
    ("\\boxed{ 2.50 }", "Numerical", "2.50"),
    ("\\boxed{(D)}", "Matching", "D"),
    ("first \\boxed{1} then \\boxed{2}", "Numerical", "1"),       # upstream: FIRST box
    ("\\boxed{\\frac{1}{2}}", "Numerical", "1"),                   # upstream: nested braces
    ("no box\nfinal line 42", "Numerical", "42"),                  # last-line fallback
])
def test_extract_answer_upstream_behaviour(text, qtype, expected):
    assert extract_answer(text, qtype) == expected


def test_scoring_types():
    assert is_correct("B", q("MCQ-Single", "B"))
    assert not is_correct("C", q("MCQ-Single", "B"))
    assert is_correct("CA", q("MCQ-Multiple", "AC"))
    assert not is_correct("A", q("MCQ-Multiple", "AC"))          # no partial credit
    assert is_correct("A", q("Matching", "A"))
    assert is_correct("100.5", q("Numerical", "100"))            # 1% relative tolerance
    assert not is_correct("102", q("Numerical", "100"))
    assert is_correct("0.505", q("Numerical", "0.5"))            # 0.01 absolute below 1
    assert not is_correct("0.52", q("Numerical", "0.5"))


def test_range_and_or_golds_via_adapter():
    rng = q("Numerical", "8.70 TO 9.10")
    assert rng["expanded_answer"]
    for pred in ("8.70", "8.7", "8.9", "9.10", "9.1"):
        assert is_correct(pred, rng), pred
    for pred in ("8.69", "9.11", "abc"):
        assert not is_correct(pred, rng), pred
    alt = q("Numerical", "80 OR 150 OR 220")
    assert is_correct("150", alt) and is_correct("220.0", alt) and not is_correct("151", alt)
    neg = q("Numerical", "[0.23 TO 0.25] OR [-0.23 TO -0.25]")
    assert is_correct("-0.24", neg) and is_correct("0.23", neg) and not is_correct("0.3", neg)


def test_missing_adapter_columns_make_numerical_fail():
    """Documents why the adapter exists: without it the upstream scorer returns False."""
    import pandas as pd

    from mmjee_reasoner.scoring import _SCORER

    row = pd.Series({"question_type": "Numerical", "answer": "4"})
    assert _SCORER.is_answer_correct("4", row) is False


ALL_FORMATS = ['0.14 TO 0.16', '0.2 to 0.3', '0.83 OR 0.84', '1991 TO 2053', '2 OR 4 OR 6',
               '3.2 OR 3.90', '85018 TO 85138.02', '9 OR 13 OR 14',
               '[-2640.00 to -2620.00] or [-5280.00 to -5240.00]',
               '[-29.95 to -29.8] OR [29.8 to 29.95]', '[0.23 TO 0.25] OR [-0.23 TO -0.25]',
               '[0.5] OR [3.13 TO 3.15]', '[75 to 79] OR [94 to 95]']


@pytest.mark.parametrize("gold", ALL_FORMATS)
def test_every_range_format_parses_and_endpoints_score(gold):
    intervals = parse_numeric_gold(gold)
    row = q("Numerical", gold)
    for lo, hi in intervals:
        assert is_correct(repr(lo), row) and is_correct(repr(hi), row)
        assert is_correct(f"{(lo + hi) / 2:.2f}", row) or (hi - lo) < 0.02


def test_expand_grid():
    assert expand_intervals([(0.14, 0.16)], 0.01) == [0.14, 0.15, 0.16]
    assert 0.155 in expand_intervals([(0.14, 0.16)])  # default 0.001 grid
    assert expand_intervals([(2, 2)]) == [2.0]


def test_calculate_statistics_upstream():
    s = calculate_statistics([10.0, 20.0, 30.0])
    assert s["mean_accuracy"] == pytest.approx(20.0)
    assert s["num_runs"] == 3


@pytest.mark.dataset
def test_all_real_golds_score_themselves(real_cfg):
    from mmjee_reasoner.data.dataset import load_questions

    df = load_questions(real_cfg)
    num = df[df["question_type"] == "Numerical"]
    for _, row in num.iterrows():
        ivs = parse_numeric_gold(row["answer"])
        assert is_correct(repr(ivs[0][0]), row), row["uid"]
    for _, row in df[df["question_type"] != "Numerical"].iterrows():
        assert is_correct(row["answer"], row), row["uid"]
