"""Answer extraction and scoring.

All correctness decisions in this project go through the verbatim upstream
functions in :mod:`mmjee_reasoner.scoring.upstream`. This module only adapts
call signatures (plain dicts instead of ``pd.Series``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from mmjee_reasoner.scoring.upstream import UpstreamExtractor, UpstreamScorer

_EXTRACTOR = UpstreamExtractor()
_SCORER = UpstreamScorer()

# Columns the upstream scorer reads from a question row.
SCORING_COLUMNS = ("question_type", "answer", "expanded_answer", "acceptable_values")


def extract_answer(response_text: str, question_type: str) -> str:
    """Upstream ``extract_answer`` (first ``\\boxed{}`` + type-specific cleanup)."""
    return _EXTRACTOR.extract_answer(response_text, question_type)


def is_correct(predicted_answer: str, question: Mapping[str, Any]) -> bool:
    """Upstream ``is_answer_correct`` on a question row (dict or Series).

    ``question`` must contain the adapter columns ``expanded_answer`` and
    ``acceptable_values`` (see :mod:`mmjee_reasoner.data.adapter`).
    """
    row = question if isinstance(question, pd.Series) else pd.Series(
        {k: question[k] for k in SCORING_COLUMNS}
    )
    return bool(_SCORER.is_answer_correct(predicted_answer, row))


def score_response(response_text: str, question: Mapping[str, Any]) -> tuple[str, bool]:
    """Extract the answer from a full model response and score it."""
    pred = extract_answer(response_text, question["question_type"])
    return pred, is_correct(pred, question)


def calculate_statistics(run_accuracies: list[float]) -> dict:
    """Upstream ``calculate_statistics`` over per-run accuracies (in %)."""
    return _SCORER.calculate_statistics([{"accuracy": a} for a in run_accuracies])
