"""Data-side adapter that makes HF rows usable by the verbatim upstream scorer.

The upstream ``is_answer_correct`` reads two columns that only existed in the
authors' private CSV (``jee_advanced_combined_fixed.csv``):

* ``expanded_answer`` -- truthy when a Numerical gold is a set of accepted values
* ``acceptable_values`` -- a Python literal; the prediction is accepted iff
  ``float(pred) in eval(acceptable_values)``

The public HF dataset stores such golds as strings such as ``"8.70 TO 9.10"``,
``"80 OR 150 OR 220"`` or ``"[75 to 79] OR [94 to 95]"``. Without these columns
every Numerical row would raise ``KeyError`` inside the scorer and score False.

We rebuild the two columns here. A range is expanded to all values on a
``step`` grid (0.001 by config; JEE asks for 2 decimals, the finer grid also
accepts in-range 3-decimal answers) between its
endpoints, and alternatives (``OR``) are unioned. Plain numeric golds get
``expanded_answer = False`` so that the upstream tolerance branch applies.
The scoring functions themselves are untouched.
"""

from __future__ import annotations

import math
import re

_NUM = r"-?\d+(?:\.\d+)?"
_RANGE_RE = re.compile(rf"^\s*\[?\s*({_NUM})\s*(?:\s+to\s+({_NUM}))?\s*\]?\s*$", re.IGNORECASE)
_PLAIN_RE = re.compile(rf"^\s*{_NUM}\s*$")


def is_plain_number(answer: str) -> bool:
    return bool(_PLAIN_RE.match(str(answer)))


def parse_numeric_gold(answer: str) -> list[tuple[float, float]]:
    """Parse a Numerical gold into a list of closed intervals ``(lo, hi)``.

    ``"3"`` -> [(3, 3)]; ``"0.7 to 0.8"`` -> [(0.7, 0.8)];
    ``"[0.5] OR [3.13 TO 3.15]"`` -> [(0.5, 0.5), (3.13, 3.15)].
    Raises ``ValueError`` for anything else.
    """
    parts = re.split(r"\s+or\s+", str(answer).strip(), flags=re.IGNORECASE)
    intervals = []
    for part in parts:
        m = _RANGE_RE.match(part)
        if not m:
            raise ValueError(f"unparseable numerical gold: {answer!r}")
        a = float(m.group(1))
        b = float(m.group(2)) if m.group(2) is not None else a
        intervals.append((min(a, b), max(a, b)))
    return intervals


def expand_intervals(intervals: list[tuple[float, float]], step: float = 0.001) -> list[float]:
    """All grid values (rounded to the step's decimals) inside the intervals, sorted."""
    decimals = max(0, -int(math.floor(math.log10(step))))
    values: set[float] = set()
    for lo, hi in intervals:
        start = math.ceil(round(lo / step, 6))
        stop = math.floor(round(hi / step, 6))
        for i in range(start, stop + 1):
            values.add(round(i * step, decimals))
        # Endpoints are always accepted, even if they are off-grid.
        values.add(lo)
        values.add(hi)
    return sorted(values)


def adapter_columns(question_type: str, answer: str, step: float = 0.001) -> dict:
    """Return ``{"expanded_answer": bool, "acceptable_values": str}`` for one row."""
    if question_type != "Numerical" or is_plain_number(answer):
        return {"expanded_answer": False, "acceptable_values": ""}
    values = expand_intervals(parse_numeric_gold(answer), step)
    return {"expanded_answer": True, "acceptable_values": repr(values)}
