"""v3 dev A/B helpers: pairing, paired differences and changed-row statistics."""

from __future__ import annotations

import pandas as pd
import pytest

from mmjee_reasoner.analysis.devab import changed_rows, compare, pair


def _rows(correct, text, tokens, finish, pred=None):
    n = len(correct)
    return pd.DataFrame({
        "uid": ["q1", "q1", "q2", "q2"][:n], "sample": [0, 1, 0, 1][:n],
        "correct": correct, "pred": pred or ["1"] * n, "text": text,
        "completion_tokens": tokens, "finish": finish, "twin_key": ["t1", "t1", "t2", "t2"][:n],
        "language": ["English", "English", "Hindi", "Hindi"][:n],
        "requires_image": [False, False, True, True][:n],
    })


def test_pair_compare_and_changed_rows():
    base = _rows([True, False, False, False], ["a", "b", "c", "d"], [10, 4096, 4096, 20],
                 ["stop", "length", "length", "stop"], pred=["1", "", "", "2"])
    arm = _rows([True, True, False, False], ["a", "b+", "c+", "d"], [10, 5000, 8192, 20],
                ["stop", "stop", "length", "stop"], pred=["1", "1", "", "2"])
    m = pair(base, arm)
    r = compare(m, n_boot=200)
    assert r["acc_base"] == pytest.approx(25.0) and r["acc_arm"] == pytest.approx(50.0)
    assert r["diff"] == pytest.approx(25.0) and r["n_questions"] == 2
    c = changed_rows(m)
    assert c["n_changed"] == 2 and c["wrong_to_right"] == 1 and c["right_to_wrong"] == 0
    assert c["answered_base"] == 0 and c["answered_arm"] == 1 and c["still_cut_off"] == 1
    assert c["extra_tokens_total"] == (5000 - 4096) + (8192 - 4096)


def test_pair_rejects_unmatched_arm_rows():
    base = _rows([True, False], ["a", "b"], [1, 1], ["stop", "stop"])
    arm = _rows([True, False, True], ["a", "b", "c"], [1, 1, 1], ["stop"] * 3)
    with pytest.raises(ValueError):
        pair(base, arm)
