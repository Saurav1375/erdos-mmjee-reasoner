"""Paper-style metacognition metrics (v3)."""

from __future__ import annotations

import pandas as pd

from mmjee_reasoner.analysis.metacog import chain_outcomes, metacog_summary, pass_at_k


def _q():
    return pd.DataFrame([{"uid": "u1", "question_type": "MCQ-Single", "answer": "A",
                          "expanded_answer": "A", "acceptable_values": None}])


def _chain(cid, answers, flags):
    return {"cid": cid, "uid": "u1",
            "solutions": [{"answer": a} for a in answers],
            "critiques": [{"has_error": f} for f in flags]}


def test_chain_outcomes_and_summary():
    chains = [
        _chain("w_fixed_r1", ["B", "A"], [True, False]),         # wrong, detected, fixed r1
        _chain("w_fixed_r2", ["B", "C", "A"], [True, True]),     # wrong, fixed only in round 2
        _chain("w_missed", ["B"], [False]),                      # wrong, not detected
        _chain("r_ok", ["A"], [False]),                          # right, no alarm
        _chain("r_broken", ["A", "D"], [True, False]),           # right, false alarm, broken
    ]
    df = chain_outcomes(chains, _q()).set_index("cid")
    assert df.loc["w_fixed_r1", "r1_correct"] and not df.loc["w_fixed_r2", "r1_correct"]
    assert df.loc["w_fixed_r2", "final_correct"] and not df.loc["w_missed", "flagged_r1"]
    s = metacog_summary(df.reset_index())
    assert s["n_wrong"] == 3 and s["n_right"] == 2
    assert abs(s["EP"] - 200 / 3) < 1e-9                         # 2 of 3 wrong detected
    assert s["EC_r1_of_detected"] == 50.0 and s["EC_final_of_detected"] == 100.0
    assert abs(s["EP_to_EC_r1"] - 100 / 3) < 1e-9
    assert s["false_alarm"] == 50.0
    assert s["right_to_wrong_r1"] == 1 and s["right_to_wrong_final"] == 1


def test_pass_at_k():
    df = pd.DataFrame({"uid": ["a"] * 4 + ["b"] * 4,
                       "correct": [1, 0, 0, 0] + [0, 0, 0, 0]})
    assert pass_at_k(df, 1) == 12.5                               # (1/4 + 0) / 2
    assert pass_at_k(df, 4) == 50.0                               # a always, b never
    assert abs(pass_at_k(df, 2) - 100 * (0.5 / 2)) < 1e-9          # a: 1 - C(3,2)/C(4,2) = 0.5
