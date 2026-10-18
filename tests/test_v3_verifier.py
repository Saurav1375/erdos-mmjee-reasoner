"""Verifier v3: pool assembly, chain features, LOYO training, gate (mock data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmjee_reasoner.config import load_experiment
from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.llm import MockEngine, make_engine
from mmjee_reasoner.llm.mock import FakeVLM, golds_for
from mmjee_reasoner.verifier.v3 import gate_choices, gate_outcome, pool_agreement, prepare_pool


def _row(uid, sample, source, pred, correct, **kw):
    return {"uid": uid, "sample": sample, "source": source, "pred": pred,
            "question_type": "MCQ-Single", "correct": correct, **kw}


def test_prepare_pool_copies_chain_info():
    df = pd.DataFrame([
        _row("q", 0, "pot", "A", True), _row("q", 1, "pot", "B", False),
        _row("q", 0, "corrected", "A", True, corr_critic_flagged_first=False, corr_rounds=1,
             corr_n_flags=0, corr_answer_changed=False, correction_failed_last=False),
        _row("q", 1, "corrected", "A", True, corr_critic_flagged_first=True, corr_rounds=2,
             corr_n_flags=1, corr_answer_changed=True, correction_failed_last=False),
        _row("r", 0, "pot", "C", False),                       # question without chains
    ])
    p = prepare_pool(df)
    o1 = p[(p.uid == "q") & (p["sample"] == 1) & (p.source == "pot")].iloc[0]
    assert o1["corr_critic_flagged_first"] == 1.0 and o1["corr_rounds"] == 2.0
    assert o1["has_chain"] == 1.0 and o1["is_corrected"] == 0.0
    r = p[p.uid == "r"].iloc[0]
    assert r["has_chain"] == 0.0 and r["corr_rounds"] == 0.0
    a = pool_agreement(p)
    # q: answers A,B | A,A -> A has 3 of 4; the original "B" agrees with no corrected answer
    assert a.loc[o1.name, "other_half_agree"] == 0.0
    assert a.loc[p[(p.uid == "q") & (p.source == "corrected")].index[0], "full_agree"] == 2 / 3
    assert a.loc[r.name, "other_half_agree"] == -1.0


def test_gate_outcome():
    df = prepare_pool(pd.DataFrame([
        _row("q", 0, "pot", "B", False), _row("q", 1, "pot", "A", True),
        _row("q", 0, "corrected", "A", True, corr_rounds=1),
        _row("q", 1, "corrected", "B", False, corr_rounds=1),
    ]))
    s = np.array([0.2, 0.9, 0.8, 0.3])          # trust the fix of sample 0, not of sample 1
    g = gate_outcome(df, s, 0.0)
    assert g["acc_original"] == 50.0 and g["acc_corrected"] == 50.0 and g["acc_gated"] == 100.0
    assert g["right_to_wrong_gated"] == 0 and g["wrong_to_right_gated"] == 1
    assert gate_outcome(df, s, 0.7)["accepted"] == 0      # high delta keeps every original
    ch = gate_choices(df, s, 0.0)                         # per-chain view used by `gated` rows
    assert list(ch["taken"]) == [True, False] and ch["final_correct"].mean() == 1.0
    assert ch["final_correct"].mean() * 100 == g["acc_gated"]


def test_train_verifier_v3_and_evaluate(cfg, dataset, fake_embedder):
    from mmjee_reasoner.pipeline.common import load_candidates
    from mmjee_reasoner.pipeline.correct import run_correct_stage
    from mmjee_reasoner.pipeline.solve import run_solve_stage
    from mmjee_reasoner.verifier.v3 import train_verifier_v3, verifier_v3_dir

    cfg["verifier"]["dir_name"] = "verifier_v3"
    qs = load_questions(cfg)
    fake = MockEngine(FakeVLM(golds_for(qs), p_correct=0.5))
    solver = make_engine(cfg, "solver", mock=fake)
    run_solve_stage(cfg, load_experiment("pot_train"), solver)
    run_correct_stage(cfg, load_experiment("train_corr_cross"),
                      mock_engines={"solver": fake, "critic_cross": fake})
    rep = train_verifier_v3(cfg)
    assert rep["primary"]["model"] in ("logreg", "xgboost", "xgb_ranker")
    assert rep["primary"]["selection"] in ("max_prob", "weighted_vote", "log_odds_vote")
    assert -0.3 <= rep["gate"]["delta"] <= 0.3
    nl = rep["gate"]["nested_loyo"]
    assert nl["n_chains"] == rep["gate"]["chosen"]["n_chains"] and nl["deltas_by_year"]
    assert set(rep["configs"][0]["sel_16pool_by_year"]) <= set(range(2019, 2025))
    assert (verifier_v3_dir(cfg) / "models" / "xgb_ranker__all.joblib").exists()

    # The Evaluator scores a test pool through predict_pool.
    run_solve_stage(cfg, load_experiment("pot"), solver)
    run_correct_stage(cfg, load_experiment("pot_corr_cross"),
                      mock_engines={"solver": fake, "critic_cross": fake})
    from mmjee_reasoner.pipeline.evaluate import Evaluator

    ev = Evaluator(cfg)
    pool = ev.pool("pot_corr_cross")
    scores = ev.verifier_scores(pool)
    assert len(scores) == len(pool) and np.all((scores >= 0) & (scores <= 1))
    assert len(load_candidates(cfg, "pot_corr_cross")) * 2 == len(pool)
    # `gated` Table I row: one gated answer per chain, delta from the trained gate.
    ser, extra = ev.row_scores({"kind": "gated", "exp": "pot_corr_cross"})
    assert ser.between(0, 1).all() and len(ser) == pool["uid"].nunique()
    assert extra["gate"]["n_chains"] == len(load_candidates(cfg, "pot_corr_cross"))
    assert extra["gate"]["delta"] == rep["gate"]["delta"]
    # matched self-consistency is sized with the SC row's own run, not a hardcoded `cot`
    assert ev.matched_samples("pot_corr_cross", "pot") >= 1
    assert ev.matched_samples("pot_corr_cross", "no_such_exp") is None
