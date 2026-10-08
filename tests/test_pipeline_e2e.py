"""End-to-end dry run of every stage with the fake VLM on a synthetic dataset."""

from __future__ import annotations

import json

import pandas as pd

from mmjee_reasoner.config import load_experiment
from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.llm import MockEngine, make_engine
from mmjee_reasoner.llm.mock import FakeVLM, golds_for
from mmjee_reasoner.pipeline.common import load_candidates


def test_full_pipeline_with_mock(cfg, dataset, fake_embedder):
    from mmjee_reasoner.pipeline.correct import run_correct_stage
    from mmjee_reasoner.pipeline.evaluate import run_evaluate_stage
    from mmjee_reasoner.pipeline.solve import run_solve_stage
    from mmjee_reasoner.rag.kb import build_kb
    from mmjee_reasoner.rag.tagger import run_tag_stage, run_transcribe_stage
    from mmjee_reasoner.report.build import build_report
    from mmjee_reasoner.verifier.train import train_verifier

    cfg["verifier"]["cv_folds"] = 3
    cfg["rag"]["threshold"] = 0.0
    fake = MockEngine(FakeVLM(golds_for(load_questions(cfg)), p_correct=0.5))
    solver = make_engine(cfg, "solver", mock=fake)

    qs = load_questions(cfg)
    qs = qs[qs["split"] != "excluded"]
    run_tag_stage(cfg, qs, solver)
    run_transcribe_stage(cfg, qs, solver)

    for name in ("cot", "pot", "pot_train", "cot_train"):
        exp = load_experiment(name)
        run_solve_stage(cfg, exp, solver)
    pot_train = load_candidates(cfg, "pot_train")
    assert set(pot_train["year"]) <= set(cfg["data"]["train_years"])
    assert len(pot_train) == 8 * (qs["split"] == "train").sum()
    assert pot_train["n_code_ok"].gt(0).all()          # the fake model always writes code

    meta = build_kb(cfg)
    assert meta["entries"] > 0
    run_solve_stage(cfg, load_experiment("pot_rag"), solver)
    rag = load_candidates(cfg, "pot_rag")
    assert rag["exemplars"].map(len).gt(0).any()
    assert rag["exemplar_text"].dropna().str.contains("Worked solution").all()

    mocks = {"solver": fake, "critic_cross": fake}
    for name in ("pot_corr_same", "pot_corr_cross", "full"):
        run_correct_stage(cfg, load_experiment(name), mock_engines=mocks)
    full = load_candidates(cfg, "full")
    assert {"orig_correct", "corr_rounds", "corr_stop_reason"} <= set(full.columns)
    assert (full["completion_tokens"] >= rag["completion_tokens"].values).all()

    report = train_verifier(cfg)
    assert report["primary"]["selection"] in ("max_prob", "weighted_vote")

    results = run_evaluate_stage(cfg)
    rows = {r["row"]: r for r in results["table1"]}
    assert all(r["available"] for r in rows.values()), [k for k, r in rows.items()
                                                        if not r["available"]]
    sc = rows["Self-consistency (matched compute)"]
    assert sc["n_samples"] >= 1
    base = rows["Baseline (single-pass CoT)"]
    assert 0 <= base["overall"] <= 100 and base["n_questions"] == (qs["split"] == "test").sum()
    assert "correction" in results and "verifier_test" in results
    con = results["contamination"]
    assert con["n_train"] == (qs["split"] == "train").sum() and con["n_test"] == base["n_questions"]
    # SC@1 equals the mean accuracy of sample 0
    cot = load_candidates(cfg, "cot")
    test_cot = cot[cot["uid"].isin(qs.loc[qs.split == "test", "uid"])]
    sc1 = results["sc_curve"][1]
    assert abs(sc1 - test_cot[test_cot["sample"] == 0]["correct"].mean() * 100) < 1e-6

    md = build_report(cfg)
    text = md.read_text()
    assert "Table I" in text and "Baseline (single-pass CoT)" in text
    per_q = pd.read_csv(f"{cfg['paths']['artifacts']}/eval/per_question.csv")
    assert len(per_q) == (qs["split"] == "test").sum()
    with open(f"{cfg['paths']['artifacts']}/eval/results.json") as f:
        json.load(f)


def test_cache_makes_reruns_free(cfg, dataset):
    from mmjee_reasoner.pipeline.solve import run_solve_stage

    calls = []

    def fn(req):
        calls.append(1)
        return FakeVLM(golds_for(load_questions(cfg)))(req)

    exp = load_experiment("pot")
    exp["n_samples"] = 2
    run_solve_stage(cfg, exp, make_engine(cfg, "solver", mock=MockEngine(fn)), limit=3)
    n_first = len(calls)
    first = load_candidates(cfg, "pot", ".limit3")
    run_solve_stage(cfg, exp, make_engine(cfg, "solver", mock=MockEngine(fn)), limit=3)
    assert len(calls) == n_first  # everything (incl. code outputs) came from the caches
    second = load_candidates(cfg, "pot", ".limit3")
    assert first["text"].tolist() == second["text"].tolist()
