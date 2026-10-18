"""v3 memorisation probes and the Qwen2.5 bridge experiments (mock engine)."""

from __future__ import annotations

from mmjee_reasoner.analysis.probes import (
    completion_probe,
    ngram_overlap,
    rouge_l,
    split_half,
)
from mmjee_reasoner.config import load_experiment
from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.llm import MockEngine, make_engine
from mmjee_reasoner.llm.mock import FakeVLM, golds_for
from mmjee_reasoner.pipeline.common import load_candidates


def test_text_metrics():
    assert rouge_l("a b c d", "a b c d") == 1.0
    assert rouge_l("x y", "a b") == 0.0
    assert 0 < rouge_l("a b x d", "a b c d") < 1
    ref = "one two three four five six"
    assert ngram_overlap(ref, ref, 5) == 1.0
    assert ngram_overlap("one two three four nine six", ref, 5) == 0.0
    assert ngram_overlap("x", "short", 5) == 0.0          # reference shorter than n
    p, r = split_half("w1 w2 w3 w4 w5")
    assert p == "w1 w2" and r == "w3 w4 w5"


def test_answer_only_probe_all_years(cfg, dataset):
    from mmjee_reasoner.pipeline.solve import run_solve_stage

    qs = load_questions(cfg)
    eng = make_engine(cfg, "solver", mock=MockEngine(FakeVLM(golds_for(qs))))
    exp = load_experiment("probe_answer_q3")
    run_solve_stage(cfg, exp, eng)
    df = load_candidates(cfg, "probe_answer_q3")
    assert len(df) == len(qs) and set(df["year"]) == set(qs["year"])   # every year, 1 sample
    assert (df["source"] == "answer_only").all()


def test_answer_only_prompt_has_no_reasoning_request():
    from mmjee_reasoner.prompts.templates import answer_only_prompt, baseline_prompt

    p = answer_only_prompt("English", "Numerical")
    assert "Do NOT show any reasoning" in p and "\\boxed{}" in p
    assert "step-by-step" not in p
    assert baseline_prompt("English", "Numerical") != p


def test_completion_probe(cfg, dataset):
    qs = load_questions(cfg)
    words = " ".join(f"w{i}" for i in range(40))
    transcripts = {u: {"text": words} for u in qs["uid"]}
    eng = make_engine(cfg, "solver", mock=MockEngine(FakeVLM(golds_for(qs))))
    out = completion_probe(cfg, eng, qs, transcripts, "solver")
    assert len(out) == (qs["language"] == "English").sum()        # English only
    assert all(0.0 <= r["rouge_l"] <= 1.0 and 0.0 <= r["ngram5"] <= 1.0 for r in out)


def test_q25_correction_uses_repro_role(cfg, dataset):
    from mmjee_reasoner.pipeline.correct import run_correct_stage
    from mmjee_reasoner.pipeline.solve import run_solve_stage

    qs = load_questions(cfg)
    fake = MockEngine(FakeVLM(golds_for(qs)))
    run_solve_stage(cfg, load_experiment("q25_pot"), make_engine(cfg, "repro", mock=fake))
    recs = run_correct_stage(cfg, load_experiment("q25_corr_same"), mock_engines={"solver": fake})
    assert len(recs) == 8 * (qs["split"] == "test").sum()
    assert {r["exp"] for r in recs} == {"q25_corr_same"}
