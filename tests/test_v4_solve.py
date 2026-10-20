"""Phase 4 solver options: subset_pairs in solve (mock engine)."""

from __future__ import annotations

from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.llm import MockEngine
from mmjee_reasoner.llm.mock import FakeVLM, golds_for
from mmjee_reasoner.pipeline.common import pilot_twins


def test_subset_pairs_shared_across_year_parts(cfg, dataset):
    from mmjee_reasoner.pipeline.solve import run_solve_stage

    cfg["sandbox"]["force_answer"] = False
    keys = set(pilot_twins(load_questions(cfg), "train", 4, cfg["seed"]))
    years = sorted(load_questions(cfg, "train")["year"].unique())
    got = []
    for part in (years[:3], years[3:]):
        exp = {"name": f"sub{part[0]}", "kind": "solve", "mode": "pot", "split": "train",
               "years": [int(y) for y in part], "subset_pairs": 4, "n_samples": 1}
        qs = load_questions(cfg, "train")
        fake = MockEngine(FakeVLM(golds_for(qs)))
        recs = run_solve_stage(cfg, exp, fake)
        assert {r["year"] for r in recs} <= set(exp["years"])
        got += [r["twin_key"] for r in recs]
    assert set(got) == keys                       # the two parts cover exactly the subset
    again = set(pilot_twins(load_questions(cfg), "train", 4, cfg["seed"]))
    assert again == keys                          # deterministic


def test_limit_draws_inside_subset(cfg, dataset):
    from mmjee_reasoner.pipeline.common import select_questions, set_subset

    keys = pilot_twins(load_questions(cfg), "train", 6, cfg["seed"])
    set_subset(cfg, limit=3)
    qs = select_questions(cfg, "train", twin_keys=keys)
    assert len(qs) == 3 and set(qs["twin_key"]) <= set(keys)
