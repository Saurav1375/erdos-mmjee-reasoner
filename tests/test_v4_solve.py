"""Phase 4 solver options: subset_pairs in solve, rag.matched_seeds (mock engine)."""

from __future__ import annotations

from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.llm import MockEngine
from mmjee_reasoner.llm.mock import FakeVLM, golds_for
from mmjee_reasoner.pipeline.common import pilot_twins


def _seeds(cfg, qs, exemplars):
    from mmjee_reasoner.pipeline.solve import solve_pot

    fake = MockEngine(FakeVLM(golds_for(qs)))
    solve_pot(cfg, {"name": "t", "n_samples": 2}, fake, qs, exemplars)
    out = [(c.meta["rid"].split("|")[1], c.sampling["seed"], c.messages[0]["content"][-1]["text"])
           for c in fake.calls if "turn" in c.meta and c.meta["turn"] == 0]
    assert out
    return out


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


def test_matched_seeds_default_off_keeps_v3_seeds(cfg, dataset):
    qs = load_questions(cfg, "train").head(2)
    no_ex = {u: {"text": None, "kb_ids": []} for u in qs["uid"]}
    plain = _seeds(cfg, qs, None)
    rag = _seeds(cfg, qs, no_ex)
    assert [p[1] for p in plain] != [r[1] for r in rag]   # v2/v3 behaviour unchanged


def test_matched_seeds_pairs_no_exemplar_questions(cfg, dataset):
    cfg["rag"]["matched_seeds"] = True
    qs = load_questions(cfg, "train").head(2)
    u0, u1 = list(qs["uid"])
    ex = {u0: {"text": "EXEMPLAR BLOCK", "kb_ids": ["k"]}, u1: {"text": None, "kb_ids": []}}
    plain = _seeds(cfg, qs, None)
    rag = _seeds(cfg, qs, ex)
    p1 = [p for p in plain if p[0] == u1]
    r1 = [r for r in rag if r[0] == u1]
    assert p1 and p1 == r1                           # identical requests -> cache hit
    p0 = [p for p in plain if p[0] == u0]
    r0 = [r for r in rag if r[0] == u0]
    assert p0 and [p[1] for p in p0] == [r[1] for r in r0]  # same seeds, other prompt
    assert all("EXEMPLAR BLOCK" in r[2] for r in r0)
