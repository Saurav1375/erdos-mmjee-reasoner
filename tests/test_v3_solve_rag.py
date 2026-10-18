"""v3 solver options (adaptive budget, text questions, transcripts) and the v3 KB (mock)."""

from __future__ import annotations

import json

from mmjee_reasoner.config import load_experiment
from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.llm import MockEngine, make_engine
from mmjee_reasoner.llm.mock import FakeVLM, golds_for
from mmjee_reasoner.llm.types import GenResult
from mmjee_reasoner.pipeline.common import load_candidates


class TruncatingEngine:
    """First calls hit the token limit; a continuation (prefilled assistant turn) finishes."""

    name = "trunc"

    def __init__(self):
        self.calls = []

    def generate(self, reqs):
        out = []
        for r in reqs:
            self.calls.append(r)
            if r.continue_final:
                out.append(GenResult("\nStep 9: done. \\boxed{A}", 10, 5, "stop"))
            else:
                n = r.sampling["max_tokens"]
                out.append(GenResult("Step 1: a very long derivation", 10, n, "length"))
        return out


def _cfg_ext(cfg, ext):
    cfg["sampling"]["extend_on_truncation"] = ext
    return cfg


def test_cot_extension(cfg, dataset):
    from mmjee_reasoner.pipeline.solve import solve_cot

    qs = load_questions(cfg, "test").head(2)
    exp = {"name": "cot", "mode": "cot", "n_samples": 1}
    eng = TruncatingEngine()
    recs = solve_cot(_cfg_ext(cfg, 0), exp, eng, qs)
    assert len(eng.calls) == 2 and all(r["pred"] == "" or r["finish"] == "length" for r in recs)
    assert all("extend_on_truncation" not in c.sampling for c in eng.calls)  # never sent
    eng = TruncatingEngine()
    recs = solve_cot(_cfg_ext(cfg, 4096), exp, eng, qs)
    assert len(eng.calls) == 4                                   # 2 first + 2 continuations
    assert all(r["finish"] == "stop" and r["pred"] == "A" for r in recs)
    assert all(r["completion_tokens"] == 4096 + 5 for r in recs)


def test_pot_extension_continues_instead_of_fallback(cfg, dataset):
    from mmjee_reasoner.pipeline.solve import solve_pot

    qs = load_questions(cfg, "test").head(1)
    exp = {"name": "pot", "mode": "pot", "n_samples": 1}
    eng = TruncatingEngine()
    rec = solve_pot(_cfg_ext(cfg, 4096), exp, eng, qs)[0]
    assert not rec["used_fallback"] and rec["pred"] == "A"
    assert rec["completion_tokens"] == 4096 + 5
    eng = TruncatingEngine()
    rec = solve_pot(_cfg_ext(cfg, 0), exp, eng, qs)[0]
    assert rec["used_fallback"]                                    # v2: CoT fallback


def _write_jeebench(cfg):
    from mmjee_reasoner.config import artifacts_dir

    raw = [{"description": "JEE Adv 2016 Paper 1", "index": i, "subject": s, "type": t,
            "question": f"Compute the {s} quantity number {i} for a block on an incline.",
            "gold": g}
           for i, (s, t, g) in enumerate([("phy", "MCQ", "B"), ("chem", "Integer", "4"),
                                          ("math", "MCQ(multiple)", "AC")], start=1)]
    raw.append({"description": "JEE Adv 2020 Paper 1", "index": 9, "subject": "phy",
                "type": "MCQ", "question": "not used (2020)", "gold": "A"})
    artifacts_dir(cfg, "data", "external", "jeebench_test.json").write_text(json.dumps(raw))


def test_text_questions_solved_without_image(cfg, dataset):
    from mmjee_reasoner.data.external import load_external
    from mmjee_reasoner.pipeline.solve import run_solve_stage

    _write_jeebench(cfg)
    ext = load_external(cfg, "jeebench")
    assert len(ext) == 3 and set(ext["year"]) == {2016}            # 2020 item dropped
    fake = MockEngine(FakeVLM())
    run_solve_stage(cfg, load_experiment("kb_ext_pot"), make_engine(cfg, "solver", mock=fake))
    df = load_candidates(cfg, "kb_ext_pot")
    assert len(df) == 3 * 8
    first = fake.calls[0].messages[0]["content"]
    assert all(i["type"] == "text" for i in first)                # no image part
    assert "Question:\nCompute the" in first[0]["text"]


def test_transcript_block(cfg, dataset):
    from mmjee_reasoner.pipeline.solve import solve_pot

    qs = load_questions(cfg, "train").head(1)
    fake = MockEngine(FakeVLM(golds_for(qs)))
    solve_pot(cfg, {"name": "t", "n_samples": 1}, fake, qs, None, {qs["uid"][0]: "TXT 123"})
    text = fake.calls[0].messages[0]["content"][-1]["text"]
    assert "Transcription:\nTXT 123" in text and "image is authoritative" in text


def test_v3_kb_external_hybrid_loyo(cfg, dataset, fake_embedder):
    from mmjee_reasoner.data.external import load_external
    from mmjee_reasoner.pipeline.solve import run_solve_stage
    from mmjee_reasoner.rag.kb import build_kb, load_problem_vecs
    from mmjee_reasoner.rag.retrieve import exemplars_for_questions
    from mmjee_reasoner.rag.tagger import (
        load_english_transcripts,
        run_tag_stage,
        run_transcribe_stage,
        run_translate_stage,
    )

    _write_jeebench(cfg)
    cfg["rag"].update(kb_name="kb_v3", extra_kb=[{"exp": "kb_ext_pot", "dataset": "jeebench"}],
                      retrieval="hybrid", hybrid_weight=0.5, threshold_tuning="loyo",
                      dedup_years=[2025])
    qs = load_questions(cfg)
    qs = qs[qs["split"] != "excluded"]
    fake = MockEngine(FakeVLM(golds_for(qs), p_correct=0.9))
    eng = make_engine(cfg, "solver", mock=fake)
    run_tag_stage(cfg, qs, eng)
    run_tag_stage(cfg, load_external(cfg, "jeebench"), eng)
    run_transcribe_stage(cfg, qs, eng)
    run_translate_stage(cfg, qs, eng)
    tx = load_english_transcripts(cfg)
    assert set(tx) == set(qs["uid"])                              # Hindi via translation
    run_solve_stage(cfg, load_experiment("pot_train"), eng)
    run_solve_stage(cfg, load_experiment("kb_ext_pot"), eng)
    # The fake model only knows golds of image questions: mark the text items as solved.
    from mmjee_reasoner.pipeline.common import candidates_path, write_jsonl

    ext = load_candidates(cfg, "kb_ext_pot").assign(correct=True)
    write_jsonl(candidates_path(cfg, "kb_ext_pot"), ext.to_dict("records"))
    meta = build_kb(cfg)
    assert meta["entries_by_source"].get("jeebench", 0) > 0
    assert meta["threshold"]["method"].startswith("loyo")
    assert load_problem_vecs(cfg).shape[0] == meta["entries"]
    test = qs[qs["split"] == "test"]
    ex = exemplars_for_questions(cfg, test)
    assert set(ex) == set(test["uid"])


def test_limit_applies_after_year_and_dataset_filters(cfg, dataset):
    """--limit smoke runs draw from the filtered questions (bug: 2024-only dev runs and the
    JEEBench run ignored the limit / sampled from all years)."""
    from mmjee_reasoner.data.external import load_external
    from mmjee_reasoner.pipeline.common import select_questions, set_subset

    _write_jeebench(cfg)
    set_subset(cfg, limit=2)
    train = load_questions(cfg, "train")
    y = int(train["year"].iloc[0])
    qs = select_questions(cfg, "train", years=[y])
    assert len(qs) == min(2, int((train["year"] == y).sum())) and set(qs["year"]) == {y}
    ext = select_questions(cfg, "external", df=load_external(cfg, "jeebench"))
    assert len(ext) == 2
