"""Phase 4b (V4B_PREREG.md): diagram-first transcript view and sample_offset (mock engine)."""

from __future__ import annotations

import pytest

from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.llm import MockEngine
from mmjee_reasoner.llm.mock import FakeVLM, golds_for


def _first_turns(cfg, qs, exp, transcripts=None):
    from mmjee_reasoner.pipeline.solve import solve_pot

    fake = MockEngine(FakeVLM(golds_for(qs)))
    recs = solve_pot(cfg, {"name": "t", **exp}, fake, qs, None, transcripts)
    calls = {c.meta["rid"]: (c.sampling["seed"], c.messages[0]["content"][-1]["text"])
             for c in fake.calls if c.meta.get("turn") == 0}
    return recs, calls


def test_sample_offset_reuses_seeds_of_those_samples(cfg, dataset):
    qs = load_questions(cfg, "train").head(2)
    recs, calls = _first_turns(cfg, qs, {"n_samples": 2, "sample_offset": 4})
    _, full = _first_turns(cfg, qs, {"n_samples": 6})
    assert sorted({r["sample"] for r in recs}) == [4, 5]
    assert calls and calls == {rid: full[rid] for rid in calls}   # same rid, seed and prompt


def test_default_offset_unchanged(cfg, dataset):
    qs = load_questions(cfg, "train").head(2)
    recs, calls = _first_turns(cfg, qs, {"n_samples": 2})
    _, full = _first_turns(cfg, qs, {"n_samples": 6})
    assert sorted({r["sample"] for r in recs}) == [0, 1]
    assert calls == {rid: full[rid] for rid in calls}


def test_diagram_transcribe_writes_own_file(cfg, dataset):
    from mmjee_reasoner.prompts.templates import TRANSCRIBE_DIAGRAM_PROMPT, TRANSCRIBE_PROMPT
    from mmjee_reasoner.rag.tagger import (
        load_diagram_transcripts,
        run_transcribe_stage,
        transcripts_diagram_path,
        transcripts_path,
    )

    qs = load_questions(cfg, "train").head(3)
    fake = MockEngine(FakeVLM(golds_for(qs)))
    run_transcribe_stage(cfg, qs, fake, view="diagram")
    prompts = [c.messages[0]["content"][-1]["text"] for c in fake.calls]
    assert prompts and all(p == TRANSCRIBE_DIAGRAM_PROMPT for p in prompts)
    assert transcripts_diagram_path(cfg).exists() and not transcripts_path(cfg).exists()
    assert set(load_diagram_transcripts(cfg)) == set(qs["uid"])
    fake = MockEngine(FakeVLM(golds_for(qs)))
    run_transcribe_stage(cfg, qs, fake)                       # default view: v3 prompt and file
    assert all(c.messages[0]["content"][-1]["text"] == TRANSCRIBE_PROMPT for c in fake.calls)
    assert transcripts_path(cfg).exists()


def test_transcript_view_selects_the_diagram_reading(cfg, dataset):
    from mmjee_reasoner.pipeline.common import write_jsonl
    from mmjee_reasoner.pipeline.solve import run_solve_stage
    from mmjee_reasoner.rag.tagger import transcripts_diagram_path, transcripts_path

    cfg["sandbox"]["force_answer"] = False
    qs = load_questions(cfg, "train")
    year = int(qs["year"].min())
    uids = list(qs[qs["year"] == year]["uid"])
    write_jsonl(transcripts_path(cfg), [{"uid": u, "text": "PLAIN"} for u in uids])
    exp = {"name": "v4b_t", "kind": "solve", "mode": "pot", "split": "train", "years": [year],
           "transcript": True, "transcript_view": "diagram", "n_samples": 1, "sample_offset": 4}
    with pytest.raises(RuntimeError, match="transcribe --view diagram"):
        run_solve_stage(cfg, exp, MockEngine(FakeVLM(golds_for(qs))))
    write_jsonl(transcripts_diagram_path(cfg), [{"uid": u, "text": "DIAGRAM ITEMS"} for u in uids])
    fake = MockEngine(FakeVLM(golds_for(qs)))
    recs = run_solve_stage(cfg, exp, fake)
    first = [c.messages[0]["content"][-1]["text"] for c in fake.calls if c.meta.get("turn") == 0]
    assert recs and {r["sample"] for r in recs} == {4}
    assert first and all("Transcription:\nDIAGRAM ITEMS" in t and "PLAIN" not in t
                         for t in first)

