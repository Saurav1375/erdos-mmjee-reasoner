"""Phase 4b (V4B_PREREG.md): diagram-first transcript view and sample_offset (mock engine)."""

from __future__ import annotations

import numpy as np
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


def _toy_frame() -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for q, (year, preds, gold) in enumerate([(2019, ["A", "A", "B", "C"], "B"),
                                             (2020, ["1.50", "1.5", "2", "2"], "2"),
                                             (2021, ["AB", "BA", "A", "AB"], "AB")]):
        qtype = {2019: "MCQ-Single", 2020: "Numerical", 2021: "MCQ-Multiple"}[year]
        for s, p in enumerate(preds):
            rows.append({"uid": f"q{q}", "year": year, "pool": "v3", "pred": p,
                         "question_type": qtype, "correct": p.replace("BA", "AB") == gold
                         or (qtype == "Numerical" and float(p) == float(gold)),
                         "is_corrected": s >= 2, "has_chain": True,
                         "corr_critic_flagged_first": float(s % 2), "corr_n_flags": 0.0,
                         "corr_rounds": 1.0, "corr_answer_changed": float(s == 3),
                         "correction_failed_last": 0.0, "text": "step\\boxed{x}" * (s + 1),
                         "completion_tokens": 100 * (s + 1), "n_steps": s + 1,
                         "n_code_blocks": 1, "n_code_ok": 1, "n_code_err": 0,
                         "used_fallback": False, "finish": "stop", "subject": "Physics",
                         "language": "English", "requires_image": q == 0, "source": "x"})
    return pd.DataFrame(rows)


def test_answer_rows_one_per_distinct_answer():
    from mmjee_reasoner.verifier.v4b import answer_rows, feature_cols, picks

    df = _toy_frame()
    a = answer_rows(df)
    assert a.groupby("uid").size().to_dict() == {"q0": 3, "q1": 2, "q2": 2}  # 1.50 == 1.5, AB == BA
    q1 = a[a["uid"] == "q1"].set_index("key")
    assert q1["F_vote_share"].tolist() == [0.5, 0.5] and q1["label"].tolist() == [0.0, 1.0]
    q0 = a[a["uid"] == "q0"].set_index("key")
    assert q0.loc["A", "F_vote_rank"] == 0.0 and q0.loc["A", "F_orig_share"] == 1.0
    assert q0.loc["B", "F_gap_top"] == 0.25 and q0.loc["B", "F_corr_share"] == 0.5
    assert answer_rows(df).equals(a)                              # deterministic
    assert not a[feature_cols(a)].isna().any().any()
    pk = picks(a, a["label"].to_numpy(float))                     # perfect scores -> oracle
    assert pk["label"].tolist() == [1.0, 1.0, 1.0]
    tie = picks(a, np.zeros(len(a)))                              # ties -> earliest answer
    assert tie.loc["q0", "label"] == 0.0 and tie.loc["q1", "label"] == 0.0
