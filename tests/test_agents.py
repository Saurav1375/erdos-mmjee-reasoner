"""Step splitting, critic parsing and the correction loop."""

from __future__ import annotations

from mmjee_reasoner.agents.loop import Chain, new_solution, run_correction_inprocess
from mmjee_reasoner.agents.steps import parse_critic, split_steps
from mmjee_reasoner.llm import MockEngine


def test_split_steps_markers_and_code_comments():
    text = ("Intro.\nStep 1: a\n```python\n# Step 2: not a step\nprint(1)\n```\n"
            "```output\n1\n```\n**Step 2:** b\n### Step 3 c")
    steps = split_steps(text)
    assert len(steps) == 3
    assert steps[0].startswith("Intro.") and "# Step 2: not a step" in steps[0]
    assert steps[1].startswith("**Step 2:**")


def test_split_steps_paragraph_fallback():
    text = "first para\n\nsecond\n```python\nx=1\n\ny=2\n```\n\nthird"
    steps = split_steps(text)
    assert len(steps) == 3 and "y=2" in steps[1]


def test_parse_critic():
    v = parse_critic("thinking...\nVERDICT: ERROR\nFIRST_ERROR_STEP: 3\nEXPLANATION: bad sign", 5)
    assert v.has_error and v.step == 3 and v.explanation == "bad sign" and v.parsed
    v = parse_critic("VERDICT: CORRECT\nFIRST_ERROR_STEP: NONE\nEXPLANATION: fine", 5)
    assert not v.has_error and v.step is None
    v = parse_critic("VERDICT: **ERROR**\nFIRST_ERROR_STEP: Step 9", 5)  # out of range -> 1
    assert v.has_error and v.step == 1
    v = parse_critic("I cannot tell.", 5)
    assert not v.has_error and not v.parsed
    v = parse_critic("VERDICT: ERROR\nFIRST_ERROR_STEP: 2\nlater VERDICT: CORRECT", 5)
    assert not v.has_error  # last verdict wins


def _chain(text="Step 1: a\nStep 2: b \\boxed{5}"):
    c = Chain(cid="u|s0", uid="u", language="English", question_type="Numerical",
              image="img.png", seed_base="x")
    c.solutions.append(new_solution(text, "Numerical", {"calls": 1, "prompt_tokens": 1,
                                                         "completion_tokens": 1}))
    return c


def test_loop_stops_when_critic_accepts(cfg):
    critic = MockEngine(lambda r: "VERDICT: CORRECT\nFIRST_ERROR_STEP: NONE\nEXPLANATION: ok")
    solver = MockEngine(lambda r: "unused")
    c = _chain()
    run_correction_inprocess([c], solver, critic, cfg)
    assert c.done and c.stop_reason == "critic_ok" and len(c.solutions) == 1
    assert solver.calls == []


def test_loop_corrects_from_located_step(cfg):
    critic = MockEngine(lambda r: "VERDICT: ERROR\nFIRST_ERROR_STEP: 2\nEXPLANATION: wrong")
    answers = iter(["7", "7", "7"])

    def solver_fn(req):
        if len(req.messages) == 2 and req.messages[-1]["content"].endswith("Step 2:"):
            return f" fixed \\boxed{{{next(answers)}}}"
        return "x"

    solver = MockEngine(solver_fn)
    c = _chain()
    cfg["correction"]["unchanged_patience"] = 2
    run_correction_inprocess([c], solver, critic, cfg)
    # prefix keeps step 1 and restarts at step 2
    prefill = solver.calls[0].messages[-1]["content"]
    assert prefill.startswith("Step 1: a") and prefill.endswith("Step 2:")
    assert "wrong" in solver.calls[0].messages[0]["content"][-1]["text"]
    assert c.solutions[1]["answer"] == "7"
    # 5 -> 7 (changed), 7 -> 7 (streak 1), 7 -> 7 (streak 2 = patience)
    assert c.stop_reason == "unchanged" and len(c.solutions) == 4


def test_loop_max_rounds(cfg):
    critic = MockEngine(lambda r: "VERDICT: ERROR\nFIRST_ERROR_STEP: 1\nEXPLANATION: e")
    counter = iter(range(100))
    solver = MockEngine(lambda r: f"Step 1: \\boxed{{{next(counter)}}}")
    c = _chain()
    run_correction_inprocess([c], solver, critic, cfg)
    assert c.stop_reason == "max_rounds" and len(c.critiques) == cfg["correction"]["max_rounds"]


def test_failed_correction_keeps_previous(cfg):
    critic = MockEngine(lambda r: "VERDICT: ERROR\nFIRST_ERROR_STEP: 1\nEXPLANATION: e")
    solver = MockEngine(lambda r: "no answer here")
    cfg["sandbox"]["force_answer"] = False      # test the no-answer path itself
    c = _chain()
    run_correction_inprocess([c], solver, critic, cfg)
    assert all(s["answer"] == "5" for s in c.solutions)
    assert c.solutions[1]["correction_failed"]


def test_critic_verdict_forcing_on_truncated_reply(cfg):
    from mmjee_reasoner.agents.loop import critic_phase

    def fn(req):
        if len(req.messages) == 2:          # forced continuation after the cue
            return " ERROR\nFIRST_ERROR_STEP: 2\nEXPLANATION: wrong integral"
        return "Let me re-solve everything ... (cut off)"

    c = _chain()
    critic_phase([c], MockEngine(fn), cfg, 0)
    k = c.critiques[0]
    assert k["forced_verdict"] and k["parsed"] and k["has_error"] and k["step"] == 2
    assert k["calls"] == 2
