"""Code sandbox and the program-of-thoughts loop."""

from __future__ import annotations

from mmjee_reasoner.llm import GenRequest, MockEngine
from mmjee_reasoner.sandbox.executor import ExecCache, check_code, run_code, run_many
from mmjee_reasoner.sandbox.pot import (
    FENCE_STOP,
    Trajectory,
    apply_cot_fallback,
    needs_fallback,
    pending_python_block,
    run_pot,
)


def test_runs_simple_code():
    r = run_code("print(6*7)")
    assert r.ok and r.output == "42"


def test_sympy_and_numpy_allowed():
    r = run_code("import sympy as sp\nx=sp.symbols('x')\nprint(sp.solve(x**2-4, x))\n"
                 "import numpy as np\nprint(np.arange(3).sum())")
    assert r.ok, r.output
    assert "[-2, 2]" in r.output and "3" in r.output


def test_blocked_imports_and_calls():
    for code in ("import os\nprint(os.getcwd())", "from subprocess import run",
                 "open('/etc/passwd').read()", "print(().__class__.__bases__)",
                 "eval('1+1')", "__import__('os')", "import sympy.utilities.lambdify as l; "
                 "exec('x')"):
        r = run_code(code)
        assert not r.ok and r.error in ("blocked", "syntax"), code


def test_timeout_is_enforced():
    r = run_code("while True:\n    pass", timeout_s=1)
    assert not r.ok and r.timed_out


def test_memory_limit():
    r = run_code("x = [0] * (10**9)\nprint(len(x))", memory_mb=256)
    assert not r.ok and "MemoryError" in r.output


def test_runtime_error_reported():
    r = run_code("print(1/0)")
    assert not r.ok and "ZeroDivisionError" in r.output


def test_syntax_error():
    r = run_code("print(")
    assert not r.ok and r.error == "syntax"


def test_no_output_hint():
    r = run_code("x = 3")
    assert r.ok and "print" in r.output


def test_check_code_allows_relative_names():
    check_code("from fractions import Fraction\nprint(Fraction(1, 3))")


def test_run_many_and_cache(tmp_path):
    cache = ExecCache(str(tmp_path / "c.sqlite"))
    out = run_many(["print(1)", "print(2)"], workers=2, cache=cache)
    assert [o.output for o in out] == ["1", "2"]
    again = run_many(["print(1)"], workers=1, cache=cache)
    assert again[0].output == "1"


def test_pending_python_block():
    t = "Step 1: x\n```python\nprint(2)\n```\n"
    assert pending_python_block(t) == "print(2)"
    assert pending_python_block("text\n```output\n2\n```\n") is None
    assert pending_python_block("```python\na=1\n```\n```output\n1\n```\nthen\n```\n") is None


class Scripted:
    """Responds to PoT turns: code first, then a boxed answer."""

    def __init__(self, final="\\boxed{42}", code="print(6*7)"):
        self.final, self.code = final, code

    def __call__(self, req: GenRequest) -> str:
        if len(req.messages) == 1:
            return f"Step 1: compute\n```python\n{self.code}\n```\nhallucinated output"
        return f"\nStep 2: done. {self.final}"


def _traj(i=0):
    return Trajectory(rid=f"t{i}", prompt="solve ```python", image="img.png", language="English",
                      question_type="Numerical", seed_base=f"s{i}")


def test_run_pot_executes_and_continues(cfg):
    eng = MockEngine(Scripted())
    trajs = [_traj(0), _traj(1)]
    run_pot(trajs, eng, cfg)
    t = trajs[0]
    assert t.n_blocks == 1 and t.n_ok == 1 and t.done
    assert "```output\n42\n```" in t.full_text and "hallucinated" not in t.full_text
    assert t.full_text.strip().endswith("\\boxed{42}")
    assert t.calls == 2
    # first call stops on the fence, continuation uses continue_final
    assert eng.calls[0].sampling["stop"] == [FENCE_STOP]
    assert eng.calls[-1].continue_final


def test_fallback_when_code_never_runs(cfg):
    eng = MockEngine(Scripted(code="import os"))
    trajs = [_traj()]
    run_pot(trajs, eng, cfg)
    assert trajs[0].n_err == 1 and needs_fallback(trajs[0])
    cot = MockEngine(lambda r: "CoT answer \\boxed{7}")
    apply_cot_fallback(trajs, cot, cfg)
    assert trajs[0].used_fallback and trajs[0].full_text == "CoT answer \\boxed{7}"


def test_max_code_blocks_respected(cfg):
    class Loop:
        def __call__(self, req):
            return "more\n```python\nprint(1)\n```\n"

    cfg["sandbox"]["max_code_blocks"] = 2
    eng = MockEngine(Loop())
    t = _traj()
    run_pot([t], eng, cfg)
    assert t.n_blocks == 2
    assert eng.calls[-1].sampling["stop"] == []


def test_prelude_makes_earlier_variables_available():
    r = run_code("print(x * 2)", prelude=["x = 21\nprint('hidden')"])
    assert r.ok and r.output == "42"            # prelude output is silenced


def test_unsafe_prelude_block_is_dropped():
    r = run_code("print(1)", prelude=["import os\nos.system('echo pwned')"])
    assert r.ok and r.output == "1"


def test_scipy_allowed_but_file_io_submodules_blocked():
    r = run_code("from scipy.integrate import quad\nprint(round(quad(lambda t: t, 0, 2)[0], 3))")
    assert r.ok and r.output == "2.0", r.output
    for code in ("import scipy.io", "from scipy import io", "from sympy.parsing.sympy_parser import parse_expr"):
        assert run_code(code).error == "blocked", code


def test_run_pot_passes_earlier_blocks_as_prelude(cfg):
    def fn(req):
        if len(req.messages) == 1:
            return "Step 1:\n```python\na = 5\nprint(a)\n```\n"
        so_far = req.messages[-1]["content"]
        if so_far.count("```output") == 1:
            return "Step 2:\n```python\nprint(a + 1)\n```\n"
        return "\nStep 3: \\boxed{6}"

    t = _traj()
    run_pot([t], MockEngine(fn), cfg)
    assert t.n_ok == 2 and "```output\n6\n```" in t.full_text


def test_force_answers_commits_truncated_trajectories(cfg):
    from mmjee_reasoner.sandbox.pot import force_answers

    t = _traj()
    t.text, t.done = "long reasoning without an answer", True
    failed = _traj(1)
    failed.text, failed.n_blocks, failed.done = "```python\nimport os\n```", 1, True
    eng = MockEngine(lambda r: "7")
    force_answers([t, failed], eng, cfg)
    assert t.forced_answer and t.full_text.endswith("\\boxed{7}") and not needs_fallback(t)
    assert not failed.forced_answer and needs_fallback(failed)   # SOP fallback still applies
    assert len(eng.calls) == 1 and eng.calls[0].continue_final


def test_prelude_system_exit_does_not_stop_current_block():
    r = run_code("print('ran')", prelude=["raise SystemExit(0)"])
    assert r.ok and r.output == "ran"


def test_vllm_engine_isolates_invalid_requests():
    """A too-long request must not abort the batch: retried one by one, bad one -> empty."""
    from mmjee_reasoner.llm.engine import VLLMEngine
    from mmjee_reasoner.llm.types import GenResult

    class VLLMValidationError(Exception):
        pass

    eng = VLLMEngine.__new__(VLLMEngine)

    def run_batch(reqs, cont):
        if any(r.meta.get("long") for r in reqs):
            raise VLLMValidationError("prompt longer than max_model_len")
        return [GenResult("ok", 1, 1, "stop") for _ in reqs]

    eng._run_batch = run_batch
    reqs = [GenRequest([], {}, meta={"long": i == 1}) for i in range(3)]
    out = eng._run(reqs, False)
    assert [o.text for o in out] == ["ok", "", "ok"] and out[1].finish_reason == "error"


def test_pot_runs_code_when_model_stops_after_fence(cfg):
    from mmjee_reasoner.llm import MockEngine
    from mmjee_reasoner.sandbox.pot import Trajectory, run_pot

    def fn(req):
        if len(req.messages) == 1:
            return "Step 1:\n```python\nprint(5)\n```"   # EOS right after the fence
        return "\nStep 2: \\boxed{5}"

    t = Trajectory(rid="r", prompt="p ```python", image="i.png", language="English",
                   question_type="Numerical", seed_base="s")
    run_pot([t], MockEngine(fn), cfg)
    assert t.n_ok == 1 and "```output\n5\n```" in t.full_text


def test_sandbox_blocks_sympify_and_caps_output():
    from mmjee_reasoner.sandbox.executor import run_code

    assert run_code("import sympy\nsympy.sympify('1')").error == "blocked"
    assert run_code("from operator import attrgetter").error == "blocked"
    r = run_code("while True:\n    print('x' * 100000)", timeout_s=5)
    assert not r.ok and len(r.output) < 10000
