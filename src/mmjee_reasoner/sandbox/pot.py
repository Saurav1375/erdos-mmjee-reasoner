"""Program-of-Thoughts solving with code execution (Module A).

The model reasons in natural language and writes calculations as
```python blocks. Generation stops when a code fence closes, the block is run in
the sandbox, its output is appended as an ```output block, and generation
continues from there (``continue_final_message``). All trajectories of a stage
advance together, one batched model call per turn.

Fallback (SOP III-A): if the trajectory has no ``\\boxed`` answer, or code was
attempted but never ran successfully, the question is answered with plain
chain-of-thought (base-paper prompt) instead, so this module should not score
below the baseline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from mmjee_reasoner.llm import Engine, GenRequest, assistant_message, derive_seed, user_message
from mmjee_reasoner.prompts.templates import CODE_OUTPUT_TEMPLATE, baseline_prompt
from mmjee_reasoner.sandbox.executor import ExecCache, run_many

log = logging.getLogger(__name__)

FENCE_STOP = "\n```\n"
_OPEN_PY = re.compile(r"```(?:python|py|python3)[ \t]*\n", re.IGNORECASE)
_PY_BLOCK = re.compile(r"```(?:python|py|python3)[ \t]*\n(.*?)\n```", re.IGNORECASE | re.DOTALL)


def python_blocks(text: str) -> list[str]:
    """All closed ```python code blocks in ``text``, in order."""
    return [m.group(1) for m in _PY_BLOCK.finditer(text)]


@dataclass
class Trajectory:
    rid: str                       # unique id (experiment | uid | sample | ...)
    prompt: str                    # user text (image is added before it)
    image: str                     # image path relative to artifacts dir
    language: str
    question_type: str
    seed_base: str
    prefix: str = ""               # assistant prefill (corrector keeps accepted steps)
    text: str = ""                 # generated text after the prefix
    n_blocks: int = 0              # executed code blocks
    n_ok: int = 0
    n_err: int = 0
    exec_errors: list[str] = field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    done: bool = False
    finish: str = ""
    used_fallback: bool = False
    fallback_text: str = ""
    forced_answer: bool = False    # final answer forced after the token budget ran out

    @property
    def full_text(self) -> str:
        """Solution text used for scoring and as the next round's input."""
        return self.fallback_text if self.used_fallback else self.prefix + self.text

    def stats(self) -> dict[str, Any]:
        return {
            "n_code_blocks": self.n_blocks, "n_code_ok": self.n_ok, "n_code_err": self.n_err,
            "exec_errors": self.exec_errors, "calls": self.calls,
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "finish": self.finish, "used_fallback": self.used_fallback,
            "forced_answer": self.forced_answer,
        }


def pending_python_block(text: str) -> str | None:
    """If ``text`` ends with a fence that closes a ```python block, return its code.

    ``text`` must end with the closing fence ("```\\n"). Returns ``None`` when
    the final fence closes/opens any other kind of block.
    """
    body = text.rstrip("\n")
    if not body.endswith("```"):
        return None
    body = body[:-3]
    prev = body.rfind("```")
    if prev < 0:
        return None
    m = _OPEN_PY.match(body, prev)
    if not m:
        return None
    return body[m.end():].strip("\n")


def run_pot(trajs: list[Trajectory], engine: Engine, cfg: dict,
            sampling: dict | None = None, exec_cache: ExecCache | None = None) -> None:
    """Advance all trajectories to completion (in place)."""
    sb = cfg["sandbox"]
    samp = dict(cfg["sampling"] if sampling is None else sampling)
    budget = int(samp.pop("max_tokens"))
    max_ctx = int(cfg["models"]["solver"]["max_model_len"])
    next_prompt: dict[str, int] = {}  # rid -> estimated prompt tokens of the next turn
    exec_kwargs = dict(timeout_s=sb["timeout_s"], memory_mb=sb["memory_mb"],
                       max_output_chars=sb["max_output_chars"],
                       allowed_imports=tuple(sb["allowed_imports"]))

    for turn in range(sb["max_turns"]):
        active = [t for t in trajs if not t.done]
        if not active:
            break
        reqs, owners = [], []
        for t in active:
            remaining = budget - t.completion_tokens
            if t.rid in next_prompt:  # keep prompt + generation inside the context window
                remaining = min(remaining, max_ctx - next_prompt[t.rid] - 64)
            if remaining < 32:
                t.done, t.finish = True, "budget"
                continue
            so_far = t.prefix + t.text
            msgs = [user_message(t.prompt, t.image)]
            if so_far:
                msgs.append(assistant_message(so_far))
            stop = [FENCE_STOP] if t.n_blocks < sb["max_code_blocks"] else []
            reqs.append(GenRequest(
                messages=msgs,
                sampling={**samp, "max_tokens": remaining, "stop": stop,
                          "seed": derive_seed(t.seed_base, "turn", turn)},
                continue_final=bool(so_far),
                meta={"rid": t.rid, "turn": turn},
            ))
            owners.append(t)
        if not reqs:
            break
        results = engine.generate(reqs)

        to_exec: list[tuple[Trajectory, str, list[str]]] = []
        for t, r in zip(owners, results):
            t.calls += 1
            t.prompt_tokens += r.prompt_tokens
            t.completion_tokens += r.completion_tokens
            t.text += r.text
            next_prompt[t.rid] = r.prompt_tokens + r.completion_tokens
            code = None
            if r.stop_reason == FENCE_STOP:
                t.text += FENCE_STOP
                code = pending_python_block(t.prefix + t.text)
                # code None: a non-python fence; keep generating next turn
            elif (r.finish_reason == "stop" and t.n_blocks < sb["max_code_blocks"]
                  and t.text.rstrip().endswith("```")):
                # model ended its turn right after the closing fence (no trailing newline)
                t.text = t.text.rstrip() + "\n"
                code = pending_python_block(t.prefix + t.text)
                if code is None:
                    t.done, t.finish = True, r.finish_reason
            else:
                t.done, t.finish = True, r.finish_reason
            if code is not None:
                # earlier blocks (incl. those in a corrector's prefix) are replayed first
                prelude = python_blocks(t.prefix + t.text)[:-1]
                to_exec.append((t, code, prelude))

        outputs = run_many([c for _, c, _ in to_exec], workers=sb["workers"], cache=exec_cache,
                           preludes=[p for _, _, p in to_exec], **exec_kwargs)
        for (t, _, _), res in zip(to_exec, outputs):
            t.n_blocks += 1
            if res.ok:
                t.n_ok += 1
            else:
                t.n_err += 1
                t.exec_errors.append(res.error or "error")
            block = CODE_OUTPUT_TEMPLATE.format(output=res.output)
            t.text += block
            next_prompt[t.rid] = next_prompt.get(t.rid, 0) + len(block) // 2 + 8

    for t in trajs:
        if not t.done:
            t.done, t.finish = True, "max_turns"


def code_failed(t: Trajectory) -> bool:
    return t.n_blocks > 0 and t.n_ok == 0


def force_answers(trajs: list[Trajectory], engine: Engine, cfg: dict) -> None:
    """Budget forcing: trajectories that ran out of tokens without a ``\\boxed`` answer
    (and whose code did not fail outright) commit to an answer from their own reasoning.

    One short greedy continuation after an explicit "final answer" cue. Disabled by
    default: on the pilot's 480 truncated no-code trajectories it scored 21.9% vs 30.0%
    for the SOP CoT fallback (kept as an option and reported as a negative result).
    """
    from mmjee_reasoner.prompts.templates import FORCE_ANSWER_CUE

    todo = [t for t in trajs if "\\boxed" not in t.prefix + t.text and not code_failed(t)]
    if not todo:
        return
    reqs = [GenRequest(
        messages=[user_message(t.prompt, t.image),
                  assistant_message((t.prefix + t.text).rstrip() + FORCE_ANSWER_CUE)],
        sampling={"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 24,
                  "stop": ["}"], "seed": 0},
        continue_final=True, meta={"rid": t.rid, "force": True},
    ) for t in todo]
    for t, r in zip(todo, engine.generate(reqs)):
        t.text = t.text.rstrip() + FORCE_ANSWER_CUE + r.text.strip() + "}"
        t.forced_answer = True
        t.calls += 1
        t.prompt_tokens += r.prompt_tokens
        t.completion_tokens += r.completion_tokens
    log.info("answer forcing used for %d/%d trajectories", len(todo), len(trajs))


def needs_fallback(t: Trajectory) -> bool:
    """SOP III-A: no boxed answer, or code attempted but never ran successfully."""
    text = t.prefix + t.text
    return ("\\boxed" not in text) or code_failed(t)


def apply_cot_fallback(trajs: list[Trajectory], engine: Engine, cfg: dict,
                       sampling: dict | None = None) -> None:
    """Answer trajectories that need it with one plain-CoT sample (base-paper prompt)."""
    samp = dict(cfg["sampling"] if sampling is None else sampling)
    todo = [t for t in trajs if needs_fallback(t)]
    if not todo:
        return
    reqs = [GenRequest(
        messages=[user_message(baseline_prompt(t.language, t.question_type), t.image)],
        sampling={**samp, "seed": derive_seed(t.seed_base, "fallback")},
        meta={"rid": t.rid, "fallback": True},
    ) for t in todo]
    for t, r in zip(todo, engine.generate(reqs)):
        t.used_fallback = True
        t.fallback_text = r.text
        t.calls += 1
        t.prompt_tokens += r.prompt_tokens
        t.completion_tokens += r.completion_tokens
    log.info("CoT fallback used for %d/%d trajectories", len(todo), len(trajs))
