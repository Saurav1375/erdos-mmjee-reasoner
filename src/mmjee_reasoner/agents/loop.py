"""Agentic iterative correction: Solver -> Critic -> Corrector (Module B).

Each candidate solution starts a *chain*. One round is:

1. **Critic** (same VLM or a different one) reads the image, the instructions
   and the numbered steps, and names the FIRST wrong step (or says CORRECT).
2. **Corrector** (the Solver VLM with the PoT sandbox) keeps the accepted
   steps 1..k-1 as a prefilled answer, receives the critic's explanation and
   re-derives the solution from step k.

Stopping rule: the critic finds no error, OR the final answer stayed the same
for ``unchanged_patience`` consecutive rounds, OR ``max_rounds`` rounds ran.

Rounds are split into two *phases* (critic, corrector) that operate on a
persisted list of chains, so a cross-model critic can run in a separate
process from the solver (one 8B model fits on the 12 GB GPU at a time).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mmjee_reasoner.agents.steps import number_steps, parse_critic, split_steps
from mmjee_reasoner.llm import Engine, GenRequest, assistant_message, derive_seed, user_message
from mmjee_reasoner.prompts.templates import CRITIC_VERDICT_CUE, corrector_prompt, critic_prompt
from mmjee_reasoner.sandbox.executor import ExecCache
from mmjee_reasoner.sandbox.pot import Trajectory, force_answers, needs_fallback, run_pot
from mmjee_reasoner.scoring import extract_answer

log = logging.getLogger(__name__)


@dataclass
class Chain:
    cid: str
    uid: str
    language: str
    question_type: str
    image: str
    seed_base: str
    exemplars: str | None = None
    solutions: list[dict] = field(default_factory=list)   # round 0 = original candidate
    critiques: list[dict] = field(default_factory=list)
    done: bool = False
    stop_reason: str = ""
    unchanged_streak: int = 0

    @property
    def current(self) -> dict:
        return self.solutions[-1]


def new_solution(text: str, question_type: str, stats: dict | None = None,
                 failed: bool = False) -> dict:
    return {
        "text": text,
        "answer": extract_answer(text, question_type),
        "n_steps": len(split_steps(text)),
        "stats": stats or {},
        "correction_failed": failed,
    }


def save_chains(chains: list[Chain], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(c) for c in chains], ensure_ascii=False))
    tmp.replace(path)


def load_chains(path: Path) -> list[Chain]:
    return [Chain(**d) for d in json.loads(path.read_text())]


def critic_phase(chains: list[Chain], engine: Engine, cfg: dict, round_idx: int) -> int:
    """Run the critic for every open chain at ``round_idx``. Returns #critiques made."""
    todo = [c for c in chains if not c.done and len(c.critiques) == round_idx]
    if not todo:
        return 0
    samp = cfg["correction"]["critic_sampling"]
    reqs = []
    for c in todo:
        steps = split_steps(c.current["text"])
        prompt = critic_prompt(c.language, c.question_type, number_steps(steps))
        reqs.append(GenRequest(
            messages=[user_message(prompt, c.image)],
            sampling={**samp, "seed": derive_seed(c.seed_base, "critic", round_idx)},
            meta={"cid": c.cid, "round": round_idx},
        ))
    results = engine.generate(reqs)
    # Verdict forcing: replies without the required lines (mostly cut off at max_tokens
    # while re-solving the problem) get one short greedy continuation after a cue.
    replies = [r.text for r in results]
    extra = [(0, 0)] * len(todo)
    redo = [i for i, (c, r) in enumerate(zip(todo, results))
            if not parse_critic(r.text, len(split_steps(c.current["text"]))).parsed]
    if redo:
        freqs = [GenRequest(
            messages=[reqs[i].messages[0],
                      assistant_message(results[i].text.rstrip() + CRITIC_VERDICT_CUE)],
            sampling={"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 96,
                      "seed": 0},
            continue_final=True, meta={"cid": todo[i].cid, "force": True},
        ) for i in redo]
        for i, fr in zip(redo, engine.generate(freqs)):
            replies[i] = results[i].text.rstrip() + CRITIC_VERDICT_CUE + fr.text
            extra[i] = (fr.prompt_tokens, fr.completion_tokens)
    for i, (c, r) in enumerate(zip(todo, results)):
        n_steps = len(split_steps(c.current["text"]))
        v = parse_critic(replies[i], n_steps)
        c.critiques.append({
            "reply": replies[i], "has_error": v.has_error, "step": v.step,
            "explanation": v.explanation, "parsed": v.parsed, "n_steps": n_steps,
            "forced_verdict": i in redo,
            "prompt_tokens": r.prompt_tokens + extra[i][0],
            "completion_tokens": r.completion_tokens + extra[i][1],
            "calls": 2 if i in redo else 1,
            "finish": r.finish_reason, "model": getattr(engine, "name", "?"),
        })
        if not v.has_error:
            c.done, c.stop_reason = True, "critic_ok"
    return len(todo)


def corrector_phase(chains: list[Chain], engine: Engine, cfg: dict, round_idx: int,
                    exec_cache: ExecCache | None = None) -> int:
    """Re-derive flagged solutions from the located step. Returns #corrections made."""
    corr = cfg["correction"]
    todo = [c for c in chains
            if not c.done and len(c.critiques) == round_idx + 1
            and len(c.solutions) == round_idx + 1]
    if not todo:
        return 0
    trajs = []
    for c in todo:
        crit = c.critiques[-1]
        k = int(crit["step"] or 1)
        steps = split_steps(c.current["text"])
        kept = "\n\n".join(steps[: k - 1])
        prefix = (kept + "\n\n" if kept else "") + f"Step {k}:"
        trajs.append(Trajectory(
            rid=f"{c.cid}|corr{round_idx}",
            prompt=corrector_prompt(c.language, c.question_type,
                                    cfg["sandbox"]["max_code_blocks"], k,
                                    crit["explanation"], c.exemplars),
            image=c.image, language=c.language, question_type=c.question_type,
            seed_base=f"{c.seed_base}|corr{round_idx}", prefix=prefix,
        ))
    run_pot(trajs, engine, cfg, exec_cache=exec_cache)
    if cfg["sandbox"].get("force_answer"):
        force_answers(trajs, engine, cfg)
    for c, t in zip(todo, trajs):
        failed = needs_fallback(t)
        if failed:  # no usable corrected solution: keep the previous one
            sol = dict(c.current)
            sol["correction_failed"] = True
        else:
            sol = new_solution(t.full_text, c.question_type, t.stats())
        sol["attempt"] = t.stats()  # compute spent by this corrector call
        prev_answer = c.current["answer"]
        c.solutions.append(sol)
        c.unchanged_streak = c.unchanged_streak + 1 if sol["answer"] == prev_answer else 0
        if c.unchanged_streak >= corr["unchanged_patience"]:
            c.done, c.stop_reason = True, "unchanged"
        elif round_idx + 1 >= corr["max_rounds"]:
            c.done, c.stop_reason = True, "max_rounds"
    return len(todo)


def run_correction_inprocess(chains: list[Chain], solver: Engine, critic: Engine, cfg: dict,
                             exec_cache: ExecCache | None = None,
                             state_path: Path | None = None) -> list[Chain]:
    """All rounds in one process (used when critic and solver share a model, and in tests)."""
    for r in range(cfg["correction"]["max_rounds"]):
        critic_phase(chains, critic, cfg, r)
        if state_path:
            save_chains(chains, state_path)
        corrector_phase(chains, solver, cfg, r, exec_cache)
        if state_path:
            save_chains(chains, state_path)
        if all(c.done for c in chains):
            break
    for c in chains:
        if not c.done:
            c.done, c.stop_reason = True, "max_rounds"
    return chains


def chain_summary(c: Chain) -> dict:
    """Per-chain statistics used for the correction analysis and the verifier."""
    flagged = [x for x in c.critiques if x["has_error"]]
    tokens = sum(x["completion_tokens"] for x in c.critiques) + sum(
        s["attempt"]["completion_tokens"] for s in c.solutions[1:])
    calls = sum(x.get("calls", 1) for x in c.critiques) + sum(
        s["attempt"]["calls"] for s in c.solutions[1:])
    return {
        "rounds": len(c.critiques),
        "critic_flagged_first": bool(c.critiques and c.critiques[0]["has_error"]),
        "n_flags": len(flagged),
        "answer_changed": c.solutions[-1]["answer"] != c.solutions[0]["answer"],
        "stop_reason": c.stop_reason,
        "extra_completion_tokens": tokens,
        "extra_calls": calls,
        "critic_unparsed": sum(not x["parsed"] for x in c.critiques),
    }
