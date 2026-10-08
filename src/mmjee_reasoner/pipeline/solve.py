"""Solver stage: sample N solutions per question (baseline CoT, PoT, PoT + RAG)."""

from __future__ import annotations

import logging

from mmjee_reasoner.agents.steps import split_steps
from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.llm import Engine, GenRequest, derive_seed, user_message
from mmjee_reasoner.pipeline.common import (
    base_record,
    candidates_path,
    relative_image,
    score_into,
    select_questions,
    write_jsonl,
)
from mmjee_reasoner.prompts.templates import baseline_prompt, pot_prompt
from mmjee_reasoner.sandbox.executor import ExecCache
from mmjee_reasoner.sandbox.pot import Trajectory, apply_cot_fallback, force_answers, run_pot
from mmjee_reasoner.scoring import extract_answer

log = logging.getLogger(__name__)


def seed_base(seed: int, uid: str, sample: int, mode: str) -> str:
    # Independent of the experiment name and of N, so extending N reuses samples.
    return f"{seed}|{mode}|{uid}|s{sample}"


def solve_cot(cfg: dict, exp: dict, engine: Engine, questions) -> list[dict]:
    """Base-paper single-pass CoT, ``n_samples`` independent samples per question."""
    n = int(exp["n_samples"])
    reqs, owners = [], []
    for _, q in questions.iterrows():
        image = relative_image(cfg, q["image_path"])
        prompt = baseline_prompt(q["language"], q["question_type"])
        for s in range(n):
            reqs.append(GenRequest(
                messages=[user_message(prompt, image)],
                sampling={**cfg["sampling"], "seed": derive_seed(seed_base(cfg["seed"], q["uid"],
                                                                           s, "cot"))},
                meta={"uid": q["uid"], "sample": s},
            ))
            owners.append((q, s))
    records = []
    for (q, s), r in zip(owners, engine.generate(reqs)):
        rec = base_record(q, exp["name"], s)
        rec.update({"source": "cot", "calls": 1, "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens, "finish": r.finish_reason,
                    "n_code_blocks": 0, "n_code_ok": 0, "n_code_err": 0,
                    "used_fallback": False, "n_steps": len(split_steps(r.text))})
        records.append(score_into(rec, r.text, q, extract_answer))
    return records


def solve_pot(cfg: dict, exp: dict, engine: Engine, questions,
              exemplars: dict[str, dict] | None = None) -> list[dict]:
    """PoT with the code sandbox (optionally with RAG exemplars in the prompt)."""
    n = int(exp["n_samples"])
    max_blocks = cfg["sandbox"]["max_code_blocks"]
    mode = "pot_rag" if exemplars is not None else "pot"
    trajs, owners = [], []
    for _, q in questions.iterrows():
        ex = (exemplars or {}).get(q["uid"]) or {}
        prompt = pot_prompt(q["language"], q["question_type"], max_blocks, ex.get("text"))
        for s in range(n):
            trajs.append(Trajectory(
                rid=f"{exp['name']}|{q['uid']}|s{s}", prompt=prompt,
                image=relative_image(cfg, q["image_path"]), language=q["language"],
                question_type=q["question_type"],
                seed_base=seed_base(cfg["seed"], q["uid"], s, mode),
            ))
            owners.append((q, s, ex))
    exec_cache = ExecCache(str(artifacts_dir(cfg, "cache", "generations.sqlite")))
    run_pot(trajs, engine, cfg, exec_cache=exec_cache)
    if cfg["sandbox"].get("force_answer"):
        force_answers(trajs, engine, cfg)
    apply_cot_fallback(trajs, engine, cfg)
    records = []
    for (q, s, ex), t in zip(owners, trajs):
        rec = base_record(q, exp["name"], s)
        rec.update({"source": mode, **t.stats(), "n_steps": len(split_steps(t.full_text)),
                    "pot_attempt_text": (t.prefix + t.text) if t.used_fallback else None,
                    "exemplars": ex.get("kb_ids", []), "exemplar_text": ex.get("text")})
        records.append(score_into(rec, t.full_text, q, extract_answer))
    return records


def run_solve_stage(cfg: dict, exp: dict, engine: Engine, limit: int | None = None,
                    tag_engine: Engine | None = None) -> list[dict]:
    if limit is not None:
        from mmjee_reasoner.pipeline.common import set_subset

        set_subset(cfg, limit=limit)
    questions = select_questions(cfg, exp["split"])
    log.info("[%s] %d questions x %d samples (%s)", exp["name"], len(questions),
             exp["n_samples"], exp["mode"])
    if exp["mode"] == "cot":
        records = solve_cot(cfg, exp, engine, questions)
    elif exp["mode"] == "pot":
        exemplars = None
        if exp.get("rag"):
            from mmjee_reasoner.rag.retrieve import exemplars_for_questions

            exemplars = exemplars_for_questions(cfg, questions, tag_engine or engine)
        records = solve_pot(cfg, exp, engine, questions, exemplars)
    else:
        raise ValueError(f"unknown solve mode {exp['mode']}")
    out = candidates_path(cfg, exp["name"])
    write_jsonl(out, records)
    acc = sum(r["correct"] for r in records) / max(1, len(records))
    log.info("[%s] wrote %d candidates to %s (mean accuracy %.3f)", exp["name"], len(records),
             out, acc)
    return records
