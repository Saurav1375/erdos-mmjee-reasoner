"""Solver stage: sample N solutions per question (baseline CoT, PoT, PoT + RAG)."""

from __future__ import annotations

import logging

from mmjee_reasoner.agents.steps import split_steps
from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.llm import Engine, GenRequest, derive_seed, gen_sampling, user_message
from mmjee_reasoner.pipeline.common import (
    base_record,
    candidates_path,
    relative_image,
    score_into,
    select_questions,
    write_jsonl,
)
from mmjee_reasoner.prompts.templates import (
    answer_only_prompt,
    baseline_prompt,
    pot_prompt,
    with_question_text,
    with_transcript,
)
from mmjee_reasoner.sandbox.executor import ExecCache
from mmjee_reasoner.sandbox.pot import (
    Trajectory,
    apply_cot_fallback,
    continue_truncated_results,
    extend_truncated,
    force_answers,
    run_pot,
)
from mmjee_reasoner.scoring import extract_answer

log = logging.getLogger(__name__)


def seed_base(seed: int, uid: str, sample: int, mode: str) -> str:
    # Independent of the experiment name and of N, so extending N reuses samples.
    return f"{seed}|{mode}|{uid}|s{sample}"


def solve_cot(cfg: dict, exp: dict, engine: Engine, questions) -> list[dict]:
    """Base-paper single-pass CoT, ``n_samples`` independent samples per question.

    ``mode: answer_only`` (v3 memorisation probe) uses the same loop with a no-reasoning
    prompt; ``exp["sampling"]`` overrides the default sampling (e.g. greedy, 64 tokens).
    """
    n = int(exp["n_samples"])
    mode = exp.get("mode", "cot")
    prompt_fn = answer_only_prompt if mode == "answer_only" else baseline_prompt
    sampling = gen_sampling({**cfg["sampling"], **exp.get("sampling", {})})
    reqs, owners = [], []
    for _, q in questions.iterrows():
        image = relative_image(cfg, q["image_path"])
        prompt = prompt_fn(q["language"], q["question_type"])
        for s in range(n):
            reqs.append(GenRequest(
                messages=[user_message(prompt, image)],
                sampling={**sampling, "seed": derive_seed(seed_base(cfg["seed"], q["uid"],
                                                                    s, mode))},
                meta={"uid": q["uid"], "sample": s},
            ))
            owners.append((q, s))
    results = engine.generate(reqs)
    if mode == "cot":  # v3 adaptive budget (no-op unless sampling.extend_on_truncation > 0)
        results = continue_truncated_results(reqs, results, engine, cfg)
    records = []
    for (q, s), r in zip(owners, results):
        rec = base_record(q, exp["name"], s)
        rec.update({"source": mode, "calls": 1, "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens, "finish": r.finish_reason,
                    "n_code_blocks": 0, "n_code_ok": 0, "n_code_err": 0,
                    "used_fallback": False, "n_steps": len(split_steps(r.text))})
        records.append(score_into(rec, r.text, q, extract_answer))
    return records


def question_text_of(q) -> str | None:
    """Text of a text-only question (v3 external KB items); None for image questions."""
    t = q.get("question_text") if hasattr(q, "get") else None
    return t if isinstance(t, str) and t.strip() else None


def solve_pot(cfg: dict, exp: dict, engine: Engine, questions,
              exemplars: dict[str, dict] | None = None,
              transcripts: dict[str, str] | None = None) -> list[dict]:
    """PoT with the code sandbox (optionally with RAG exemplars in the prompt).

    v3 options: ``transcripts`` (uid -> English transcription) adds the transcript block
    (transcript-assisted solving); text-only questions carry their text in the prompt.
    v4b option: ``sample_offset`` k numbers the samples k..k+n-1, so they reuse the seeds of
    those samples of another run with the same mode.
    """
    n = int(exp["n_samples"])
    offset = int(exp.get("sample_offset", 0))
    max_blocks = cfg["sandbox"]["max_code_blocks"]
    mode = "pot_rag" if exemplars is not None else "pot"
    if transcripts is not None:
        mode += "_tx"
    # v4 (rag.matched_seeds): RAG runs draw the same seeds as the no-RAG run, so a question
    # without exemplars gets an identical request (cache hit) and the RAG effect is paired.
    seed_mode = mode.replace("pot_rag", "pot") if cfg["rag"].get("matched_seeds") else mode
    trajs, owners = [], []
    for _, q in questions.iterrows():
        ex = (exemplars or {}).get(q["uid"]) or {}
        prompt = pot_prompt(q["language"], q["question_type"], max_blocks, ex.get("text"))
        qtext = question_text_of(q)
        if qtext:
            prompt = with_question_text(prompt, qtext)
        if transcripts is not None and transcripts.get(q["uid"]):
            prompt = with_transcript(prompt, transcripts[q["uid"]])
        image = None if qtext else relative_image(cfg, q["image_path"])
        for s in range(offset, offset + n):
            trajs.append(Trajectory(
                rid=f"{exp['name']}|{q['uid']}|s{s}", prompt=prompt,
                image=image, language=q["language"],
                question_type=q["question_type"],
                seed_base=seed_base(cfg["seed"], q["uid"], s, seed_mode),
                question_text=qtext,
            ))
            owners.append((q, s, ex))
    exec_cache = ExecCache(str(artifacts_dir(cfg, "cache", "generations.sqlite")))
    run_pot(trajs, engine, cfg, exec_cache=exec_cache)
    if cfg["sandbox"].get("force_answer"):
        force_answers(trajs, engine, cfg)
    extend_truncated(trajs, engine, cfg, exec_cache)   # v3 adaptive budget (off by default)
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
    if exp.get("dataset"):  # v3: external text-only questions (KB expansion), e.g. jeebench
        from mmjee_reasoner.data.external import load_external

        questions = select_questions(cfg, "external", df=load_external(cfg, exp["dataset"]))
    else:
        # split "any": every year in the dataset (probes), whatever the train/test assignment.
        # v3 dev runs may restrict years (e.g. 2024 A/B tests); --limit samples after that.
        keys = None
        if exp.get("subset_pairs"):
            # v4: fixed stratified subset of EN/HI twin pairs, drawn on the whole split before
            # the year filter, so experiments split by years share one subset.
            from mmjee_reasoner.data.dataset import load_questions
            from mmjee_reasoner.pipeline.common import pilot_twins

            keys = pilot_twins(load_questions(cfg), exp["split"], int(exp["subset_pairs"]),
                               int(exp.get("subset_seed", cfg["seed"])))
        questions = select_questions(cfg, None if exp["split"] == "any" else exp["split"],
                                     years=exp.get("years"), twin_keys=keys)
    log.info("[%s] %d questions x %d samples (%s)", exp["name"], len(questions),
             exp["n_samples"], exp["mode"])
    if exp["mode"] in ("cot", "answer_only"):
        records = solve_cot(cfg, exp, engine, questions)
    elif exp["mode"] == "pot":
        exemplars = None
        if exp.get("rag"):
            from mmjee_reasoner.rag.retrieve import exemplars_for_questions

            exemplars = exemplars_for_questions(cfg, questions, tag_engine or engine)
        transcripts = None
        if exp.get("transcript"):  # v3 extension: transcript-assisted solving
            from mmjee_reasoner.rag.tagger import load_diagram_transcripts, load_english_transcripts

            view = exp.get("transcript_view", "plain")   # v4b: diagram = diagram-first reading
            transcripts = (load_diagram_transcripts(cfg) if view == "diagram"
                           else load_english_transcripts(cfg))
            missing = [u for u in questions["uid"] if u not in transcripts]
            if missing:
                raise RuntimeError(f"{len(missing)} questions lack an English transcription - "
                                   + ("run `transcribe --view diagram` first" if view == "diagram"
                                      else "run `transcribe` / `translate` first"))
        records = solve_pot(cfg, exp, engine, questions, exemplars, transcripts)
    else:
        raise ValueError(f"unknown solve mode {exp['mode']}")
    out = candidates_path(cfg, exp["name"])
    write_jsonl(out, records)
    acc = sum(r["correct"] for r in records) / max(1, len(records))
    log.info("[%s] wrote %d candidates to %s (mean accuracy %.3f)", exp["name"], len(records),
             out, acc)
    return records
