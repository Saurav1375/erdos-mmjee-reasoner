"""Correction stage driver (Module B).

Builds one chain per source candidate (``exp['source']``), runs the critic and
corrector phases for up to R rounds, and writes the final candidates.

* Same-model critic (``critic: solver``): everything runs in this process.
* Cross-model critic (``critic: critic_cross``): every phase runs in a fresh
  subprocess (``mmjee phase ...``) so only one model occupies the GPU.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mmjee_reasoner.agents.loop import (
    Chain,
    chain_summary,
    corrector_phase,
    critic_phase,
    load_chains,
    new_solution,
    run_correction_inprocess,
    save_chains,
)
from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.llm import make_engine
from mmjee_reasoner.pipeline.common import (
    candidates_path,
    exp_dir,
    load_candidates,
    write_jsonl,
)
from mmjee_reasoner.sandbox.executor import ExecCache
from mmjee_reasoner.scoring import is_correct

log = logging.getLogger(__name__)


def chains_path(cfg: dict, exp: dict, suffix: str) -> Path:
    return exp_dir(cfg, exp["name"]) / f"chains{suffix}.json"


def init_chains(cfg: dict, exp: dict, suffix: str) -> list[Chain]:
    src = load_candidates(cfg, exp["source"], suffix)
    chains = []
    for _, c in src.iterrows():
        chain = Chain(
            cid=f"{c['uid']}|s{c['sample']}", uid=c["uid"], language=c["language"],
            question_type=c["question_type"],
            image=str(Path("data/images") / f"{c['uid']}.png"),
            seed_base=f"{cfg['seed']}|{exp['name']}|{c['uid']}|s{c['sample']}",
            exemplars=c.get("exemplar_text") if isinstance(c.get("exemplar_text"), str) else None,
        )
        stats = {k: c[k] for k in ("calls", "prompt_tokens", "completion_tokens")}
        chain.solutions.append(new_solution(c["text"], c["question_type"], stats))
        chains.append(chain)
    return chains


def run_phase(cfg: dict, exp: dict, phase: str, round_idx: int, suffix: str) -> int:
    """One phase on the persisted chains (entry point of the `phase` subcommand)."""
    path = chains_path(cfg, exp, suffix)
    chains = load_chains(path)
    if phase == "critic":
        engine = make_engine(cfg, exp["critic"])
        n = critic_phase(chains, engine, cfg, round_idx)
    else:
        engine = make_engine(cfg, "solver")
        exec_cache = ExecCache(str(artifacts_dir(cfg, "cache", "generations.sqlite")))
        n = corrector_phase(chains, engine, cfg, round_idx, exec_cache)
    save_chains(chains, path)
    log.info("[%s] phase %s round %d: %d chains processed", exp["name"], phase, round_idx, n)
    return n


def run_correct_stage(cfg: dict, exp: dict, limit: int | None = None,
                      mock_engines: dict | None = None) -> list[dict]:
    from mmjee_reasoner.pipeline.common import set_subset, subset_cli_args, subset_suffix

    if limit is not None:
        set_subset(cfg, limit=limit)
    suffix = subset_suffix(cfg)
    path = chains_path(cfg, exp, suffix)
    if not path.exists():
        save_chains(init_chains(cfg, exp, suffix), path)
    chains = load_chains(path)

    if exp["critic"] == "solver" or mock_engines is not None:
        mocks = mock_engines or {}
        solver = make_engine(cfg, "solver", mock=mocks.get("solver"))
        critic = solver if exp["critic"] == "solver" else make_engine(
            cfg, exp["critic"], mock=mocks.get(exp["critic"]))
        exec_cache = ExecCache(str(artifacts_dir(cfg, "cache", "generations.sqlite")))
        run_correction_inprocess(chains, solver, critic, cfg, exec_cache, state_path=path)
        save_chains(chains, path)
    else:
        for r in range(cfg["correction"]["max_rounds"]):
            for phase in ("critic", "corrector"):
                cmd = [sys.executable, "-m", "mmjee_reasoner"]
                for c in cfg.get("_config_files", []):
                    cmd += ["--config", c]
                cmd += ["phase", "--exp", exp["name"], "--phase", phase, "--round", str(r)]
                cmd += subset_cli_args(cfg)
                log.info("running %s", " ".join(cmd))
                subprocess.run(cmd, check=True)
            chains = load_chains(path)
            if all(c.done for c in chains):
                break
        for c in chains:
            if not c.done:
                c.done, c.stop_reason = True, "max_rounds"
        save_chains(chains, path)

    records = finalize(cfg, exp, chains, suffix)
    return records


def finalize(cfg: dict, exp: dict, chains: list[Chain], suffix: str) -> list[dict]:
    """Write one candidate per chain (its final solution) plus correction metadata."""
    src = load_candidates(cfg, exp["source"], suffix).set_index(["uid", "sample"])
    qs = load_questions(cfg).set_index("uid")
    records = []
    for c in chains:
        sample = int(c.cid.rsplit("|s", 1)[1])
        base = src.loc[(c.uid, sample)].to_dict()
        q = qs.loc[c.uid]
        final = c.solutions[-1]
        summ = chain_summary(c)
        # The text of `final` came from the original sample unless a correction changed it.
        from_original = all(s.get("correction_failed") or s["text"] == c.solutions[0]["text"]
                            for s in c.solutions[1:]) and final["text"] == c.solutions[0]["text"]
        own_tokens = int(base["completion_tokens"]) if from_original else int(
            final["stats"].get("completion_tokens", base["completion_tokens"]))
        own_finish = base.get("finish") if from_original else final["stats"].get("finish")
        rec = {k: base[k] for k in ("question_id", "twin_key", "year", "paper", "language",
                                    "subject", "question_type", "requires_image", "answer")}
        rec.update({
            "uid": c.uid, "exp": exp["name"], "sample": sample, "source": "corrected",
            "text": final["text"], "pred": final["answer"],
            "correct": is_correct(final["answer"], q),
            "orig_pred": c.solutions[0]["answer"], "orig_correct": bool(base["correct"]),
            "n_steps": final["n_steps"],
            "n_code_blocks": final["stats"].get("n_code_blocks", base.get("n_code_blocks", 0)),
            "n_code_ok": final["stats"].get("n_code_ok", base.get("n_code_ok", 0)),
            "n_code_err": final["stats"].get("n_code_err", base.get("n_code_err", 0)),
            "used_fallback": bool(base.get("used_fallback", False)),
            "correction_failed_last": bool(final.get("correction_failed")),
            "calls": int(base["calls"]) + summ["extra_calls"],
            "prompt_tokens": int(base["prompt_tokens"]) + sum(
                x["prompt_tokens"] for x in c.critiques) + sum(
                s["attempt"]["prompt_tokens"] for s in c.solutions[1:]),
            "completion_tokens": int(base["completion_tokens"]) + summ["extra_completion_tokens"],
            "sol_tokens": own_tokens, "finish": own_finish,
            "critic": exp["critic"],
            "exemplars": base.get("exemplars", []),
            **{f"corr_{k}": v for k, v in summ.items()},
        })
        records.append(rec)
    out = candidates_path(cfg, exp["name"], suffix)
    write_jsonl(out, records)
    df = pd.DataFrame(records)
    log.info("[%s] accuracy before %.3f -> after %.3f (%d chains)", exp["name"],
             df["orig_correct"].mean(), df["correct"].mean(), len(df))
    return records
