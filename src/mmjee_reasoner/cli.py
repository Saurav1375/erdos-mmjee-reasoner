"""Command-line interface: ``python -m mmjee_reasoner <stage> [options]`` (or ``mmjee``).

Stages, in pipeline order (see README for the full workflow):

    prepare          download dataset, export images, build splits + scorer columns
    solve --exp E    sample solutions (cot, pot, pot_train, pot_rag)
    tag / transcribe concept tags / transcriptions of question images (RAG)
    build-kb         knowledge base + FAISS index from 2019-2024 solved problems
    correct --exp E  Solver/Critic/Corrector loop (pot_corr_same, pot_corr_cross, full)
    train-verifier   train and select the learned verifier on 2019-2024 samples
    evaluate         SOP Table I + analyses on the 2025 held-out set
    report           Markdown + LaTeX report and figures
    status           what has been generated so far

``--limit N`` runs a stage on a reproducible random subset (smoke tests) and
``--pilot`` on the fixed stratified pilot subset; outputs then get a ``.limitN`` /
``.pilot`` suffix (candidates, KB, verifier, eval, report) and never overwrite full runs.
``--mock`` replaces the VLMs by a deterministic fake model (dry runs, no GPU).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from mmjee_reasoner.config import list_experiments, load_config, load_experiment

log = logging.getLogger("mmjee")


def _setup_logging(cfg: dict, stage: str) -> None:
    from mmjee_reasoner.config import artifacts_dir

    log_path = artifacts_dir(cfg, "logs", f"{time.strftime('%Y%m%d-%H%M%S')}_{stage}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,
    )


def _mock_engine(cfg: dict):
    from mmjee_reasoner.data.dataset import load_questions
    from mmjee_reasoner.llm import MockEngine
    from mmjee_reasoner.llm.mock import FakeVLM, golds_for

    return MockEngine(FakeVLM(golds_for(load_questions(cfg))))


def _engine(cfg: dict, role: str, mock: bool):
    from mmjee_reasoner.llm import make_engine

    return make_engine(cfg, role, mock=_mock_engine(cfg) if mock else None)


def cmd_prepare(cfg, args):
    from mmjee_reasoner.data.dataset import prepare_dataset

    prepare_dataset(cfg, force=args.force)


def cmd_solve(cfg, args):
    from mmjee_reasoner.pipeline.solve import run_solve_stage

    exp = load_experiment(args.exp)
    if exp["kind"] != "solve":
        raise SystemExit(f"'{args.exp}' is a {exp['kind']} experiment; use `mmjee correct`")
    if args.n_samples:
        exp["n_samples"] = args.n_samples
    role = exp.get("model", "solver")  # e.g. "repro" for the reproduction check
    run_solve_stage(cfg, exp, _engine(cfg, role, args.mock), limit=args.limit)


def cmd_correct(cfg, args):
    from mmjee_reasoner.pipeline.correct import run_correct_stage

    exp = load_experiment(args.exp)
    if exp["kind"] != "correct":
        raise SystemExit(f"'{args.exp}' is not a correction experiment")
    mocks = None
    if args.mock:
        m = _mock_engine(cfg)
        mocks = {"solver": m, exp["critic"]: m}
    run_correct_stage(cfg, exp, limit=args.limit, mock_engines=mocks)


def cmd_phase(cfg, args):
    from mmjee_reasoner.pipeline.correct import run_phase

    from mmjee_reasoner.pipeline.common import subset_suffix

    run_phase(cfg, load_experiment(args.exp), args.phase, args.round, subset_suffix(cfg))


def _questions_for(cfg, args):
    from mmjee_reasoner.data.dataset import load_questions
    from mmjee_reasoner.pipeline.common import select_questions

    if args.split == "all":
        import pandas as pd

        return pd.concat([select_questions(cfg, "train"), select_questions(cfg, "test")],
                         ignore_index=True)
    return select_questions(cfg, args.split)


def cmd_tag(cfg, args):
    from mmjee_reasoner.rag.tagger import run_tag_stage

    run_tag_stage(cfg, _questions_for(cfg, args), _engine(cfg, "solver", args.mock))


def cmd_transcribe(cfg, args):
    from mmjee_reasoner.rag.tagger import run_transcribe_stage

    run_transcribe_stage(cfg, _questions_for(cfg, args), _engine(cfg, "solver", args.mock))


def cmd_build_kb(cfg, args):
    from mmjee_reasoner.rag.kb import build_kb

    build_kb(cfg)


def cmd_train_verifier(cfg, args):
    from mmjee_reasoner.verifier.train import train_verifier

    train_verifier(cfg)


def cmd_evaluate(cfg, args):
    from mmjee_reasoner.pipeline.evaluate import run_evaluate_stage

    run_evaluate_stage(cfg, args.limit)


def cmd_report(cfg, args):
    from mmjee_reasoner.report.build import build_report

    build_report(cfg, args.limit)


def cmd_status(cfg, args):
    from mmjee_reasoner.pipeline.common import candidates_path, read_jsonl

    print(f"artifacts: {cfg['paths']['artifacts']}")
    for name in list_experiments():
        p = candidates_path(cfg, name)
        if p.exists():
            recs = read_jsonl(p)
            acc = sum(r["correct"] for r in recs) / max(1, len(recs))
            print(f"  {name:16s} {len(recs):6d} candidates  mean acc {acc:.3f}")
        else:
            print(f"  {name:16s} (not run)")


COMMANDS = {
    "prepare": cmd_prepare, "solve": cmd_solve, "correct": cmd_correct, "phase": cmd_phase,
    "tag": cmd_tag, "transcribe": cmd_transcribe, "build-kb": cmd_build_kb,
    "train-verifier": cmd_train_verifier, "evaluate": cmd_evaluate, "report": cmd_report,
    "status": cmd_status,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmjee", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", action="append", default=[],
                   help="extra YAML merged over configs/base.yaml + models.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, **kw):
        sp = sub.add_parser(name, **kw)
        sp.add_argument("--limit", type=int, default=None, help="random subset size (smoke)")
        sp.add_argument("--pilot", action="store_true",
                        help="fixed stratified pilot subset (configs/base.yaml: pilot)")
        sp.add_argument("--mock", action="store_true", help="use the fake VLM (no GPU)")
        return sp

    add("prepare").add_argument("--force", action="store_true")
    sp = add("solve")
    sp.add_argument("--exp", required=True, choices=list_experiments())
    sp.add_argument("--n-samples", type=int, default=None, help="override samples per question")
    add("correct").add_argument("--exp", required=True, choices=list_experiments())
    sp = add("phase", help="(internal) one correction phase")
    sp.add_argument("--exp", required=True)
    sp.add_argument("--phase", required=True, choices=["critic", "corrector"])
    sp.add_argument("--round", type=int, required=True)
    for name in ("tag", "transcribe"):
        add(name).add_argument("--split", default="all", choices=["all", "train", "test"])
    for name in ("build-kb", "train-verifier", "evaluate", "report", "status"):
        add(name)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(*args.config)
    cfg["_config_files"] = [str(c) for c in args.config]
    from mmjee_reasoner.pipeline.common import set_subset

    set_subset(cfg, limit=args.limit, pilot=args.pilot)
    if args.mock and not os.environ.get("MMJEE_ARTIFACTS"):
        # Keep fake generations away from real results.
        from mmjee_reasoner.config import PROJECT_ROOT

        cfg["paths"]["artifacts"] = str(PROJECT_ROOT / "artifacts_mock")
    _setup_logging(cfg, args.command)
    log.info("artifacts dir: %s", cfg["paths"]["artifacts"])
    t0 = time.time()
    COMMANDS[args.command](cfg, args)
    log.info("%s finished in %.1f min", args.command, (time.time() - t0) / 60)
