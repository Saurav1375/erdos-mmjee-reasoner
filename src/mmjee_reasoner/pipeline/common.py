"""Shared helpers for pipeline stages: candidate files and question selection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.scoring import is_correct

QUESTION_FIELDS = ["uid", "question_id", "twin_key", "year", "paper", "language", "subject",
                   "question_type", "requires_image", "answer"]


def exp_dir(cfg: dict, exp_name: str) -> Path:
    path = artifacts_dir(cfg, "generations", exp_name, "x")
    return path.parent


def set_subset(cfg: dict, limit: int | None = None, pilot: bool = False) -> None:
    """Select the question subset for this process (full / random smoke / fixed pilot)."""
    if pilot:
        cfg["subset"] = {"kind": "pilot"}
    elif limit is not None:
        cfg["subset"] = {"kind": "limit", "n": int(limit)}
    else:
        cfg["subset"] = {"kind": "full"}


def subset_suffix(cfg: dict) -> str:
    """'' for full runs, '.pilot' or '.limitN' otherwise (keeps outputs separate)."""
    sub = cfg.get("subset") or {"kind": "full"}
    if sub["kind"] == "pilot":
        return ".pilot"
    if sub["kind"] == "limit":
        return f".limit{sub['n']}"
    return ""


def subset_cli_args(cfg: dict) -> list[str]:
    sub = cfg.get("subset") or {"kind": "full"}
    if sub["kind"] == "pilot":
        return ["--pilot"]
    if sub["kind"] == "limit":
        return ["--limit", str(sub["n"])]
    return []


def candidates_path(cfg: dict, exp_name: str, suffix: str | None = None) -> Path:
    suffix = subset_suffix(cfg) if suffix is None else suffix
    return exp_dir(cfg, exp_name) / f"candidates{suffix}.jsonl"


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_candidates(cfg: dict, exp_name: str, suffix: str | None = None) -> pd.DataFrame:
    path = candidates_path(cfg, exp_name, suffix)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run the '{exp_name}' stage first")
    return pd.DataFrame(read_jsonl(path))


def pilot_twins(questions: pd.DataFrame, split: str, n_pairs: int, seed: int) -> list[str]:
    """Twin keys of a stratified pilot subset (test: by type; train: by year)."""
    import numpy as np

    df = questions[questions["split"] == split]
    strat = "question_type" if split == "test" else "year"
    pairs = df.groupby("twin_key")[strat].first()
    sizes = pairs.value_counts().sort_index()
    quota = sizes / sizes.sum() * n_pairs
    take = np.floor(quota).astype(int)
    rest = n_pairs - int(take.sum())
    for k in (quota - take).sort_values(ascending=False).index[:rest]:
        take[k] += 1
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for k, n in take.items():
        keys = sorted(pairs[pairs == k].index)
        chosen += list(rng.choice(keys, size=min(int(n), len(keys)), replace=False))
    return sorted(chosen)


def select_questions(cfg: dict, split: str, limit: int | None = None,
                     seed: int | None = None) -> pd.DataFrame:
    """Questions of a split, restricted to the active subset (pilot / random smoke)."""
    df = load_questions(cfg, split)
    sub = cfg.get("subset") or {"kind": "full"}
    if sub["kind"] == "pilot":
        n = cfg["pilot"]["test_pairs" if split == "test" else "train_pairs"]
        keys = set(pilot_twins(load_questions(cfg), split, n, cfg["seed"]))
        return df[df["twin_key"].isin(keys)].reset_index(drop=True)
    if limit is None and sub["kind"] == "limit":
        limit = sub["n"]
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed if seed is not None else cfg["seed"])
        df = df.sort_values("uid").reset_index(drop=True)
    return df


def relative_image(cfg: dict, abs_path: str) -> str:
    return str(Path(abs_path).relative_to(cfg["paths"]["artifacts"]))


def base_record(q: pd.Series | dict, exp_name: str, sample: int) -> dict:
    rec = {k: (q[k].item() if hasattr(q[k], "item") else q[k]) for k in QUESTION_FIELDS}
    rec.update({"exp": exp_name, "sample": sample})
    return rec


def score_into(rec: dict, text: str, question: pd.Series | dict, extract) -> dict:
    """Fill ``text`` / ``pred`` / ``correct`` using the upstream functions."""
    pred = extract(text, question["question_type"])
    rec.update({"text": text, "pred": pred, "correct": is_correct(pred, question)})
    return rec
