"""Loading, cleaning and splitting the mmJEE-Eval dataset.

``prepare_dataset`` downloads ``ArkaMukherjee/mmJEE-Eval`` once, writes every
question image to ``artifacts/data/images/<uid>.png`` and a metadata table to
``artifacts/data/questions.parquet``. All later stages read that table through
``load_questions``.

Split (SOP Sec. IV-A): train = 2019-2024, test = 2025; 2026 is excluded.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.data.adapter import adapter_columns

log = logging.getLogger(__name__)

META_COLUMNS = [
    "question_id", "subject", "question_type", "year", "paper", "language",
    "answer", "answer_sources", "requires_image",
]


def questions_path(cfg: dict) -> Path:
    return artifacts_dir(cfg, "data", "questions.parquet")


def image_path(cfg: dict, uid: str) -> Path:
    return artifacts_dir(cfg, "data", "images", f"{uid}.png")


def parse_question_id(question_id: str) -> dict:
    """Parse ``2019_P1_English_Mathematics_MCQ-Single_q1_MCQ-Single_page20``.

    The id holds two type tokens; the second one (after ``qN``) is consistent
    between English and Hindi versions, so it is used for the twin key.
    """
    parts = question_id.split("_")
    return {
        "id_year": parts[0],
        "id_paper": parts[1],
        "id_language": parts[2],
        "id_subject": parts[3],
        "id_type1": parts[4],
        "qnum": int(parts[5].lstrip("q")),
        "id_type2": parts[6],
    }


def twin_key(question_id: str) -> str:
    """Key shared by the English and Hindi versions of one exam question."""
    p = parse_question_id(question_id)
    return f"{p['id_year']}_{p['id_paper']}_{p['id_subject']}_{p['id_type2']}_q{p['qnum']}"


def assign_split(year: int, cfg: dict) -> str:
    if year in cfg["data"]["train_years"]:
        return "train"
    if year in cfg["data"]["test_years"]:
        return "test"
    return "excluded"


def clean_metadata(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Add uid / twin key / split / adapter columns and fix known metadata errors.

    Returns the cleaned frame and a dict describing every change (logged to
    ``artifacts/data/prepare_report.json``).
    """
    report: dict = {}
    df = df.copy()
    df["year"] = df["year"].astype(int)

    # 1) Unique ids: 4 question_ids occur twice (different questions, same id).
    seen: Counter = Counter()
    uids = []
    for qid in df["question_id"]:
        uids.append(qid if seen[qid] == 0 else f"{qid}__dup{seen[qid]}")
        seen[qid] += 1
    df["uid"] = uids
    report["duplicate_question_ids"] = sorted(q for q, c in seen.items() if c > 1)

    # 2) Known metadata error: 2023 P1 English rows typed MCQ-Single although
    #    their id says MCQ-Multiple, their Hindi twins are MCQ-Multiple and the
    #    gold has several letters. Fix the type so prompt and scorer agree.
    bad = (df["question_type"] == "MCQ-Single") & (df["answer"].str.strip().str.len() > 1)
    report["retyped_mcq_single_to_multiple"] = df.loc[bad, "uid"].tolist()
    df.loc[bad, "question_type"] = "MCQ-Multiple"

    df["twin_key"] = [twin_key(q) + (u[len(q):] if u != q else "")
                      for q, u in zip(df["question_id"], df["uid"])]
    df["qnum"] = df["question_id"].map(lambda q: parse_question_id(q)["qnum"])
    df["split"] = df["year"].map(lambda y: assign_split(int(y), cfg))

    step = float(cfg["data"]["range_grid_step"])
    cols = [adapter_columns(t, a, step) for t, a in zip(df["question_type"], df["answer"])]
    df["expanded_answer"] = [c["expanded_answer"] for c in cols]
    df["acceptable_values"] = [c["acceptable_values"] for c in cols]
    report["expanded_numerical_golds"] = int(df["expanded_answer"].sum())
    return df, report


def prepare_dataset(cfg: dict, force: bool = False) -> pd.DataFrame:
    """Download the HF dataset, export images and write the metadata table."""
    out = questions_path(cfg)
    if out.exists() and not force:
        log.info("dataset already prepared: %s", out)
        return load_questions(cfg)

    from datasets import load_dataset  # heavy import, only needed here

    ds = load_dataset(cfg["paths"]["hf_dataset"], split="train")
    rows = []
    images = []
    for ex in ds:
        rows.append({k: ex[k] for k in META_COLUMNS})
        images.append(ex["image"])
    df, report = clean_metadata(pd.DataFrame(rows), cfg)
    for uid, img in zip(df["uid"], images):
        path = image_path(cfg, uid)
        if force or not path.exists():
            img.convert("RGB").save(path)
    df["image_path"] = [str(image_path(cfg, u).relative_to(cfg["paths"]["artifacts"]))
                        for u in df["uid"]]

    validate_splits(df, cfg)
    report["counts"] = {
        s: int((df["split"] == s).sum()) for s in ("train", "test", "excluded")
    }
    report["test_by_type"] = df[df.split == "test"]["question_type"].value_counts().to_dict()
    df.to_parquet(out, index=False)
    artifacts_dir(cfg, "data", "prepare_report.json").write_text(json.dumps(report, indent=2))
    log.info("prepared dataset: %s", report["counts"])
    return df


def validate_splits(df: pd.DataFrame, cfg: dict) -> None:
    """Assert the SOP split sizes and train/test disjointness."""
    counts = df["split"].value_counts().to_dict()
    for split, n in cfg["data"]["expected_counts"].items():
        if counts.get(split, 0) != n:
            raise AssertionError(f"split '{split}' has {counts.get(split, 0)} rows, expected {n}")
    train, test = df[df.split == "train"], df[df.split == "test"]
    assert not set(train["uid"]) & set(test["uid"]), "uid overlap between train and test"
    assert not set(train["twin_key"]) & set(test["twin_key"]), "twin overlap train/test"
    assert train["year"].max() < test["year"].min(), "train years must precede test years"


def load_questions(cfg: dict, split: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load the prepared metadata table (optionally one split / first ``limit`` rows)."""
    path = questions_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `mmjee prepare` first")
    df = pd.read_parquet(path)
    df["image_path"] = [str(Path(cfg["paths"]["artifacts"]) / p) for p in df["image_path"]]
    if split is not None:
        df = df[df["split"] == split]
    df = df.sort_values(["year", "paper", "language", "subject", "qnum", "uid"])
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)
