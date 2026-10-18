"""External text-only question sources for the v3 knowledge base (Module D, v3).

JEEBench (Arora et al., EMNLP 2023; MIT licence; HF ``daman1209arora/jeebench``) holds 515
JEE Advanced 2016-2023 questions as LaTeX text with official answers (problems whose
diagrams were essential were removed by its authors). v3 uses only the **2016-2018**
editions, which mmJEE-Eval does not contain (2019-2023 would duplicate our own KB years).

The questions are solved by our Solver (PoT, text prompt) and only solutions scored
correct against the official answer enter the KB, exactly like the 2019-2024 entries.
"""

from __future__ import annotations

import json
import logging
import urllib.request

import pandas as pd

from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.data.adapter import adapter_columns

log = logging.getLogger(__name__)

JEEBENCH_URL = ("https://huggingface.co/datasets/daman1209arora/jeebench/resolve/"
                "d070d1113ce365df05903763a2017ec2d0d56118/test.json")   # pinned revision
JEEBENCH_YEARS = (2016, 2017, 2018)
SUBJECTS = {"phy": "Physics", "chem": "Chemistry", "math": "Mathematics"}
TYPES = {"MCQ": "MCQ-Single", "MCQ(multiple)": "MCQ-Multiple", "Numeric": "Numerical",
         "Integer": "Numerical"}


def jeebench_rows(raw: list[dict], years=JEEBENCH_YEARS) -> pd.DataFrame:
    """Raw JEEBench records -> question table in the mmJEE layout (text-only)."""
    rows = []
    for x in raw:
        _, _, year, _, paper = x["description"].split()           # "JEE Adv 2016 Paper 1"
        year = int(year)
        if year not in years:
            continue
        qtype = TYPES[x["type"]]
        subj = SUBJECTS[x["subject"]]
        uid = f"jb{year}_P{paper}_{subj}_{qtype}_i{x['index']}"
        rows.append({"uid": uid, "question_id": uid, "twin_key": uid, "year": year,
                     "paper": f"P{paper}", "language": "English", "subject": subj,
                     "question_type": qtype, "answer": str(x["gold"]).strip(),
                     "requires_image": False, "question_text": x["question"],
                     "split": "external", "image_path": "", "qnum": int(x["index"]),
                     "source_dataset": "jeebench", **adapter_columns(qtype, str(x["gold"]))})
    return pd.DataFrame(rows).sort_values("uid").reset_index(drop=True)


def load_external(cfg: dict, name: str) -> pd.DataFrame:
    if name != "jeebench":
        raise ValueError(f"unknown external dataset {name!r}")
    path = artifacts_dir(cfg, "data", "external", "jeebench_test.json")
    if not path.exists():
        log.info("downloading JEEBench (pinned revision) to %s", path)
        with urllib.request.urlopen(JEEBENCH_URL, timeout=60) as r:  # noqa: S310 (fixed URL)
            path.write_bytes(r.read())
    df = jeebench_rows(json.loads(path.read_text()))
    log.info("JEEBench %s: %d questions", list(JEEBENCH_YEARS), len(df))
    return df
