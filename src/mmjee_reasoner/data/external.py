"""External text-only question sources for the v3 knowledge base (Module D, v3).

JEEBench (Arora et al., EMNLP 2023; MIT licence; HF ``daman1209arora/jeebench``) holds 515
JEE Advanced 2016-2023 questions as LaTeX text with official answers (problems whose
diagrams were essential were removed by its authors). v3 uses only the **2016-2018**
editions, which mmJEE-Eval does not contain (2019-2023 would duplicate our own KB years).

OlympiadBench (He et al., ACL 2024; Apache-2.0; HF ``Hothan/OlympiadBench``, pinned revision)
is the Phase-4 open-licensed KB extension (W4c): English, text-only, open-ended competition
problems in physics and mathematics. Only single-answer items whose official answer is a
plain number (after removing ``$`` and ``\times 10^{k}``) are used, because the JEE numerical
scorer grades them; at most ``cap`` items (all physics first, then a seeded maths sample). They
get placeholder year 0 (no exam year). SciBench was considered and not used: its Hugging Face
card has no licence and its problems come from copyrighted textbooks.

The questions are solved by our Solver (PoT, text prompt) and only solutions scored
correct against the official answer enter the KB, exactly like the 2019-2024 entries.
"""

from __future__ import annotations

import json
import logging
import re
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


OLYMPIAD_REV = "91184b52131e7fc9455fef848035173aea8cc01a"
OLYMPIAD_FILES = {"Physics": "OE_TO_physics_en_COMP", "Mathematics": "OE_TO_maths_en_COMP"}
OLYMPIAD_CAP = 300


def plain_number(answer: str) -> str | None:
    """'$2022.2$' -> '2022.2'; anything that is not a plain decimal (scientific notation,
    expressions) or is tiny (|x| < 0.01, which a 2-decimal JEE scorer cannot grade) -> None."""
    a = str(answer).strip().strip("$").strip()
    a = re.sub(r"\\(?:mathrm|text)\{[^}]*\}", "", a).replace("~", "").strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", a) and (float(a) == 0 or abs(float(a)) >= 0.01):
        return a
    return None


def olympiad_rows(frames: dict[str, pd.DataFrame], cap: int = OLYMPIAD_CAP,
                  seed: int = 1234) -> pd.DataFrame:
    """OlympiadBench records -> question table in the mmJEE layout (text-only, Numerical)."""
    rows = []
    for subj, df in frames.items():
        for _, x in df.iterrows():
            fa = list(x["final_answer"]) if not isinstance(x["final_answer"], str) else \
                [x["final_answer"]]
            multiple = str(x.get("is_multiple_answer")).lower() == "true"
            gold = plain_number(fa[0]) if len(fa) == 1 and not multiple else None
            if gold is None or str(x.get("modality")) != "Text-only":
                continue
            ctx = x.get("context")
            text = (f"{ctx}\n\n" if isinstance(ctx, str) and ctx.strip() and ctx != "nan"
                    else "") + str(x["question"])
            uid = f"ob_{subj}_Numerical_i{x['id']}"
            rows.append({"uid": uid, "question_id": uid, "twin_key": uid, "year": 0,
                         "paper": "OB", "language": "English", "subject": subj,
                         "question_type": "Numerical", "answer": gold, "requires_image": False,
                         "question_text": text, "split": "external", "image_path": "",
                         "qnum": int(x["id"]), "source_dataset": "olympiadbench",
                         **adapter_columns("Numerical", gold)})
    df = pd.DataFrame(rows)
    phys = df[df["subject"] == "Physics"]
    rest = df[df["subject"] != "Physics"]
    take = max(0, cap - len(phys))
    if len(phys) > cap:
        phys = phys.sample(n=cap, random_state=seed)
    out = pd.concat([phys, rest.sample(n=min(take, len(rest)), random_state=seed)])
    return out.sort_values("uid").reset_index(drop=True)


def load_olympiad(cfg: dict) -> pd.DataFrame:
    frames = {}
    for subj, name in OLYMPIAD_FILES.items():
        path = artifacts_dir(cfg, "data", "external", f"olympiadbench_{name}.parquet")
        if not path.exists():
            url = (f"https://huggingface.co/datasets/Hothan/OlympiadBench/resolve/{OLYMPIAD_REV}/"
                   f"OlympiadBench/{name}/{name}.parquet")
            log.info("downloading OlympiadBench %s (pinned revision) to %s", name, path)
            with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (fixed URL)
                path.write_bytes(r.read())
        frames[subj] = pd.read_parquet(path)
    df = olympiad_rows(frames, int(cfg["rag"].get("olympiad_cap", OLYMPIAD_CAP)), cfg["seed"])
    log.info("OlympiadBench: %d questions (%s)", len(df), df["subject"].value_counts().to_dict())
    return df


def load_external(cfg: dict, name: str) -> pd.DataFrame:
    if name == "olympiadbench":
        return load_olympiad(cfg)
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
