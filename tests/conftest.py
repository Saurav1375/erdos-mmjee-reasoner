"""Shared fixtures: temporary config, a small synthetic mmJEE-like dataset, fake embedder."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mmjee_reasoner.config import load_config
from mmjee_reasoner.data.dataset import clean_metadata, image_path, questions_path

SUBJECTS = ["Physics", "Chemistry", "Mathematics"]
TYPES = ["Numerical", "MCQ-Single", "MCQ-Multiple", "Matching"]
ANSWERS = {"Numerical": ["4", "2.5", "8.70 TO 9.10", "80 OR 150 OR 220", "0"],
           "MCQ-Single": ["A", "B", "C", "D"], "MCQ-Multiple": ["AB", "ACD", "B", "BCD"],
           "Matching": ["A", "C", "D", "B"]}


@pytest.fixture
def cfg(tmp_path) -> dict:
    c = load_config()
    c["paths"]["artifacts"] = str(tmp_path / "artifacts")
    c["paths"]["report"] = str(tmp_path / "report")
    c["sandbox"]["workers"] = 4
    return c


def make_raw_rows(per_year: int = 8, years=range(2019, 2026)) -> pd.DataFrame:
    rows = []
    for year in years:
        for i in range(per_year):
            subj = SUBJECTS[i % 3]
            qtype = TYPES[i % 4]
            ans = ANSWERS[qtype][(i + year) % len(ANSWERS[qtype])]
            for lang in ("English", "Hindi"):
                qid = f"{year}_P{1 + i % 2}_{lang}_{subj}_{qtype}_q{i + 1}_{qtype}_page{i + 3}"
                rows.append({"question_id": qid, "subject": subj, "question_type": qtype,
                             "year": str(year), "paper": f"P{1 + i % 2}", "language": lang,
                             "answer": ans, "answer_sources": "{}",
                             "requires_image": bool(i % 3 == 0)})
    return pd.DataFrame(rows)


@pytest.fixture
def dataset(cfg) -> pd.DataFrame:
    """Write a synthetic prepared dataset (as `mmjee prepare` would) into cfg's artifacts."""
    df, _ = clean_metadata(make_raw_rows(), cfg)
    for uid in df["uid"]:
        Image.new("RGB", (32, 16), "white").save(image_path(cfg, uid))
    df["image_path"] = [str(image_path(cfg, u).relative_to(cfg["paths"]["artifacts"]))
                        for u in df["uid"]]
    df.to_parquet(questions_path(cfg), index=False)
    return df


class FakeSentenceModel:
    """Deterministic bag-of-words hashing embedder (no model download in tests)."""

    device = "cpu"

    def encode(self, texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
               convert_to_numpy=True):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in str(t).lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
                out[i, h % 64] += 1.0
        out += 1e-6
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def fake_embedder(monkeypatch):
    import mmjee_reasoner.embed as emb

    emb._model.cache_clear()
    monkeypatch.setattr(emb, "_model", lambda name: FakeSentenceModel())
    yield


REAL_ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


@pytest.fixture
def real_cfg() -> dict:
    c = load_config()
    if not (Path(c["paths"]["artifacts"]) / "data" / "questions.parquet").exists():
        pytest.skip("real dataset not prepared (run `mmjee prepare`)")
    return c
