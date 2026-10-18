"""v3 infrastructure: config-driven split, test-year tags, configurable dirs, merged tags."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from mmjee_reasoner.config import CONFIG_DIR, load_config
from mmjee_reasoner.data.dataset import clean_metadata, image_path, load_questions, questions_path
from mmjee_reasoner.pipeline.common import candidates_path, experiment_split, year_tag

from conftest import make_raw_rows


@pytest.fixture
def dataset26(cfg) -> pd.DataFrame:
    """Synthetic dataset including a 2026 year (excluded by the base config)."""
    df, _ = clean_metadata(make_raw_rows(years=range(2019, 2027)), cfg)
    for uid in df["uid"]:
        Image.new("RGB", (32, 16), "white").save(image_path(cfg, uid))
    df["image_path"] = [str(image_path(cfg, u).relative_to(cfg["paths"]["artifacts"]))
                        for u in df["uid"]]
    df.to_parquet(questions_path(cfg), index=False)
    return df


def _with_2026(cfg: dict) -> dict:
    cfg["data"].update(test_years=[2026], exclude_years=[2025], test_tag=".y2026")
    return cfg


def test_split_follows_config(cfg, dataset26):
    base = load_questions(cfg)
    assert set(base.loc[base.split == "test", "year"]) == {2025}
    assert set(base.loc[base.split == "excluded", "year"]) == {2026}
    _with_2026(cfg)
    q = load_questions(cfg)
    assert set(q.loc[q.split == "test", "year"]) == {2026}
    assert set(q.loc[q.split == "train", "year"]) == set(range(2019, 2025))
    assert set(q.loc[q.split == "excluded", "year"]) == {2025}


def test_split_unchanged_for_v2_config(real_cfg):
    """The recomputed split equals the split stored at `prepare` time (v2 behaviour)."""
    stored = pd.read_parquet(questions_path(real_cfg))
    loaded = load_questions(real_cfg).set_index("uid")["split"]
    assert (stored.set_index("uid")["split"] == loaded.loc[stored["uid"]].values).all()


def test_year_tag_only_on_test_experiments(cfg):
    assert experiment_split("cot") == "test"
    assert experiment_split("full") == "test"          # correction inherits its source
    assert experiment_split("pot_train") == "train"
    assert year_tag(cfg, "cot") == ""                    # default: no tag (v2 paths)
    _with_2026(cfg)
    assert candidates_path(cfg, "cot").name == "candidates.y2026.jsonl"
    assert candidates_path(cfg, "full").name == "candidates.y2026.jsonl"
    assert candidates_path(cfg, "pot_train").name == "candidates.jsonl"


def test_v3_configs_load():
    c = load_config(CONFIG_DIR / "v3.yaml", CONFIG_DIR / "v3_test2026.yaml")
    assert Path(c["paths"]["artifacts"]).name == "artifacts_v3"
    assert c["data"]["test_years"] == [2026] and c["data"]["test_tag"] == ".y2026"
    assert c["verifier"]["dir_name"] == "verifier_v3" and c["rag"]["kb_name"] == "kb_v3"
    base = load_config()
    assert base["data"]["test_years"] == [2025] and not base["data"].get("test_tag")


def test_dir_names_configurable(cfg):
    from mmjee_reasoner.rag.kb import kb_dir
    from mmjee_reasoner.verifier.train import verifier_dir

    assert verifier_dir(cfg).name == "verifier" and kb_dir(cfg).name == "kb"
    cfg["verifier"]["dir_name"], cfg["rag"]["kb_name"] = "verifier_v3", "kb_v3"
    assert verifier_dir(cfg).name == "verifier_v3" and kb_dir(cfg).name == "kb_v3"


def test_limit_runs_read_full_kb_and_build_kb_refuses_limit(cfg):
    import pytest

    from mmjee_reasoner.pipeline.common import set_subset
    from mmjee_reasoner.rag.kb import build_kb, kb_dir

    set_subset(cfg, limit=4)                       # smoke run: same KB as the full run
    assert kb_dir(cfg).name == "kb"
    with pytest.raises(ValueError, match="--limit"):
        build_kb(cfg)
    set_subset(cfg, pilot=True)                    # the pilot keeps its own KB
    assert kb_dir(cfg).name == "kb.pilot"


def test_tag_stage_merges(cfg, dataset26):
    from mmjee_reasoner.llm import MockEngine, make_engine
    from mmjee_reasoner.llm.mock import FakeVLM, golds_for
    from mmjee_reasoner.rag.tagger import load_tags, run_tag_stage

    qs = load_questions(cfg)
    eng = make_engine(cfg, "solver", mock=MockEngine(FakeVLM(golds_for(qs))))
    run_tag_stage(cfg, qs[qs.year == 2025], eng)
    run_tag_stage(cfg, qs[qs.year == 2026], eng)
    tags = load_tags(cfg)
    assert set(qs.loc[qs.year >= 2025, "uid"]) == set(tags)   # 2025 kept after the 2026 run


def test_correction_subset_pairs(cfg, dataset):
    from mmjee_reasoner.config import load_experiment
    from mmjee_reasoner.llm import MockEngine, make_engine
    from mmjee_reasoner.llm.mock import FakeVLM, golds_for
    from mmjee_reasoner.pipeline.correct import init_chains
    from mmjee_reasoner.pipeline.solve import run_solve_stage

    qs = load_questions(cfg)
    eng = make_engine(cfg, "solver", mock=MockEngine(FakeVLM(golds_for(qs))))
    run_solve_stage(cfg, load_experiment("pot_train"), eng)
    exp = {"name": "t", "kind": "correct", "source": "pot_train", "critic": "solver",
           "subset_pairs": 6}
    chains = init_chains(cfg, exp, "")
    twins = {c.uid.split("_")[0] for c in chains}
    assert len({qs.set_index("uid").loc[c.uid, "twin_key"] for c in chains}) == 6
    assert len(chains) == 6 * 2 * 8                      # 6 pairs x EN/HI x 8 samples
    assert twins <= {str(y) for y in range(2019, 2025)}


def test_embed_device_env(monkeypatch):
    """MMJEE_EMBED_DEVICE pins the sentence embedder (lab: cpu, so vLLM keeps the GPU)."""
    import sys
    import types

    from mmjee_reasoner import embed

    seen = {}

    class FakeST:
        device = "x"

        def __init__(self, name, device=None):
            seen["device"] = device

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeST))
    monkeypatch.setenv("MMJEE_EMBED_DEVICE", "cpu")
    embed._model.cache_clear()
    embed._model("fake-model")
    assert seen["device"] == "cpu"
    monkeypatch.delenv("MMJEE_EMBED_DEVICE")
    embed._model.cache_clear()
    embed._model("fake-model")
    assert seen["device"] is None
    embed._model.cache_clear()
