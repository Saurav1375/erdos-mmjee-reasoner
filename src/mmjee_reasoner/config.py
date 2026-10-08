"""Configuration loading.

Configs are plain YAML files under ``configs/``:

* ``base.yaml``    -- paths, dataset, split, sampling defaults, module settings
* ``models.yaml``  -- model ids and vLLM engine settings, keyed by role
* ``experiments/<name>.yaml`` -- one file per evaluated configuration

``load_config`` deep-merges ``base.yaml`` and ``models.yaml`` (and optional
override files) into one nested dict and resolves relative paths against the
project root.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(*overrides: str | Path) -> dict[str, Any]:
    """Load base + models config, then apply override YAML files in order."""
    cfg = deep_merge(_read_yaml(CONFIG_DIR / "base.yaml"), _read_yaml(CONFIG_DIR / "models.yaml"))
    for path in overrides:
        cfg = deep_merge(cfg, _read_yaml(Path(path)))
    # Environment override for the artifacts dir (useful on the lab PC).
    if os.environ.get("MMJEE_ARTIFACTS"):
        cfg["paths"]["artifacts"] = os.environ["MMJEE_ARTIFACTS"]
    for key, value in cfg["paths"].items():
        if isinstance(value, str) and not key.startswith("hf_"):
            p = Path(value)
            cfg["paths"][key] = str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())
    return cfg


def load_experiment(name: str) -> dict[str, Any]:
    """Load ``configs/experiments/<name>.yaml`` (without merging)."""
    path = CONFIG_DIR / "experiments" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown experiment '{name}' ({path})")
    exp = _read_yaml(path)
    exp.setdefault("name", name)
    return exp


def list_experiments() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "experiments").glob("*.yaml"))


def artifacts_dir(cfg: dict, *parts: str) -> Path:
    """Path inside the artifacts directory (parent dirs created)."""
    path = Path(cfg["paths"]["artifacts"]).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
