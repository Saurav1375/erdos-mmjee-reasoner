"""LLM access: engines, cache and request types."""

from __future__ import annotations

from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.llm.cache import GenerationCache
from mmjee_reasoner.llm.engine import CachedEngine, Engine, MockEngine, VLLMEngine
from mmjee_reasoner.llm.types import (
    GenRequest,
    GenResult,
    assistant_message,
    derive_seed,
    user_message,
)

__all__ = [
    "CachedEngine", "Engine", "GenRequest", "GenResult", "GenerationCache", "MockEngine",
    "VLLMEngine", "assistant_message", "derive_seed", "make_engine", "user_message",
]

_CACHES: dict[str, GenerationCache] = {}


def get_cache(cfg: dict) -> GenerationCache:
    path = str(artifacts_dir(cfg, "cache", "generations.sqlite"))
    if path not in _CACHES:
        _CACHES[path] = GenerationCache(path)
    return _CACHES[path]


def make_engine(cfg: dict, role: str, offline: bool = False,
                mock: Engine | None = None) -> CachedEngine:
    """Cached engine for a model role in ``configs/models.yaml`` (``solver``, ``critic_cross``).

    ``mock`` replaces the real backend (tests / dry runs). Mock generations are
    cached under ``mock::<model name>`` so they can never be mistaken for real
    model outputs.
    """
    model_cfg = cfg["models"][role]
    if mock is not None:
        name = f"mock::{model_cfg['name']}"
        factory = lambda: mock
    else:
        name = model_cfg["name"]
        factory = lambda: VLLMEngine(model_cfg, cfg["paths"]["artifacts"])
    return CachedEngine(name, get_cache(cfg), factory=factory, offline=offline)
