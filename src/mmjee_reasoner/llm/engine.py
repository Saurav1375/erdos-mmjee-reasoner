"""Inference engines.

* :class:`VLLMEngine` -- offline batched generation with vLLM (lab GPU).
* :class:`MockEngine` -- scripted responses for tests / dry runs.
* :class:`CachedEngine` -- wraps any engine with the persistent cache; the
  underlying engine is only constructed when a cache miss occurs, so fully
  cached stages never load a model.
"""

from __future__ import annotations

import base64
import gc
import logging
import mimetypes
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from mmjee_reasoner.llm.cache import GenerationCache
from mmjee_reasoner.llm.types import GenRequest, GenResult

log = logging.getLogger(__name__)


class Engine(Protocol):
    name: str

    def generate(self, requests: list[GenRequest]) -> list[GenResult]: ...


def _image_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def to_openai_messages(messages: list[dict], artifacts_root: Path) -> list[dict]:
    """Replace our relative image references with base64 ``image_url`` items."""
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            items = []
            for item in content:
                if item["type"] == "image":
                    uri = _image_data_uri(artifacts_root / item["image"])
                    items.append({"type": "image_url", "image_url": {"url": uri}})
                else:
                    items.append(item)
            content = items
        out.append({"role": msg["role"], "content": content})
    return out


class VLLMEngine:
    """vLLM offline engine for one model role (see ``configs/models.yaml``)."""

    def __init__(self, model_cfg: dict, artifacts_root: str | Path, chunk_size: int = 256):
        from vllm import LLM  # imported lazily: not installed on the laptop

        self.name = model_cfg["name"]
        self.artifacts_root = Path(artifacts_root)
        self.chunk_size = chunk_size
        kwargs = dict(
            model=model_cfg["hf_id"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            max_model_len=model_cfg["max_model_len"],
            gpu_memory_utilization=model_cfg["gpu_memory_utilization"],
            max_num_seqs=model_cfg["max_num_seqs"],
            enable_prefix_caching=model_cfg.get("enable_prefix_caching", True),
            kv_cache_dtype=model_cfg.get("kv_cache_dtype", "auto"),
            limit_mm_per_prompt=model_cfg.get("limit_mm_per_prompt", {"image": 1}),
            seed=0,
        )
        if model_cfg.get("mm_processor_kwargs"):
            kwargs["mm_processor_kwargs"] = model_cfg["mm_processor_kwargs"]
        if model_cfg.get("max_num_batched_tokens"):
            kwargs["max_num_batched_tokens"] = model_cfg["max_num_batched_tokens"]
        if model_cfg.get("enforce_eager"):
            kwargs["enforce_eager"] = True
        log.info("loading vLLM model %s", model_cfg["hf_id"])
        self.llm = LLM(**kwargs)

    def _sampling_params(self, s: dict):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=s.get("temperature", 0.0),
            top_p=s.get("top_p", 1.0),
            top_k=s.get("top_k", -1) if s.get("top_k", -1) not in (None, 0) else -1,
            presence_penalty=s.get("presence_penalty", 0.0),
            max_tokens=s["max_tokens"],
            seed=s.get("seed"),
            stop=s.get("stop") or None,
            include_stop_str_in_output=False,
        )

    def _run(self, reqs: list[GenRequest], continue_final: bool) -> list[GenResult]:
        try:
            return self._run_batch(reqs, continue_final)
        except Exception as e:
            # Request validation errors (e.g. prompt longer than max_model_len) abort the
            # whole batch; vLLM raises VLLMValidationError, which is not a ValueError.
            if not (isinstance(e, ValueError) or "Validation" in type(e).__name__):
                raise
            if len(reqs) == 1:
                log.warning("request failed (%s); returning empty result", e)
                return [GenResult(text="", prompt_tokens=0, completion_tokens=0,
                                  finish_reason="error", stop_reason=None)]
            log.warning("batch failed (%s); retrying requests one by one", e)
            return [res for r in reqs for res in self._run([r], continue_final)]

    def _run_batch(self, reqs: list[GenRequest], continue_final: bool) -> list[GenResult]:
        convs = [to_openai_messages(r.messages, self.artifacts_root) for r in reqs]
        params = [self._sampling_params(r.sampling) for r in reqs]
        outs = self.llm.chat(
            convs,
            sampling_params=params,
            use_tqdm=True,
            add_generation_prompt=not continue_final,
            continue_final_message=continue_final,
        )
        results = []
        for o in outs:
            c = o.outputs[0]
            stop_reason = c.stop_reason if isinstance(c.stop_reason, str) else None
            results.append(GenResult(
                text=c.text,
                prompt_tokens=len(o.prompt_token_ids or []),
                completion_tokens=len(c.token_ids),
                finish_reason=c.finish_reason or "stop",
                stop_reason=stop_reason,
            ))
        return results

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        results: list[GenResult | None] = [None] * len(requests)
        for flag in (False, True):
            idx = [i for i, r in enumerate(requests) if r.continue_final == flag]
            for start in range(0, len(idx), self.chunk_size):
                chunk = idx[start : start + self.chunk_size]
                for i, res in zip(chunk, self._run([requests[i] for i in chunk], flag)):
                    results[i] = res
        return results  # type: ignore[return-value]

    def close(self) -> None:
        """Free GPU memory (best effort; stages that switch models use subprocesses)."""
        try:
            del self.llm
        finally:
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass


class MockEngine:
    """Engine driven by a Python callable ``responder(request) -> str``."""

    def __init__(self, responder: Callable[[GenRequest], str], name: str = "mock"):
        self.name = name
        self.responder = responder
        self.calls: list[GenRequest] = []

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        out = []
        for r in requests:
            self.calls.append(r)
            text = self.responder(r)
            stop_reason = None
            for s in r.sampling.get("stop") or []:
                pos = text.find(s)
                if pos >= 0:  # emulate vLLM: cut at the stop string, exclude it
                    text, stop_reason = text[:pos], s
                    break
            prompt_len = sum(
                len(m["content"].split()) if isinstance(m["content"], str)
                else sum(len(i.get("text", "").split()) for i in m["content"])
                for m in r.messages
            )
            out.append(GenResult(text=text, prompt_tokens=prompt_len,
                                 completion_tokens=max(1, len(text.split())),
                                 finish_reason="stop", stop_reason=stop_reason))
        return out


class CachedEngine:
    """Cache-first wrapper. ``offline=True`` raises on any cache miss."""

    def __init__(self, name: str, cache: GenerationCache,
                 factory: Callable[[], Engine] | None = None, offline: bool = False,
                 persist_every: int = 256):
        self.name = name
        self.persist_every = persist_every
        self.cache = cache
        self.factory = factory
        self.offline = offline
        self._engine: Engine | None = None
        self.stats = {"hits": 0, "misses": 0}

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            if self.factory is None or self.offline:
                raise RuntimeError(f"cache miss for model '{self.name}' in offline mode")
            self._engine = self.factory()
        return self._engine

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        if not requests:
            return []
        keys = [r.cache_key(self.name) for r in requests]
        found = self.cache.get_many(list(dict.fromkeys(keys)))
        miss_idx: dict[str, int] = {}
        for i, k in enumerate(keys):
            if k not in found and k not in miss_idx:
                miss_idx[k] = i
        self.stats["hits"] += len(requests) - len(miss_idx)
        self.stats["misses"] += len(miss_idx)
        if miss_idx:
            todo_keys = list(miss_idx.keys())
            log.info("[%s] generating %d requests (%d cached)", self.name, len(todo_keys),
                     len(requests) - len(todo_keys))
            # Persist chunk by chunk so a crash loses at most one chunk of work.
            for start in range(0, len(todo_keys), self.persist_every):
                chunk_keys = todo_keys[start : start + self.persist_every]
                new = self.engine.generate([requests[miss_idx[k]] for k in chunk_keys])
                self.cache.put_many(self.name, list(zip(chunk_keys, new)))
                found.update(zip(chunk_keys, new))
        return [found[k] for k in keys]

    def close(self) -> None:
        if self._engine is not None and hasattr(self._engine, "close"):
            self._engine.close()  # type: ignore[attr-defined]
        self._engine = None
