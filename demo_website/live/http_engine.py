"""Engines for the live demo.

``HTTPEngine`` sends the project's ``GenRequest``s to a ``vllm serve`` OpenAI-compatible
server (one per lab PC) instead of loading vLLM in-process. It keeps the offline engine's
behaviour (``llm/engine.py``): the same message conversion, continue-final-message turns,
per-request seeds and stop strings, and an empty "error" result for an over-long prompt.

``ObservedEngine`` wraps any engine and reports each call to the live page (progress and the
streamed text of the featured sample).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import httpx

from mmjee_reasoner.llm import GenRequest, GenResult
from mmjee_reasoner.llm.engine import to_openai_messages

log = logging.getLogger(__name__)


class LabUnavailable(RuntimeError):
    """The model server could not be reached."""


class Cancelled(RuntimeError):
    """The run was stopped from the page. Raised (never returned as a result), so a stopped
    batch is not written to the generation cache."""


class HTTPEngine:
    def __init__(self, base_url: str, model: str, artifacts_root: str | Path,
                 workers: int = 32, timeout_s: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.artifacts_root = Path(artifacts_root)
        self.workers = workers
        self.timeout = httpx.Timeout(timeout_s, connect=10.0)
        # rid -> callback(delta_text); set by the runner for the featured sample only
        self.stream_to: Callable[[GenRequest], Callable[[str], None] | None] = lambda r: None
        # Stop support: every request is streamed, so closing its response aborts it on the
        # server at once (vLLM frees the slot when the client disconnects).
        self.cancel = threading.Event()
        self._open: set = set()
        self._lock = threading.Lock()

    def abort(self) -> None:
        self.cancel.set()
        with self._lock:
            for resp in list(self._open):
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass

    def healthy(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/health", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=min(self.workers, len(requests))) as pool:
            return list(pool.map(self._one, requests))

    def _body(self, r: GenRequest) -> dict:
        s = r.sampling
        top_k = s.get("top_k", -1)
        return {
            "model": self.model,
            "messages": to_openai_messages(r.messages, self.artifacts_root),
            "temperature": s.get("temperature", 0.0),
            "top_p": s.get("top_p", 1.0),
            "top_k": top_k if top_k not in (None, 0) else -1,
            "presence_penalty": s.get("presence_penalty", 0.0),
            "max_tokens": s["max_tokens"],
            "seed": s.get("seed"),
            "stop": s.get("stop") or None,
            "include_stop_str_in_output": False,
            "add_generation_prompt": not r.continue_final,
            "continue_final_message": r.continue_final,
        }

    def _one(self, r: GenRequest) -> GenResult:
        # A 5xx is retried: vLLM answers some internal errors (e.g. its multi-modal cache going
        # out of sync after a rejected request) with "retry". After the last try the run stops
        # with an error rather than returning an empty result, which the cache would keep.
        for attempt in range(3):
            try:
                return self._try(r)
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500 or attempt == 2:
                    raise
                log.warning("model server error %s, retrying (%d)", e.response.status_code, attempt + 1)
                time.sleep(2 * (attempt + 1))
        raise AssertionError("unreachable")

    def _try(self, r: GenRequest) -> GenResult:
        if self.cancel.is_set():
            raise Cancelled("stopped")
        body = self._body(r)
        sink = self.stream_to(r)
        url = f"{self.base_url}/v1/chat/completions"
        try:
            return self._stream(url, body, sink)
        except (Cancelled, httpx.HTTPStatusError):
            raise
        except Exception as e:  # noqa: BLE001
            if self.cancel.is_set():          # the response was closed by abort()
                raise Cancelled("stopped") from e
            if isinstance(e, (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError,
                              httpx.ConnectTimeout)):
                raise LabUnavailable(f"model server {self.base_url} unreachable: {e}") from e
            raise

    def _stream(self, url: str, body: dict, sink: Callable[[str], None] | None) -> GenResult:
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}
        text, finish, stop, usage = [], None, None, {}
        with httpx.stream("POST", url, json=body, timeout=self.timeout) as resp:
            if resp.status_code == 400:       # e.g. prompt longer than max_model_len
                resp.read()
                log.warning("request rejected (%s); returning empty result", resp.text[:200])
                return GenResult(text="", prompt_tokens=0, completion_tokens=0,
                                 finish_reason="error", stop_reason=None)
            resp.raise_for_status()
            with self._lock:
                self._open.add(resp)
            try:
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    d = json.loads(data)
                    if d.get("usage"):
                        usage = d["usage"]
                    for c in d.get("choices") or []:
                        delta = (c.get("delta") or {}).get("content")
                        if delta:
                            text.append(delta)
                            if sink:
                                sink(delta)
                        if c.get("finish_reason"):
                            finish, stop = c["finish_reason"], c.get("stop_reason")
            finally:
                with self._lock:
                    self._open.discard(resp)
        if self.cancel.is_set():
            raise Cancelled("stopped")
        return self._result("".join(text), finish, stop, usage)

    @staticmethod
    def _result(text: str, finish: str | None, stop, usage: dict) -> GenResult:
        return GenResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason="length" if finish == "length" else "stop",
            stop_reason=stop if isinstance(stop, str) else None,
        )


class ObservedEngine:
    """Reports every call of the wrapped engine to ``emit`` (no change to the results)."""

    def __init__(self, inner, role: str, emit: Callable[[dict], None],
                 featured: Callable[[GenRequest], bool] = lambda r: False):
        self.inner, self.role, self.emit, self.featured = inner, role, emit, featured

    @property
    def name(self) -> str:   # critic_phase records the model name
        return getattr(self.inner, "name", self.role)

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        kinds = sorted({str(r.meta.get("kind") or ("critic" if "round" in r.meta and "cid" in r.meta
                                                     else "solve")) for r in requests})
        self.emit({"type": "calls", "role": self.role, "n": len(requests), "kinds": kinds})
        for r in requests:
            if self.featured(r):
                prior = r.messages[-1]["content"] if r.continue_final else ""
                self.emit({"type": "stream_start", "rid": r.meta.get("rid") or r.meta.get("cid", ""),
                           "role": self.role,
                           "text": prior if isinstance(prior, str) else ""})
        results = self.inner.generate(requests)
        for r, res in zip(requests, results):
            ev = {"type": "result", "role": self.role, "rid": r.meta.get("rid") or r.meta.get("cid"),
                  "turn": r.meta.get("turn"), "tokens": res.completion_tokens,
                  "finish": res.finish_reason}
            if self.featured(r):
                prior = r.messages[-1]["content"] if r.continue_final else ""
                ev["text"] = (prior if isinstance(prior, str) else "") + res.text
            self.emit(ev)
        return results
