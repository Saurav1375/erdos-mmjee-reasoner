"""Request / result types shared by all engines.

Messages follow the OpenAI chat format. Images are referenced by a path
*relative to the artifacts directory* (``{"type": "image", "image": "data/images/x.png"}``)
so that cache keys are identical on the laptop and on the lab PC.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def user_message(text: str, image: str | None = None) -> dict:
    """A user turn with an optional image placed before the text (as in the base paper)."""
    content: list[dict] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def assistant_message(text: str) -> dict:
    return {"role": "assistant", "content": text}


@dataclass
class GenRequest:
    messages: list[dict]
    sampling: dict[str, Any]
    # True: the last message is a partial assistant turn that the model continues.
    continue_final: bool = False
    meta: dict[str, Any] = field(default_factory=dict)  # not part of the cache key

    def cache_key(self, model_name: str) -> str:
        payload = {
            "model": model_name,
            "messages": self.messages,
            "sampling": self.sampling,
            "continue_final": self.continue_final,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str              # "stop" (EOS or stop string) or "length"
    stop_reason: str | None = None  # matched stop string, None for EOS
    cached: bool = False

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("cached")
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, blob: str) -> GenResult:
        return cls(**json.loads(blob), cached=True)


def derive_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from arbitrary parts (uid, sample index, turn, ...)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF
