"""Method cards for RAG exemplars (Module D, Phase 4 / W4b).

v2/v3 exemplars show a KB problem plus its full worked solution (up to 3,000 characters). A
method card is a short summary of the same solution (about 150-250 tokens): the method, when it
applies, the plan, a common mistake and the key equations - but never the old problem's final
answer. Cards are written once per KB entry by the Solver (text only, greedy) and stored in
``rag/cards.jsonl``, keyed by a hash of the entry's problem + solution, so the same entry in
several KBs (e.g. the 2024-dev KB and kb_v3) shares one card. ``rag.exemplar_format: card``
makes retrieval show cards instead of solutions; the default (``solution``) is the v3 prompt.
"""

from __future__ import annotations

import hashlib
import logging
import re

from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.llm import Engine, GenRequest, user_message
from mmjee_reasoner.pipeline.common import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

CARD_PROMPT = (
    "Below is a solved JEE Advanced problem. Write a short METHOD CARD that would help a student "
    "solve a DIFFERENT problem of the same kind. Use exactly these five lines:\n"
    "Method: <the concept or technique>\n"
    "Applicable when: <conditions that signal this method>\n"
    "Plan: <3-5 steps joined by ->, no numbers from this problem>\n"
    "Key equations: <the general formulas used>\n"
    "Common mistake: <the trap to avoid>\n"
    "Do NOT state this problem's final answer, its numerical values or option letters. "
    "At most 150 words.\n\nProblem:\n{problem}\n\nWorked solution:\n{solution}"
)
_BOXED = re.compile(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
_OPTION = re.compile(r"\b(?:option|answer|choice)s?\s*(?:is|are|:)?\s*\(?[A-D](?:\s*(?:,|and)\s*"
                     r"\(?[A-D]\)?)*\)?", re.IGNORECASE)


def card_key(entry: dict) -> str:
    h = hashlib.sha256((entry["problem"] + "\0" + entry["solution"]).encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def sanitize_card(text: str, answer: str | None) -> str:
    """Drop boxed expressions, option-letter answers and the gold answer value from a card."""
    t = _BOXED.sub("[answer omitted]", text.strip())
    t = _OPTION.sub("[answer omitted]", t)
    a = (answer or "").strip()
    if a and re.fullmatch(r"-?\d+(?:\.\d+)?", a):
        t = re.sub(rf"(?<![\d.]){re.escape(a)}(?![\d.])", "[answer omitted]", t)
    return t


def cards_path(cfg: dict):
    return artifacts_dir(cfg, "rag", "cards.jsonl")


def load_cards(cfg: dict) -> dict[str, str]:
    p = cards_path(cfg)
    return {r["key"]: r["card"] for r in read_jsonl(p)} if p.exists() else {}


def _answers(cfg: dict) -> dict[str, str]:
    from mmjee_reasoner.data.dataset import load_questions

    out = dict(zip(load_questions(cfg)["uid"], load_questions(cfg)["answer"].astype(str)))
    for ext in cfg["rag"].get("extra_kb") or []:
        from mmjee_reasoner.data.external import load_external

        df = load_external(cfg, ext["dataset"])
        out.update(dict(zip(df["uid"], df["answer"].astype(str))))
    return out


def run_cards_stage(cfg: dict, engine: Engine) -> int:
    """Write a card for every entry of the configured KB that has none yet (merge, resumable)."""
    from mmjee_reasoner.rag.kb import load_kb

    entries, _, _ = load_kb(cfg)
    have = load_cards(cfg)
    todo = [e for e in entries if card_key(e) not in have]
    answers = _answers(cfg)
    samp = {**cfg["rag"]["transcribe_sampling"], "max_tokens": 400, "seed": 0}
    reqs = [GenRequest(messages=[user_message(CARD_PROMPT.format(problem=e["problem"].strip(),
                                                                 solution=e["solution"].strip()))],
                       sampling=samp, meta={"kb_id": e["kb_id"], "kind": "card"}) for e in todo]
    results = engine.generate(reqs) if reqs else []
    recs = [{"key": card_key(e), "kb_id": e["kb_id"],
             "card": sanitize_card(r.text, answers.get(e["uid"])),
             "completion_tokens": r.completion_tokens} for e, r in zip(todo, results)]
    old = read_jsonl(cards_path(cfg)) if cards_path(cfg).exists() else []
    write_jsonl(cards_path(cfg), old + recs)
    log.info("method cards: %d new, %d already present (%d KB entries)", len(recs),
             len(entries) - len(todo), len(entries))
    return len(recs)
