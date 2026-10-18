"""Concept tagging and transcription of question images with the Solver VLM (Module D).

* Tags (``SUBJECT | TOPIC | METHOD``) are the retrieval query / key. Tagging the
  image avoids relying on raw OCR (SOP III-D) and gives English text for Hindi
  questions as well.
* Transcriptions are used as the "problem" text of knowledge-base exemplars
  and, for 2025 questions, ONLY to de-duplicate the KB against the test set.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.llm import Engine, GenRequest, user_message
from mmjee_reasoner.pipeline.common import read_jsonl, relative_image, write_jsonl
from mmjee_reasoner.prompts.templates import (
    TAGGER_PROMPT,
    TRANSCRIBE_PROMPT,
    TRANSLATE_PROMPT,
    with_question_text,
)

log = logging.getLogger(__name__)

_TAG_RE = re.compile(
    r"SUBJECT\s*:\s*(?P<subject>[^|\n]*)\|\s*TOPIC\s*:\s*(?P<topic>[^|\n]*)\|\s*METHOD\s*:\s*"
    r"(?P<method>[^\n]*)", re.IGNORECASE)


def parse_tag(text: str) -> dict:
    """Parse ``SUBJECT: .. | TOPIC: .. | METHOD: ..``; falls back to the raw text."""
    m = _TAG_RE.search(text)
    if not m:
        clean = " ".join(text.split())[:300]
        return {"subject": "", "topic": clean, "method": "", "parsed": False, "query": clean}
    d = {k: m.group(k).strip(" *.") for k in ("subject", "topic", "method")}
    d["parsed"] = True
    d["query"] = f"{d['topic']}. {d['method']}".strip()
    return d


def tags_path(cfg: dict) -> Path:
    return artifacts_dir(cfg, "rag", "tags.jsonl")


def transcripts_path(cfg: dict) -> Path:
    return artifacts_dir(cfg, "rag", "transcripts.jsonl")


def _run(cfg: dict, questions: pd.DataFrame, engine: Engine, prompt: str, sampling: dict,
         kind: str) -> list[dict]:
    reqs = []
    for _, q in questions.iterrows():
        qtext = q.get("question_text")
        if isinstance(qtext, str) and qtext.strip():  # v3 text-only question (no image)
            msg = user_message(with_question_text(prompt, qtext))
        else:
            msg = user_message(prompt, relative_image(cfg, q["image_path"]))
        reqs.append(GenRequest(messages=[msg], sampling={**sampling, "seed": 0},
                               meta={"uid": q["uid"], "kind": kind}))
    out = []
    for (_, q), r in zip(questions.iterrows(), engine.generate(reqs)):
        rec = {"uid": q["uid"], "split": q["split"], "text": r.text.strip(),
               "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens}
        if kind == "tag":
            rec.update(parse_tag(r.text))
        out.append(rec)
    return out


def _merge_write(path: Path, recs: list[dict]) -> None:
    """Replace the records of these questions, keep all others (e.g. other test years)."""
    new = {r["uid"] for r in recs}
    old = [r for r in read_jsonl(path) if r["uid"] not in new] if path.exists() else []
    write_jsonl(path, old + recs)


def run_tag_stage(cfg: dict, questions: pd.DataFrame, engine: Engine) -> list[dict]:
    recs = _run(cfg, questions, engine, TAGGER_PROMPT, cfg["rag"]["tag_sampling"], "tag")
    _merge_write(tags_path(cfg), recs)
    log.info("tagged %d questions (%d parsed)", len(recs), sum(r["parsed"] for r in recs))
    return recs


def run_transcribe_stage(cfg: dict, questions: pd.DataFrame, engine: Engine) -> list[dict]:
    recs = _run(cfg, questions, engine, TRANSCRIBE_PROMPT, cfg["rag"]["transcribe_sampling"],
                "transcribe")
    _merge_write(transcripts_path(cfg), recs)
    log.info("transcribed %d questions", len(recs))
    return recs


def translations_path(cfg: dict) -> Path:
    return artifacts_dir(cfg, "rag", "transcripts_en.jsonl")


def run_translate_stage(cfg: dict, questions: pd.DataFrame, engine: Engine) -> list[dict]:
    """v3: English translation of the Hindi transcriptions (text only, greedy)."""
    trans = load_transcripts(cfg)
    hi = questions[(questions["language"] == "Hindi") & questions["uid"].isin(trans.keys())]
    samp = {**cfg["rag"]["transcribe_sampling"], "seed": 0}
    reqs = [GenRequest(messages=[user_message(TRANSLATE_PROMPT.format(text=trans[u]["text"]))],
                       sampling=samp, meta={"uid": u, "kind": "translate"}) for u in hi["uid"]]
    recs = [{"uid": u, "text": r.text.strip(), "prompt_tokens": r.prompt_tokens,
             "completion_tokens": r.completion_tokens}
            for u, r in zip(hi["uid"], engine.generate(reqs))]
    _merge_write(translations_path(cfg), recs)
    log.info("translated %d Hindi transcriptions", len(recs))
    return recs


def load_english_transcripts(cfg: dict) -> dict[str, str]:
    """uid -> English transcription (English questions: as transcribed; Hindi: translated)."""
    out = {u: r["text"] for u, r in load_transcripts(cfg).items()}
    p = translations_path(cfg)
    translated = {r["uid"]: r["text"] for r in read_jsonl(p)} if p.exists() else {}
    from mmjee_reasoner.data.dataset import load_questions

    lang = load_questions(cfg).set_index("uid")["language"]
    return {u: (translated.get(u) if lang.get(u) == "Hindi" else t) for u, t in out.items()
            if lang.get(u) != "Hindi" or u in translated}


def load_tags(cfg: dict) -> dict[str, dict]:
    return {r["uid"]: r for r in read_jsonl(tags_path(cfg))}


def load_transcripts(cfg: dict) -> dict[str, dict]:
    return {r["uid"]: r for r in read_jsonl(transcripts_path(cfg))}
