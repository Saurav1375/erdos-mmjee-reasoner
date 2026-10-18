"""v3 memorisation probes (contamination analysis, not part of the framework).

1. Answer-only probe: ``solve --exp probe_answer_q3`` / ``probe_answer_q25`` (mode
   ``answer_only``). The model must answer without reasoning; markedly higher accuracy on
   years it may have seen than on years it cannot have seen hints at memorised answers.
2. Completion probe (this module): the model gets the first half of a question's
   transcription (text only, no image) and must continue it word for word. Verbatim
   continuations (high ROUGE-L / 5-gram overlap) on seen years, but not on unseen years,
   hint at memorised question text.

Seen vs. unseen: Qwen2.5-VL (released Jan 2025) cannot have seen 2025/2026; Qwen3-VL
(Oct 2025) cannot have seen 2026. Limitation: the reference text is the Qwen3-VL
transcription of the question image, not the official text.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from mmjee_reasoner.config import artifacts_dir
from mmjee_reasoner.llm import Engine, GenRequest, user_message
from mmjee_reasoner.pipeline.common import write_jsonl
from mmjee_reasoner.prompts.templates import COMPLETION_PROMPT

log = logging.getLogger(__name__)

_WORD = re.compile(r"\S+")
MIN_WORDS = 20          # shorter transcriptions are skipped (too little to continue)
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 768, "seed": 0}


def split_half(text: str) -> tuple[str, str]:
    """Split a text at the middle word boundary -> (prefix, reference continuation)."""
    words = _WORD.findall(text)
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+|[^\w\s]", text)]


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1 on lower-cased word/punctuation tokens."""
    c, r = _tokens(candidate), _tokens(reference)
    if not c or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for tc in c:
        cur = [0]
        for j, tr in enumerate(r):
            cur.append(prev[j] + 1 if tc == tr else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(c), lcs / len(r)
    return 2 * p * rec / (p + rec)


def ngram_overlap(candidate: str, reference: str, n: int = 5) -> float:
    """Share of the reference's n-grams that also occur in the candidate."""
    c, r = _tokens(candidate), _tokens(reference)
    ref = {tuple(r[i:i + n]) for i in range(len(r) - n + 1)}
    if not ref:
        return 0.0
    cand = {tuple(c[i:i + n]) for i in range(len(c) - n + 1)}
    return len(ref & cand) / len(ref)


def completion_probe(cfg: dict, engine: Engine, questions: pd.DataFrame,
                     transcripts: dict[str, dict], role: str) -> list[dict]:
    """Run the completion probe on English questions; write probes/completion_<role>.jsonl."""
    rows = []
    for _, q in questions[questions["language"] == "English"].iterrows():
        t = transcripts.get(q["uid"])
        if t is None or len(_WORD.findall(t["text"])) < MIN_WORDS:
            continue
        prefix, ref = split_half(t["text"])
        rows.append((q, prefix, ref))
    reqs = [GenRequest(messages=[user_message(COMPLETION_PROMPT.format(prefix=p))],
                       sampling=dict(SAMPLING), meta={"uid": q["uid"]}) for q, p, _ in rows]
    out = []
    for (q, prefix, ref), r in zip(rows, engine.generate(reqs)):
        out.append({"uid": q["uid"], "year": int(q["year"]), "subject": q["subject"],
                    "question_type": q["question_type"], "role": role,
                    "prefix_words": len(prefix.split()), "ref_words": len(ref.split()),
                    "continuation": r.text, "rouge_l": rouge_l(r.text, ref),
                    "ngram5": ngram_overlap(r.text, ref, 5)})
    write_jsonl(artifacts_dir(cfg, "probes", f"completion_{role}.jsonl"), out)
    log.info("completion probe (%s): %d questions", role, len(out))
    return out


def summarise_by_year(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean of ``cols`` per year (+ count)."""
    g = df.groupby("year")
    return g[cols].mean().join(g.size().rename("n"))
