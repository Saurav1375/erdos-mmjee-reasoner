"""Feature extraction for the learned solution verifier (Module C).

A *pool* is the set of candidate solutions for one question (e.g. the 8 PoT
samples, or 8 originals + 8 corrected). Features per candidate:

* interpretable text/trace features: length, #steps, code blocks run / failed,
  CoT fallback, truncation, hedging/self-doubt phrases;
* answer features: format validity for the question type, numeric magnitude,
  integer-ness, #decimals, #letters;
* sibling agreement (self-consistency signal): share of the other candidates
  in the pool with the same normalised answer, majority flag, #distinct answers;
* question metadata: type, subject, language, diagram required;
* a sentence embedding of the solution tail (optional feature set).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

QTYPES = ["MCQ-Single", "MCQ-Multiple", "Numerical", "Matching"]
SUBJECTS = ["Physics", "Chemistry", "Mathematics"]
_HEDGES = re.compile(r"\b(wait|however|re-?check|mistake|hmm|assum\w*|approximately|"
                     r"not sure|let me re|recalculat\w*|contradict\w*)\b", re.IGNORECASE)
_NUM = re.compile(r"^-?\d+(?:\.\d*)?$")  # upstream extractor may return "3."


def answer_key(pred: str, question_type: str) -> str:
    """Normalised answer used for voting / agreement ("2.50" == "2.5"; "BA" == "AB")."""
    p = str(pred).strip().upper()
    if question_type == "Numerical" and _NUM.match(p):
        return repr(round(float(p), 6))
    if question_type in ("MCQ-Single", "MCQ-Multiple", "Matching"):
        letters = sorted(set(re.findall(r"[ABCD]", p)))
        return "".join(letters) if letters and len(p) <= 8 else p
    return p


def valid_format(pred: str, question_type: str) -> bool:
    p = str(pred).strip().upper()
    if question_type in ("MCQ-Single", "Matching"):
        return bool(re.fullmatch(r"[ABCD]", p))
    if question_type == "MCQ-Multiple":
        return bool(re.fullmatch(r"[ABCD]{1,4}", p))
    return bool(_NUM.match(p))


def _pool_features(group: pd.DataFrame) -> pd.DataFrame:
    keys = [answer_key(p, t) for p, t in zip(group["pred"], group["question_type"])]
    counts = Counter(keys)
    n = len(keys)
    top = counts.most_common()
    majority = top[0][0] if top else ""
    rank = {k: r for r, (k, _) in enumerate(top)}
    return pd.DataFrame({
        "agree_frac": [(counts[k] - 1) / max(1, n - 1) for k in keys],
        "is_majority": [float(k == majority) for k in keys],
        "vote_rank": [rank[k] / max(1, len(top) - 1) if len(top) > 1 else 0.0 for k in keys],
        "distinct_frac": [len(counts) / n] * n,
    }, index=group.index)


def handcrafted_features(df: pd.DataFrame, pool_cols: tuple[str, ...] = ("uid",)) -> pd.DataFrame:
    """Interpretable features for each candidate row of ``df``."""
    text = df["text"].fillna("")
    pred = df["pred"].fillna("").astype(str)
    feats = pd.DataFrame(index=df.index)
    feats["log_chars"] = np.log1p(text.str.len())
    # Tokens of the trajectory that produced THIS text (corrected records store
    # their cumulative cost separately in completion_tokens).
    own = df["sol_tokens"] if "sol_tokens" in df else df["completion_tokens"]
    if "sol_tokens" in df:
        own = own.fillna(df["completion_tokens"])
    feats["log_tokens"] = np.log1p(own.fillna(0).astype(float))
    feats["n_steps"] = df["n_steps"].fillna(0).astype(float).clip(upper=40)
    feats["n_code_blocks"] = df["n_code_blocks"].fillna(0).astype(float)
    feats["n_code_ok"] = df["n_code_ok"].fillna(0).astype(float)
    feats["n_code_err"] = df["n_code_err"].fillna(0).astype(float)
    feats["code_ok_frac"] = feats["n_code_ok"] / feats["n_code_blocks"].replace(0, np.nan)
    feats["code_ok_frac"] = feats["code_ok_frac"].fillna(-1.0)
    feats["used_fallback"] = df["used_fallback"].fillna(False).astype(float)
    feats["truncated"] = (df.get("finish", pd.Series("", index=df.index)) == "length").astype(float)
    feats["n_boxed"] = text.str.count(r"\\boxed").astype(float)
    feats["hedges_per_kchar"] = text.map(lambda t: len(_HEDGES.findall(t))) / (
        text.str.len().clip(lower=1) / 1000)
    feats["valid_format"] = [float(valid_format(p, t)) for p, t in zip(pred, df["question_type"])]
    nums = [float(p) if _NUM.match(p.strip()) else math.nan for p in pred]
    feats["is_number"] = [float(not math.isnan(v)) for v in nums]
    feats["log_abs_value"] = [math.log10(abs(v) + 1e-9) if not math.isnan(v) else 0.0
                              for v in nums]
    feats["is_integer"] = [float(not math.isnan(v) and float(v).is_integer()) for v in nums]
    feats["is_negative"] = [float(not math.isnan(v) and v < 0) for v in nums]
    feats["n_decimals"] = [len(p.split(".")[1]) if "." in p and _NUM.match(p.strip()) else 0
                           for p in pred]
    feats["n_letters"] = [len(set(re.findall(r"[ABCD]", p.upper()))) for p in pred]
    for t in QTYPES:
        feats[f"type_{t}"] = (df["question_type"] == t).astype(float)
    for s in SUBJECTS:
        feats[f"subj_{s}"] = (df["subject"] == s).astype(float)
    feats["lang_hindi"] = (df["language"] == "Hindi").astype(float)
    feats["requires_image"] = df["requires_image"].fillna(False).astype(float)
    pool = pd.concat([_pool_features(g) for _, g in df.groupby(list(pool_cols), sort=False)])
    feats = feats.join(pool)
    return feats.astype(np.float32)


def solution_tail(text: str, n_chars: int) -> str:
    return text[-n_chars:] if len(text) > n_chars else text


def embedding_features(df: pd.DataFrame, model_name: str, tail_chars: int) -> np.ndarray:
    from mmjee_reasoner.embed import embed_texts

    return embed_texts([solution_tail(t or "", tail_chars) for t in df["text"]], model_name)
