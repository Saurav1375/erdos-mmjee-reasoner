"""Neural verifier (Module C, Phase 4 / W3): a fine-tuned cross-encoder that reads the solution.

The v2/v3 verifiers score a candidate from hand-made features (agreement, format, length, critic
signals) plus a frozen sentence embedding. This one fine-tunes a transformer encoder
(``verifier_neural.model``, default ModernBERT-base, Apache-2.0) on

    [CLS] English question transcript [SEP] type + extracted answer + finished/cut off
          + the last tokens of the solution [SEP]  ->  P(correct)

Training data are the same 2019-2024 pools as verifier v3 (``training_pool``), optionally plus
the Phase-4 v3-style pools (``verifier_neural.extra_pools``). Model quality is measured by
leave-one-year-out over 2019-2024 (one fold per year, resumable: each fold's out-of-year scores
are saved as soon as it finishes). No 2025/2026 data is used for training or selection; scoring
a test pool is a separate, later call (V4_PREREG.md).
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KEY_COLS = ["uid", "sample", "is_corrected", "pool"]


@dataclass
class NeuralConfig:
    model: str = "answerdotai/ModernBERT-base"
    max_len: int = 1024
    question_tokens: int = 256
    tail_chars: int = 8000          # characters of the solution tail fed to the tokenizer
    epochs: int = 2
    lr: float = 3e-5
    batch_size: int = 8
    grad_accum: int = 2
    warmup: float = 0.06
    weight_decay: float = 0.01
    eval_batch_size: int = 32
    seed: int = 1234

    @classmethod
    def from_cfg(cls, cfg: dict) -> "NeuralConfig":
        raw = dict(cfg.get("verifier_neural", {}))
        raw.setdefault("seed", cfg["seed"])
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def neural_dir(cfg: dict, tag: str) -> Path:
    from mmjee_reasoner.pipeline.common import subset_suffix

    d = Path(cfg["paths"]["artifacts"]) / cfg["verifier"].get("dir_name", "verifier_v4") / \
        f"neural_{tag}{subset_suffix(cfg)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def v4_pools(cfg: dict, pairs: list[list[str]]) -> pd.DataFrame:
    """Phase-4 v3-style pools: [[originals_exp, corrected_exp], ...] -> one prepared frame."""
    from mmjee_reasoner.pipeline.common import load_candidates
    from mmjee_reasoner.verifier.v3 import prepare_pool

    frames = []
    for orig, corr in pairs:
        parts = [load_candidates(cfg, orig)]
        if corr:
            try:
                parts.append(load_candidates(cfg, corr))
            except FileNotFoundError:
                log.warning("no corrected candidates for %s yet - originals only", corr)
        frames.append(pd.concat(parts, ignore_index=True))
    return prepare_pool(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


def training_frame(cfg: dict, with_extra: bool) -> pd.DataFrame:
    """v3 training pools (pool = "v3") plus, optionally, the Phase-4 pools (pool = "v4")."""
    from mmjee_reasoner.verifier.v3 import training_pool

    base = training_pool(cfg).assign(pool="v3")
    frames = [base]
    pairs = cfg.get("verifier_neural", {}).get("extra_pools") or []
    if with_extra and pairs:
        extra = v4_pools(cfg, pairs).assign(pool="v4")
        held_out = set(cfg["data"]["test_years"]) | set(cfg["rag"].get("dedup_years", []))
        assert not extra["year"].isin(held_out).any(), "held-out years in neural training data"
        frames.append(extra)
    df = pd.concat(frames, ignore_index=True)
    df["sample"] = df["sample"].astype(int)
    assert not df.duplicated(KEY_COLS).any(), "duplicate candidate keys in the training frame"
    return df


def _text(x) -> str:
    return x if isinstance(x, str) else ""


def row_texts(df: pd.DataFrame, transcripts: dict[str, str], tail_chars: int
              ) -> tuple[list[str], list[str], list[str]]:
    """(question, header, solution tail) strings per row."""
    qs, heads, tails = [], [], []
    for _, r in df.iterrows():
        qs.append(transcripts.get(r["uid"], ""))
        cut = "cut off" if r.get("finish") == "length" else "finished"
        heads.append(f"Type: {r['question_type']}. Answer: {_text(r.get('pred')) or 'none'}. "
                     f"Solution {cut}.")
        tails.append(_text(r.get("text"))[-tail_chars:])
    return qs, heads, tails


def encode_rows(tok, qs: list[str], heads: list[str], tails: list[str], ncfg: NeuralConfig
                ) -> list[list[int]]:
    """Token ids: [CLS] question[:q] [SEP] header + solution tail (last tokens) [SEP]."""
    cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    q_ids = tok(qs, add_special_tokens=False)["input_ids"]
    h_ids = tok(heads, add_special_tokens=False)["input_ids"]
    t_ids = tok(tails, add_special_tokens=False)["input_ids"]
    out = []
    for q, h, t in zip(q_ids, h_ids, t_ids):
        q = q[:ncfg.question_tokens]
        h = h[:64]
        budget = max(0, ncfg.max_len - 3 - len(q) - len(h))
        t = t[-budget:] if budget else []
        out.append([cls_id, *q, sep_id, *h, *t, sep_id])
    return out


# ---------------------------------------------------------------------------
# Training / scoring
# ---------------------------------------------------------------------------
def _batches(ids: list[list[int]], order: list[int], bs: int, pad_id: int):
    import torch

    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        L = max(len(ids[j]) for j in idx)
        x = torch.full((len(idx), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(idx), L), dtype=torch.long)
        for k, j in enumerate(idx):
            x[k, :len(ids[j])] = torch.tensor(ids[j])
            m[k, :len(ids[j])] = 1
        yield idx, x, m


def _load_model(ncfg: NeuralConfig, path: str | None = None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src = path or ncfg.model
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(dev), dev


def train_model(ids: list[list[int]], y: np.ndarray, ncfg: NeuralConfig, tok_model=None):
    """Fine-tune on (ids, y); returns (tok, model, device)."""
    import torch

    random.seed(ncfg.seed)
    np.random.seed(ncfg.seed)
    torch.manual_seed(ncfg.seed)
    tok, model, dev = tok_model or _load_model(ncfg)
    if dev == "cuda":
        model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=ncfg.lr, weight_decay=ncfg.weight_decay)
    steps = math.ceil(len(ids) / (ncfg.batch_size * ncfg.grad_accum)) * ncfg.epochs
    warm = max(1, int(ncfg.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    yt = torch.tensor(y, dtype=torch.float32)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    rng = np.random.default_rng(ncfg.seed)
    step = 0
    for ep in range(ncfg.epochs):
        # Length-bucketed shuffling: similar lengths per batch (less padding), random order.
        order = np.argsort([len(x) + rng.uniform(0, 64) for x in ids]).tolist()
        chunks = [order[i:i + ncfg.batch_size] for i in range(0, len(order), ncfg.batch_size)]
        rng.shuffle(chunks)
        order = [j for c in chunks for j in c]
        total, n = 0.0, 0
        for b, (idx, x, m) in enumerate(_batches(ids, order, ncfg.batch_size, pad)):
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits.squeeze(-1)
            loss = loss_fn(logits.float(), yt[idx].to(dev)) / ncfg.grad_accum
            loss.backward()
            total += loss.detach().item() * ncfg.grad_accum
            n += 1
            if (b + 1) % ncfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if n % 200 == 0:
                log.info("epoch %d batch %d/%d loss %.4f", ep, n, len(chunks), total / n)
        log.info("epoch %d done: mean loss %.4f", ep, total / max(1, n))
    return tok, model, dev


def predict_ids(tok, model, dev, ids: list[list[int]], ncfg: NeuralConfig) -> np.ndarray:
    import torch

    model.eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    order = np.argsort([len(x) for x in ids]).tolist()
    out = np.zeros(len(ids), dtype=np.float32)
    with torch.no_grad():
        for idx, x, m in _batches(ids, order, ncfg.eval_batch_size, pad):
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits.squeeze(-1)
            out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def _encode_frame(df: pd.DataFrame, cfg: dict, tok, ncfg: NeuralConfig) -> list[list[int]]:
    from mmjee_reasoner.rag.tagger import load_english_transcripts

    qs, heads, tails = row_texts(df, load_english_transcripts(cfg), ncfg.tail_chars)
    return encode_rows(tok, qs, heads, tails, ncfg)


def train_neural_loyo(cfg: dict, tag: str = "base", with_extra: bool = False,
                      folds: list[int] | None = None, final: bool = False) -> dict:
    """LOYO out-of-year scores (resumable per fold) and, with ``final``, a model on all years."""
    import torch
    from transformers import AutoTokenizer

    ncfg = NeuralConfig.from_cfg(cfg)
    df = training_frame(cfg, with_extra)
    out = neural_dir(cfg, tag)
    (out / "config.json").write_text(json.dumps(ncfg.__dict__, indent=2))
    tok = AutoTokenizer.from_pretrained(ncfg.model)
    ids = _encode_frame(df, cfg, tok, ncfg)
    lens = np.array([len(x) for x in ids])
    log.info("neural verifier [%s]: %d rows, %d questions, tokens mean %.0f / max %d", tag,
             len(df), df["uid"].nunique(), lens.mean(), lens.max())
    y = df["correct"].astype(int).to_numpy()
    years = df["year"].to_numpy()
    for yr in folds if folds is not None else sorted(set(years)):
        path = out / f"oof_{int(yr)}.parquet"
        if path.exists():
            log.info("fold %d already done (%s)", yr, path)
            continue
        tr, te = np.flatnonzero(years != yr), np.flatnonzero(years == yr)
        tok_, model, dev = train_model([ids[i] for i in tr], y[tr], ncfg)
        s = predict_ids(tok_, model, dev, [ids[i] for i in te], ncfg)
        df.iloc[te][KEY_COLS].assign(score=s, year=int(yr)).to_parquet(path)
        log.info("fold %d: %d train / %d test rows written to %s", yr, len(tr), len(te), path)
        del model
        torch.cuda.empty_cache()
    if final:
        mdir = out / "model_full"
        if not (mdir / "config.json").exists():
            tok_, model, dev = train_model(ids, y, ncfg)
            model.save_pretrained(mdir)
            tok_.save_pretrained(mdir)
            log.info("final model saved to %s", mdir)
    return {"rows": len(df), "dir": str(out)}


def load_oof(cfg: dict, tag: str, with_extra: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Training frame + its out-of-year neural scores (aligned)."""
    df = training_frame(cfg, with_extra)
    out = neural_dir(cfg, tag)
    parts = [pd.read_parquet(p) for p in sorted(out.glob("oof_*.parquet"))]
    if not parts:
        raise FileNotFoundError(f"no out-of-year scores in {out}")
    oof = pd.concat(parts, ignore_index=True)
    m = df[KEY_COLS].merge(oof[[*KEY_COLS, "score"]], on=KEY_COLS, how="left")
    if m["score"].isna().any():
        raise RuntimeError(f"{int(m['score'].isna().sum())} rows lack out-of-year scores "
                           f"(folds missing in {out})")
    return df, m["score"].to_numpy()


def score_pool(cfg: dict, tag: str, df: pd.DataFrame) -> np.ndarray:
    """Scores of the final (all-years) model for an arbitrary candidate pool."""
    ncfg = NeuralConfig.from_cfg(cfg)
    mdir = neural_dir(cfg, tag) / "model_full"
    tok, model, dev = _load_model(ncfg, str(mdir))
    return predict_ids(tok, model, dev, _encode_frame(df, cfg, tok, ncfg), ncfg)
