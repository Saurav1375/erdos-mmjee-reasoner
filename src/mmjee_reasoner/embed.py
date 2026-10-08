"""Sentence-embedding helper (shared by RAG and the verifier)."""

from __future__ import annotations

import functools
import logging

import numpy as np

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    log.info("loaded embedding model %s on %s", name, model.device)
    return model


def embed_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """L2-normalised float32 embeddings, shape (len(texts), dim)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vecs = _model(model_name).encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=len(texts) > 500, convert_to_numpy=True,
    )
    return vecs.astype(np.float32)
