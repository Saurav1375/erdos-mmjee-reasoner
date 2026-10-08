"""RAG: concept-tag parsing, topic matching and thresholded retrieval."""

from __future__ import annotations

import numpy as np

from mmjee_reasoner.rag.retrieve import search, topic_match
from mmjee_reasoner.rag.tagger import parse_tag


def test_parse_tag():
    t = parse_tag("SUBJECT: Physics | TOPIC: Projectile motion | METHOD: resolve components.")
    assert t["parsed"] and t["topic"] == "Projectile motion"
    assert t["query"].startswith("Projectile motion")
    assert not parse_tag("garbage")["parsed"]


def test_topic_match():
    assert topic_match("projectile motion", "Projectile Motion on incline", 0.5)
    assert not topic_match("chemical equilibrium", "projectile motion")


def test_search_filters_threshold_subject_and_twin():
    entries = [{"kb_id": "a", "subject": "Physics"}, {"kb_id": "b", "subject": "Chemistry"},
               {"kb_id": "c", "subject": "Physics"}]
    vecs = np.eye(3, dtype=np.float32)
    q = np.array([0.9, 0.9, 0.1], dtype=np.float32)
    assert [i for i, _ in search(q, vecs, entries, "Physics", None, 2, 0.5)] == [0]
    assert search(q, vecs, entries, "Physics", "a", 2, 0.5) == []
    assert [i for i, _ in search(q, vecs, entries, None, None, 3, 0.0)] == [0, 1, 2]
