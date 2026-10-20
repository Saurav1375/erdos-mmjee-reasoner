"""Phase 4: RAG method cards (mock)."""

from __future__ import annotations

from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.llm.types import GenResult


class CardEngine:
    name = "card"

    def __init__(self):
        self.calls = []

    def generate(self, reqs):
        self.calls += reqs
        return [GenResult("Method: energy conservation\nPlan: a -> b\nThe answer is (B), "
                          "value 42 so \\boxed{42}.", 10, 30, "stop") for _ in reqs]


def test_cards_stage_sanitizes_and_merges(cfg, dataset, monkeypatch):
    from mmjee_reasoner.rag import cards, kb

    qs = load_questions(cfg, "train")
    e = [{"kb_id": f"k{i}", "uid": u, "problem": f"P{i}", "solution": f"S{i}"}
         for i, u in enumerate(qs["uid"].head(3))]
    monkeypatch.setattr(kb, "load_kb", lambda cfg: (e, None, {}))
    monkeypatch.setattr(cards, "_answers", lambda cfg: {e[0]["uid"]: "42"})
    eng = CardEngine()
    assert cards.run_cards_stage(cfg, eng) == 3
    got = cards.load_cards(cfg)
    c0 = got[cards.card_key(e[0])]
    assert "42" not in c0 and "(B)" not in c0 and "\\boxed" not in c0
    assert "Method: energy conservation" in c0
    assert cards.run_cards_stage(cfg, eng) == 0 and len(eng.calls) == 3   # resumable merge


def test_card_exemplar_format():
    from mmjee_reasoner.prompts.templates import format_card_exemplars, format_exemplars

    assert format_card_exemplars([]) is None
    t = format_card_exemplars([{"problem": "P", "card": "Method: x"}])
    assert "Method card:\nMethod: x" in t and "Worked solution" not in t
    assert "Worked solution:\nS" in format_exemplars([{"problem": "P", "solution": "S"}])
