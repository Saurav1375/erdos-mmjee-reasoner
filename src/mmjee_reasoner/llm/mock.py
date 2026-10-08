"""Deterministic fake VLM for dry runs and end-to-end tests (no GPU needed).

It recognises which prompt it received (CoT, PoT turn, critic, tagger,
transcriber) and answers in the expected format. Answers are pseudo-random
but seeded by the request, and are right with a fixed probability, so every
downstream component (scoring, voting, verifier, report) gets non-trivial data.
"""

from __future__ import annotations

import random
import re

from mmjee_reasoner.llm.types import GenRequest

_LETTERS = "ABCD"


def _text_of(req: GenRequest) -> str:
    content = req.messages[0]["content"]
    if isinstance(content, list):
        return " ".join(i.get("text", "") for i in content if i.get("type") == "text").strip()
    return content


def _answer(rng: random.Random, qtype: str, gold: str | None, p_correct: float) -> str:
    if gold is not None and rng.random() < p_correct:
        return gold
    if qtype == "MCQ-Multiple":
        return "".join(sorted(rng.sample(_LETTERS, rng.randint(1, 3))))
    if qtype == "Numerical":
        return str(rng.randint(0, 50))
    return rng.choice(_LETTERS)


class FakeVLM:
    """Callable responder for :class:`~mmjee_reasoner.llm.engine.MockEngine`.

    ``golds`` maps image path (relative) -> (question_type, gold answer string
    usable inside ``\\boxed{}``) so the fake model can sometimes be right.
    """

    def __init__(self, golds: dict[str, tuple[str, str]] | None = None, p_correct: float = 0.4):
        self.golds = golds or {}
        self.p_correct = p_correct

    def _question(self, req: GenRequest) -> tuple[str, str | None]:
        content = req.messages[0]["content"]
        image = next((i["image"] for i in content if i.get("type") == "image"), None) \
            if isinstance(content, list) else None
        qtype, gold = self.golds.get(image, ("Numerical", None))
        return qtype, gold

    def __call__(self, req: GenRequest) -> str:
        text = _text_of(req)
        qtype, gold = self._question(req)
        content = req.messages[0]["content"]
        image = next((i["image"] for i in content if i.get("type") == "image"), "") \
            if isinstance(content, list) else ""
        rng = random.Random(f"{req.sampling.get('seed')}|{image}|{text[-40:]}")
        if text.startswith("Look at this JEE Advanced exam question"):
            topic = rng.choice(["projectile motion", "chemical equilibrium", "definite integrals",
                                "electrostatics capacitors"])
            return f"SUBJECT: Physics | TOPIC: {topic} | METHOD: apply standard formula"
        if text.startswith("Transcribe this exam question"):
            return f"Question about {rng.choice(['a block', 'a gas', 'a matrix'])} " \
                   f"number {rng.randint(0, 10**6)}."
        if text.startswith("You are a strict examiner"):
            if rng.random() < 0.5:
                return "Looks fine.\nVERDICT: CORRECT\nFIRST_ERROR_STEP: NONE\nEXPLANATION: ok"
            return ("Step 2 uses the wrong sign.\nVERDICT: ERROR\nFIRST_ERROR_STEP: 2\n"
                    "EXPLANATION: The sign in step 2 is wrong.")
        ans = _answer(rng, qtype, gold, self.p_correct)
        if "```python" in text:  # PoT prompt
            if len(req.messages) == 1 or req.messages[-1]["content"].rstrip().endswith(":"):
                # first turn (or corrector prefill): write one code block
                return ("Step 1: Set up the relation.\n```python\nx = 6 * 7\nprint(x)\n```\n"
                        "Step 2: never reached because of the stop string")
            # continuation after an output block
            return f"\nStep 2: Using the printed value, the answer follows.\n\\boxed{{{ans}}}"
        return f"Step 1: Reason.\nStep 2: Conclude.\nThe answer is \\boxed{{{ans}}}."


def golds_for(questions) -> dict[str, tuple[str, str]]:
    """Image path -> (type, a gold value the upstream scorer accepts)."""
    out = {}
    for _, q in questions.iterrows():
        gold = str(q["answer"])
        if q["question_type"] == "Numerical":
            m = re.search(r"-?\d+(?:\.\d+)?", gold)
            gold = m.group(0) if m else "0"
        out[f"data/images/{q['uid']}.png"] = (q["question_type"], gold)
    return out
