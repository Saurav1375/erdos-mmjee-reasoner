"""Splitting solutions into numbered steps and parsing critic verdicts."""

from __future__ import annotations

import re
from dataclasses import dataclass

_STEP_RE = re.compile(r"^[ \t>#*_-]*(?:\*\*)?step[ \t]*(\d+)\b", re.IGNORECASE)


def _step_marker_offsets(text: str) -> list[int]:
    """Character offsets of lines starting with "Step k", ignoring fenced code."""
    offsets, pos, in_fence = [], 0, False
    for line in text.splitlines(keepends=True):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and _STEP_RE.match(line):
            offsets.append(pos)
        pos += len(line)
    return offsets


def split_steps(text: str) -> list[str]:
    """Split a solution into steps.

    Uses "Step k:" markers outside code blocks when present (text before the
    first marker is merged into step 1). Otherwise falls back to blank-line
    separated paragraphs (code + output blocks stay attached to their paragraph).
    """
    text = text.strip()
    if not text:
        return []
    starts = _step_marker_offsets(text)
    if starts:
        bounds = starts + [len(text)]
        steps = [text[bounds[i]:bounds[i + 1]].strip() for i in range(len(starts))]
        head = text[:starts[0]].strip()
        if head:
            steps[0] = head + "\n" + steps[0]
        return [s for s in steps if s]
    # Paragraph fallback: never split inside a fenced block.
    paras, buf, in_fence = [], [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if buf:
                paras.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        paras.append("\n".join(buf).strip())
    return [p for p in paras if p]


def number_steps(steps: list[str]) -> str:
    """Render steps with explicit [Step k] labels for the critic."""
    return "\n\n".join(f"[Step {i + 1}]\n{s}" for i, s in enumerate(steps))


@dataclass
class CriticVerdict:
    has_error: bool
    step: int | None       # 1-based first wrong step (None if unknown / no error)
    explanation: str
    parsed: bool           # False if the reply did not follow the format


_VERDICT_RE = re.compile(r"VERDICT\s*:\s*\**\s*(CORRECT|ERROR|INCORRECT|WRONG)", re.IGNORECASE)
_STEP_NUM_RE = re.compile(r"FIRST[_ ]ERROR[_ ]STEP\s*:\s*\**\s*(?:\[?step\s*)?(\d+|NONE)", re.IGNORECASE)
_EXPL_RE = re.compile(r"EXPLANATION\s*\**\s*:\s*\**\s*(.+)", re.IGNORECASE)


def parse_critic(reply: str, n_steps: int) -> CriticVerdict:
    """Parse the critic's final VERDICT / FIRST_ERROR_STEP / EXPLANATION lines.

    The last occurrence of each field wins (the critic may think first).
    Unparseable replies count as "no error found" (conservative: no change).
    """
    verdicts = _VERDICT_RE.findall(reply)
    steps = _STEP_NUM_RE.findall(reply)
    expl_m = list(_EXPL_RE.finditer(reply))
    explanation = expl_m[-1].group(1).strip(" *") if expl_m else ""
    step: int | None = None
    if steps and steps[-1].upper() != "NONE":
        step = int(steps[-1])
        if not 1 <= step <= max(n_steps, 1):
            step = None
    if verdicts:
        has_error = verdicts[-1].upper() != "CORRECT"
        if has_error and step is None:
            step = 1  # error claimed but no valid step: re-derive from the start
        return CriticVerdict(has_error, step if has_error else None, explanation, True)
    if step is not None:  # no verdict line but a concrete step was named
        return CriticVerdict(True, step, explanation, False)
    return CriticVerdict(False, None, explanation, False)
