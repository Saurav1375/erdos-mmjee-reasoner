# Prompts (`prompts/templates.py`)

All prompts live in one file, so the report can render them and every change is easy to see.

| Prompt | Used by | Origin |
|---|---|---|
| `PAPER_SYSTEM`, `PAPER_TYPE_INSTRUCTIONS`, `baseline_prompt` | Baseline CoT, CoT fallback, reproduction | **Verbatim from the base paper** (mmJEE-Eval Appendix C.1 language prompts, C.2 question-type instructions). The image comes before the text in one user turn, as in the released code. |
| `STEP_FORMAT`, `POT_INSTRUCTIONS`, `pot_prompt`, `CODE_OUTPUT_TEMPLATE` | Module A (PoT solver) | Ours |
| `FORCE_ANSWER_CUE` | Budget forcing (disabled after the pilot A/B) | Ours |
| `CRITIC_PROMPT`, `CRITIC_VERDICT_CUE`, `critic_prompt` | Module B critic | Ours |
| `CORRECTOR_FEEDBACK`, `corrector_prompt` | Module B corrector | Ours |
| `TAGGER_PROMPT`, `TRANSCRIBE_PROMPT` | Module D tagging and transcription | Ours |
| `EXEMPLAR_HEADER`, `EXEMPLAR_TEMPLATE`, `format_exemplars` | Module D exemplar injection | Ours |

## Format rules shared by all module prompts

- Numbered `Step k:` lines. The critic and corrector rely on them.
- A ```` ```python ```` block for every calculation.
- Exactly **one** final `\boxed{}` with a plain decimal number. The upstream extractor takes the first `\boxed{}`.
