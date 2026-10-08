"""All prompts used by the framework.

Baseline prompts are copied verbatim from the base paper (mmJEE-Eval,
Appendix C.1 language prompts and C.2 question-type instructions). The image is
placed before the text in a single user turn, as in the released code.
Module prompts (PoT solver, critic, corrector, concept tagger, transcriber,
RAG exemplars) are our own and are documented in the report.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Base paper prompts (verbatim, Appendix C)
# ---------------------------------------------------------------------------
PAPER_SYSTEM = {
    "English": "Please analyze the image carefully, reason step-by-step and provide your "
               "answer at the end.",
    "Hindi": "This is a question image in Hindi. Please analyze the image carefully, understand "
             "the question despite the language barrier, reason step-by-step and provide your "
             "answer at the end.",
}

PAPER_TYPE_INSTRUCTIONS = {
    "MCQ-Single": "For this question:\n- Choose exactly ONE option (A, B, C, or D)\n"
                  "- Format your answer in \\boxed{} as just one letter (e.g., \\boxed{A})",
    "MCQ-Multiple": "For this question:\n- Choose ONE OR MORE options (A, B, C, and/or D)\n"
                    "- Format your answer in \\boxed{} with letters (e.g., \\boxed{ABC} or "
                    "\\boxed{B})",
    "Numerical": "For this question:\n- Provide a numerical value\n"
                 "- Round to appropriate decimal places if needed\n"
                 "- Format your answer in \\boxed{} (e.g., \\boxed{2.5} or \\boxed{42})",
    "Matching": "For this question:\n- Choose ONE option that correctly matches items from left "
                "column to right column\n- Format your answer in \\boxed as just one letter "
                "(e.g., \\boxed{A})",
}


def baseline_prompt(language: str, question_type: str) -> str:
    """Single-pass CoT prompt of the base paper."""
    return f"{PAPER_SYSTEM[language]}\n\n{PAPER_TYPE_INSTRUCTIONS[question_type]}"


# ---------------------------------------------------------------------------
# Module A: program-of-thoughts solver with a code sandbox
# ---------------------------------------------------------------------------
STEP_FORMAT = (
    "Write the solution as numbered steps, each starting on a new line with "
    "\"Step 1:\", \"Step 2:\", and so on."
)

POT_INSTRUCTIONS = (
    "How to solve:\n"
    f"- {STEP_FORMAT}\n"
    "- Explain the physics/chemistry/mathematics of each step in words.\n"
    "- Do NOT do calculations in your head. Whenever a step needs a calculation (arithmetic, "
    "algebra, solving equations, derivatives, integrals, limits, unit conversions, or checking "
    "each option), write a short Python program in a ```python code block that prints the "
    "result. You may use math, fractions, sympy and numpy.\n"
    "- End the code block with ```. The code will be run and its printed output will "
    "be given back to you in an ```output block. Then continue the solution using that output. "
    "Never invent the output yourself.\n"
    "- Use at most {max_blocks} code blocks.\n"
    "- Be concise: do not re-check the same step again and again.\n"
    "- Finish with exactly one final answer in \\boxed{{}}. Do not use \\boxed anywhere else. "
    "For a numerical answer write a plain decimal number inside the box (no units, no "
    "fractions, no LaTeX), rounded to two decimal places if it is not an integer.\n\n"
    "Required format (example):\n"
    "Step 1: <which law / concept applies and the equation to use>\n"
    "```python\nimport sympy as sp\nx = sp.symbols('x')\nprint(sp.solve(2*x - 6, x))\n```\n"
    "```output\n[3]\n```\n"
    "Step 2: <interpret the printed result, check the options>\n"
    "\\boxed{{<answer>}}"
)


def pot_prompt(language: str, question_type: str, max_blocks: int,
               exemplars: str | None = None) -> str:
    """PoT solver prompt = paper prompt + tool-use instructions (+ RAG exemplars)."""
    parts = [baseline_prompt(language, question_type)]
    if exemplars:
        parts.append(exemplars)
    parts.append(POT_INSTRUCTIONS.format(max_blocks=max_blocks))
    return "\n\n".join(parts)


CODE_OUTPUT_TEMPLATE = "```output\n{output}\n```\n"

# Budget forcing cue appended when a PoT trajectory used up its token budget.
FORCE_ANSWER_CUE = ("\n\nI have run out of time, so I must commit to my best answer now.\n"
                    "**Final answer:** \\boxed{")

# ---------------------------------------------------------------------------
# Module B: critic (step location) and corrector
# ---------------------------------------------------------------------------
CRITIC_PROMPT = (
    "You are a strict examiner checking a student's solution to a JEE Advanced question. "
    "The question is in the attached image{lang_note}.\n\n"
    "Instructions the student was given:\n{type_instruction}\n\n"
    "Student's solution (steps are numbered):\n"
    "<solution>\n{solution}\n</solution>\n\n"
    "Check the steps in order: reading of the question and figure, chosen concept and "
    "equations, algebra and arithmetic (the ```output blocks are real program outputs), and "
    "the final option or number. Find the FIRST step that contains an error. Do not solve the "
    "whole problem again. Keep your check short (at most about 250 words), then end your reply "
    "with exactly these three lines:\n"
    "VERDICT: CORRECT or ERROR\n"
    "FIRST_ERROR_STEP: <step number, or NONE>\n"
    "EXPLANATION: <one or two sentences: what is wrong and how to fix it>"
)

# Appended when a critic reply was cut off (or ignored the format) to force the verdict lines.
CRITIC_VERDICT_CUE = "\n\nI must stop checking now and give my decision.\nVERDICT:"

CORRECTOR_FEEDBACK = (
    "A reviewer checked an earlier attempt at this question. The steps before Step {k} were "
    "accepted and are already written at the start of your answer. The reviewer found an "
    "error in Step {k}:\n\"{explanation}\"\n"
    "Continue the solution from Step {k}: re-derive that step correctly (do not repeat the "
    "error) and complete the solution, following the same rules (Python code for every "
    "calculation, one final \\boxed{{}} answer)."
)


def critic_prompt(language: str, question_type: str, solution_steps: str) -> str:
    lang_note = " (the question is written in Hindi)" if language == "Hindi" else ""
    return CRITIC_PROMPT.format(
        lang_note=lang_note,
        type_instruction=PAPER_TYPE_INSTRUCTIONS[question_type],
        solution=solution_steps,
    )


def corrector_prompt(language: str, question_type: str, max_blocks: int, k: int,
                     explanation: str, exemplars: str | None = None) -> str:
    return pot_prompt(language, question_type, max_blocks, exemplars) + "\n\n" + \
        CORRECTOR_FEEDBACK.format(k=k, explanation=explanation.strip() or "unspecified error")


# ---------------------------------------------------------------------------
# Module D: concept tagging, transcription, exemplars
# ---------------------------------------------------------------------------
TAGGER_PROMPT = (
    "Look at this JEE Advanced exam question (it may be written in English or Hindi). Do not "
    "solve it. In English, describe what is needed to solve it, using exactly this format on "
    "one line:\n"
    "SUBJECT: <Physics/Chemistry/Mathematics> | TOPIC: <specific topic, 2-6 words> | METHOD: "
    "<key concept or solution technique, at most 15 words>"
)

TRANSCRIBE_PROMPT = (
    "Transcribe this exam question exactly as plain text in its original language. Write "
    "mathematical expressions in LaTeX. Include all answer options. If there is a figure, add "
    "one line \"[Figure: <short description>]\". Output only the transcription."
)

EXEMPLAR_HEADER = (
    "Below are worked solutions of earlier JEE Advanced problems that use a similar method. "
    "Use them only as guidance on the method. They are DIFFERENT problems: do not copy their "
    "numbers or answers."
)

EXEMPLAR_TEMPLATE = "### Example {i}\nProblem:\n{problem}\n\nWorked solution:\n{solution}"


def format_exemplars(examples: list[dict]) -> str | None:
    """``examples``: dicts with ``problem`` and ``solution`` text."""
    if not examples:
        return None
    blocks = [EXEMPLAR_TEMPLATE.format(i=i + 1, problem=e["problem"].strip(),
                                       solution=e["solution"].strip())
              for i, e in enumerate(examples)]
    return EXEMPLAR_HEADER + "\n\n" + "\n\n".join(blocks)
