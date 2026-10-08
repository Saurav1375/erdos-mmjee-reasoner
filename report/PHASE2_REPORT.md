# Phase-2 Progress Report

**Improving Scientific Reasoning in Vision-Language Models using Code Sandboxes, Agentic Correction, Learned Verification, and RAG on mmJEE-Eval**

Arpit Kumar (12340350) · Saurav Gupta (12341940). Phase-2 submission, 9 October 2026.

---

## 1. Summary

| Item | Status |
|---|---|
| Full framework implemented (4 modules + pipeline + evaluation + report generator) | ✅ |
| Base-paper scoring reused **byte-identically** (verified by test) | ✅ |
| Reproduction of a base-paper number with our harness | ✅ Qwen2.5-VL-7B: **13.8 ± 1.9%** vs paper 12.4 ± 1.9% |
| **Baseline on the full 2025 held-out set** (190 q, 16 samples) | ✅ Qwen3-VL-8B: **43.4%** Pass@1 (95% CI 37.1–49.9) |
| End-to-end pilot of all modules (40 test + 150 train questions) | ✅ Full framework **57.5%** vs 29.5% baseline on the pilot subset |
| Test suite | ✅ 73 automated tests (CPU, fake VLM) |
| Engineering issues found and fixed | 23 (Section 7) |
| Full run (all modules on all 190 questions; verifier on ~10k solutions) | ⏳ next phase (~45 GPU-h) |

The pilot results are **preliminary**. They cover 40 questions, so the 95% CIs are about ±15 points. Final claims will come from the full run.

---

## 2. Problem and approach (recap of the SOP)

mmJEE-Eval (Mukherjee & Ghosh, IJCNLP-AACL 2025) has 1,460 bilingual (English/Hindi) JEE Advanced questions from 2019–2025, each given as an image. The paper diagnoses two weaknesses of VLMs: weak numerical computation, and a large gap between *detecting* an error and *fixing* it. Its single-pass correction fixes only 1.1–5.2% of errors. The paper releases an evaluation harness but no solver.

We wrap a **frozen** VLM in four inference-time modules. Only the verifier is trained.

| Module | Idea | Implementation |
|---|---|---|
| A. Code sandbox (PoT) | Calculations are written as Python/SymPy and executed | Generation stops at each closed code block. The code runs in a sandbox and its output is fed back. Plain-CoT fallback if no answer is produced or the code never runs. |
| B. Agentic correction | Critic finds the **first wrong step**; Corrector re-derives from it | ≤3 rounds. Stops on "correct", on an unchanged answer, or at max rounds. Same-model vs cross-model critic. |
| C. Learned verifier (core ML) | A classifier picks the final answer from N candidates | LogReg vs XGBoost on interpretable features + sentence embeddings. Trained only on auto-labelled 2019–2024 solutions. |
| D. RAG exemplars | Retrieve solved problems that use the same method | VLM concept tag → FAISS search over a knowledge base of verified 2019–2024 solutions, with a similarity threshold and de-duplication against 2025. |

**Split:** train/dev = 2019–2024 (1,270 questions); test = 2025 (190 questions = 95 English + 95 Hindi twins). The 102 questions added to the HF release for 2026 are excluded.

---

## 3. System architecture and workflow

```
                   ┌──────────────── 2019–2024 (train) ────────────────┐
question image ──► concept tag ──► FAISS KB (verified worked solutions) │
     │                    │ top-2 exemplars (threshold)                   │
     ▼                    ▼                                               │
 Solver (Qwen3-VL-8B) + code sandbox ── N=8 samples ──► Critic ──► Corrector (≤3 rounds)
                                                          │
                       pool: 8 originals + 8 corrected ◄──┘
                                    │
                    learned verifier (trained on 2019–2024) ──► final answer
                                    │
                    upstream scorer (verbatim) on 2025 ──► Pass@1, CIs, breakdowns
```

**Software layout** (`mmjee-reasoner/`): `data/` (load, split, scorer adapter), `scoring/` (verbatim upstream functions + metrics), `llm/` (vLLM engine, SQLite generation cache, mock VLM), `sandbox/` (executor + PoT loop), `agents/` (step splitting, critic parsing, correction loop), `verifier/`, `rag/`, `pipeline/` (stages), `report/`, and `cli.py` (`mmjee <stage>`).

**Engineering choices**

- **Everything is cached and resumable.** Every model call and every code execution is cached, keyed by model, prompt, seed and settings. A crash loses at most one batch, re-runs are free, and evaluation and reports run offline.
- **Deterministic seeds** per (question, sample, turn). Increasing N reuses the earlier samples.
- **Subsets.** `--limit N` gives a smoke subset and `--pilot` gives a stratified English/Hindi-paired pilot. Each writes separately suffixed outputs, so they never mix with the full run.
- **Lab orchestration.** `scripts/lab/launch.sh` starts `supervise.sh` detached. It runs the stages in order and retries each up to 3 times (see 7.3).
- **Hardware.** Lab PC with one RTX 3060 (12 GB) and 15 GB RAM. Solver: Qwen3-VL-8B-Instruct (AWQ 4-bit). Cross-model critic: InternVL3.5-8B (AWQ 4-bit). Both are served one at a time with vLLM 0.30, FP8 KV cache, about 400 tokens/s.

---

## 4. Evaluation protocol and correctness safeguards

- **Scoring reuse.** `extract_answer` (self-improvement notebook) and `is_answer_correct` + `calculate_statistics` (accuracy notebook) are copied **byte-for-byte**. `tests/test_scoring.py` checks the copies against the notebooks with `ast`.
- **Scorer adapter (data side only).** The public dataset lacks the `expanded_answer` / `acceptable_values` columns the scorer needs for range answers such as `"8.70 TO 9.10"`. Without them, every Numerical question would score 0. We rebuild the columns (0.001 grid, OR-alternatives unioned). A test checks that all 691 Numerical gold answers are scored correct.
- **Data fixes** (logged in `prepare_report.json`):
  - 12 rows from 2023 Paper 1 were typed MCQ-Single but have multi-letter gold answers. They are retyped MCQ-Multiple.
  - 4 duplicated question IDs (different questions) are given unique IDs.
- **Prompts.** The baseline uses the paper's prompts verbatim (Appendix C.1/C.2), with the image placed before the text.
- **Pass@1** = mean accuracy over N samples, with one "run" per sample index (the paper averages 10 runs).
- **Confidence intervals.** We use a cluster bootstrap over English/Hindi pairs (95 clusters), because twins are correlated. Paired bootstrap tests are used for differences.
- **Fair comparison.** Self-consistency (majority vote over CoT samples) is evaluated at a **matched** generated-token budget.
- **Leakage guards** (asserted in code and tests):
  - No 2025 twin appears in training, the verifier data, or the KB.
  - KB entries that are near-duplicates of any 2025 transcription (cosine ≥ 0.9) are dropped.
  - The verifier and the RAG threshold are selected on the 2024 dev year only.

---

## 5. Results

### 5.1 Reproduction check (does our harness match the paper?)

| Model | Paper (Table 3) | **Our harness** (10 runs × 190) |
|---|---|---|
| Qwen2.5-VL-7B-Instruct | 12.4 ± 1.9% | **13.8 ± 1.9%** (95% CI 10.7–17.3) |

The paper's value lies inside our CI, so our prompts, image handling and scorer reproduce the paper's protocol. The small residual gap is plausibly due to the AWQ weights and sampling settings the paper does not report. By type: MCQ-Single 28.7, Matching 27.2, MCQ-Multiple 6.2, Numerical 6.5.

### 5.2 Baseline on the full 2025 held-out set (Table I, row 1)

Substrate: Qwen3-VL-8B-Instruct (AWQ). Paper prompt, single-pass CoT, 16 samples per question, max 4096 tokens.

| Metric | Value |
|---|---|
| **Pass@1** | **43.4%** (cluster-bootstrap 95% CI 37.1–49.9; run std 1.9) |
| By type: Numerical / MCQ-Single / MCQ-Multiple / Matching | 40.0 / 49.6 / 46.9 / 35.8 |
| By language: English / Hindi | 48.5 / 38.4 |
| By subject: Mathematics / Chemistry / Physics | 55.1 / 41.7 / 33.2 |
| Diagram required: no / yes | 48.5 / 30.0 |
| Self-consistency (majority vote): n = 1 / 4 / 8 / 16 | 42.6 / 49.5 / 52.6 / 53.7 |
| Mean generated tokens per sample | 2,705 (about 35% hit the 4096 cap) |

**Context.** The paper reports 2025 Pass@1 of 12.4 for Qwen2.5-VL-7B, 30.6 for Gemma-3-27B, 46.2 for Qwen3-VL-235B, and 72–80 for frontier models. The jump from Qwen2.5-VL-7B (13.8 in our harness) to Qwen3-VL-8B (43.4) is a **model-generation effect**, not a harness artefact (5.1). The same weaknesses as in the paper persist: Hindi < English by 10 points, diagram questions < text questions by 18 points, and Physics is weakest. *Caveat:* Qwen3-VL was released after the 2025 exam, so contamination cannot be excluded. We will report the paper's contamination check (2019–24 vs 2025 accuracy) in the final report.

### 5.3 Pilot: all modules end-to-end (40 test questions, preliminary)

Pilot subset = 20 English/Hindi twin pairs stratified by type, plus 150 training questions for the pilot verifier and KB. Run with the v1 pipeline. The fixes in Section 6 came out of this run.

| Configuration (additive, SOP Table I) | Overall | Numerical | Δ vs baseline | Gen. tokens / q |
|---|---|---|---|---|
| Baseline (single-pass CoT) | 29.5 | 19.4 | — | 2.7k |
| Self-consistency (16 samples ≈ matched compute) | 35.0 | 33.3 | +5.5 | 43.1k |
| + Code sandbox (PoT) | 41.2 | 38.9 | +11.7 | 3.5k |
| + Agentic correction, same-model critic | 46.6 | 43.1 | +17.0 | 7.7k |
| + Agentic correction, cross-model critic | 43.1 | 43.1 | +13.6 | 5.5k |
| + Learned verifier (pool of 16) | 50.0 | 50.0 | +20.5 | 44.0k |
| **+ RAG = Full framework** | **57.5** | **55.6** | **+28.0** | 46.6k |

- **Full framework vs compute-matched self-consistency: +22.5 points** (paired cluster bootstrap 95% CI [+7.5, +37.5]).
- The pilot questions are harder than average: the baseline is 29.5 here vs 43.4 on all 190.

**Correction loop statistics** (320 chains each; compare with the paper's single-pass fix rate of 1.1–5.2%):

| Run | Acc. before → after | Detected errors | False alarms | Fix rate of detected | wrong→right / right→wrong |
|---|---|---|---|---|---|
| Same-model critic (Qwen3-VL) | 41.2 → 46.6 | 79.3% | 53.0% | 16.8% | 25 / 8 |
| Cross-model critic (InternVL3.5) | 41.2 → 43.1 | 35.1% | 23.5% | 19.7% | 13 / 7 |
| Full (RAG source, cross critic) | 42.8 → 46.2 | 40.4% | 21.2% | 20.3% | 15 / 4 |

**Learned verifier** (pilot: 1,200 auto-labelled solutions, 47.6% correct):

| Model / features | CV AUC (2019–23) | Dev AUC (2024) | Dev F1 | Dev selection (weighted vote) |
|---|---|---|---|---|
| LogReg / handcrafted | 0.858 | 0.867 | 0.736 | 54.2 |
| LogReg / embedding | 0.704 | 0.608 | 0.619 | 54.2 |
| LogReg / all | 0.863 | **0.876** | 0.743 | 54.2 |
| XGBoost / handcrafted | 0.855 | 0.874 | 0.754 | 54.2 |
| XGBoost / embedding | 0.622 | 0.591 | 0.588 | 50.0 |
| XGBoost / all | 0.847 | 0.870 | **0.785** | 54.2 |
| *Majority vote / random pick / oracle* | | | | *54.2 / 45.3 / 79.2* |

- The verifier is a good correctness **classifier**: AUC about 0.87, and interpretable features carry most of the signal. With pilot-size data it does **not yet beat majority voting at selection**.
- On the 2025 pilot pool its AUC is 0.78–0.81. Its selection is 57.5% weighted vote vs 55.0% majority, and the oracle is 75–80%. So there is a lot of headroom for selection.
- The full run trains on about 8× more solutions.

**RAG (pilot).** The pilot KB has only 56 entries, because it is built from 75 training pairs. Only 4 of 40 questions retrieved an exemplar, so the pilot cannot attribute the gain from 50.0 to 57.5 to retrieval. The full KB has about 600 entries.

---

## 6. Issues found in the pilot and fixes (v2)

Diagnosis is from the logged traces (1,520 PoT trajectories, 320 correction chains).

| # | Observation | Root cause (measured) | Fix |
|---|---|---|---|
| 1 | 47 `NameError`s in executed code | Each code block ran in a fresh process; the model reuses variables from earlier blocks | Earlier blocks of the same solution are now silently replayed before the new block (only blocks that pass the safety check) |
| 2 | 86 code blocks blocked | `scipy.optimize` / `scipy.integrate` were not whitelisted | `scipy` allowed; file-I/O and parser submodules (`scipy.io`, `sympy.parsing`, …) still blocked |
| 3 | 76 critic replies unparsed (InternVL) | 75 of 76 hit the 1,536-token cap while re-solving the problem | "Keep your check short" in the prompt, plus **verdict forcing**: a cut-off reply gets a short greedy continuation after a decision cue |
| 4 | CoT fallback in 35–42% of PoT runs | 480 of 539 fallbacks: the model reasons without code until the 4,096-token cap, looping on "Wait, perhaps…"; only 9 are code failures | **A/B tested "answer forcing"** (commit to an answer from the truncated reasoning): 21.9% vs **30.0%** for the SOP CoT fallback, so forcing was rejected and the SOP fallback kept. *Negative result, reported.* |
| 5 | Matched-compute self-consistency needed 18 samples, but only 16 existed | Budget estimate | CoT baseline raised to **20 samples** |
| 6 | Verifier ≈ majority vote | Small pilot training set; pool composition | Kept honest: `primary_beats_majority_on_dev` is reported; the full run uses about 10k solutions |

**v2 verification run** (PoT and cross-model correction re-run on the pilot questions with the fixes). Results are listed below once that run finishes.

| Check (pilot, 40 q) | v1 | **v2** |
|---|---|---|
| Executed code blocks OK / blocked / NameError | 119 / 13 / 3 | **138 / 0 / 0** |
| PoT accuracy (320 trajectories) | 41.2 | 41.2 |
| Critic replies parsed (cross-model) | 244 / 320 (76%) | **500 / 501 (99.8%)**, 69 via verdict forcing |
| Cross critic: detection of wrong solutions | 35.1% | **56.4%** |
| Cross critic: false alarms on correct solutions | 23.5% | 29.5% |
| Cross critic: accuracy before → after (320 chains) | 41.2 → 43.1 | 41.2 → 41.9 |

The fixes remove the engineering failures: no blocked or crashed code from lost state, and the critic now always returns a verdict. The critic now *detects* many more errors, but its net effect on accuracy is unchanged within noise. Turning detections into fixes is the real bottleneck, as in the paper. The full run measures this on 1,520 chains.

During verification, one more failure surfaced (item 22): a long multi-round solution produced a 12,303-token critic prompt, over the 12,288-token context. vLLM's `VLLMValidationError` is not a `ValueError`, so the whole batch aborted. Such requests are now isolated per request and return an empty result, handled conservatively. A test covers this.

---

## 7. Engineering log: errors found and fixed

### 7.1 Pre-run code review (3 parallel reviewers, before any GPU run)
1. Verifier cross-validation crashed: `GroupKFold` returns positions, which pandas treated as column labels. Now uses positional indexing.
2. Verifier features at test time differed from training (pool of 16 vs 8, cumulative token counts). Features are now computed per sub-pool, and the solution's own token count is stored.
3. The sandbox read unbounded stdout into the parent process; a test reproduced 8 GB of RAM use. Output now goes to files capped by `RLIMIT_FSIZE`.
4. If the model ended its turn right after a code fence, the code never ran. This is now detected.
5. There was no context-length guard. A per-turn budget was added, plus per-request retry when a batch fails.
6. The critic explanation regex took the first match instead of the last. Fixed.
7. `--config` was not passed to the subprocesses of the cross-model phases. Fixed.
8. Confidence intervals ignored English/Hindi twin correlation. Now a cluster bootstrap.
9. The range grid was too coarse (0.01) and rejected valid in-range answers such as 0.275. Now 0.001.
10. Duplicate question IDs could mix two questions in the KB. Duplicates now get unique twin keys.
11. The mock VLM was deterministic across images, which hid bug 1. Fixed, and the end-to-end mock test now passes.

### 7.2 Smoke / lab runs
12. The vLLM KV cache did not fit on 12 GB (0.36 GiB free). Capped `max_pixels`, set `max_num_batched_tokens` to 4096, and enabled the FP8 KV cache, giving 1.9× throughput (measured).
13. flashinfer's JIT compile failed (missing `ninja`). Switched to vLLM's PyTorch sampler.
14. rsync excluded `src/.../report`. Exclude patterns are now anchored.
15. CUDA ran out of memory during transcription at 0.92 memory utilisation. Now 0.88 with 32 concurrent sequences.
16. vLLM's InternVL processor rejects `max_dynamic_patch`. Default tiling is used.
17. The **FP8 KV cache produces garbage with Qwen2.5-VL** (1.8% accuracy, junk tokens). Normal KV cache for that model, and its cache entries were invalidated by renaming the model.
18. The PoT prompt produced too little code, so a format example and a conciseness rule were added.

### 7.3 Infrastructure
19. systemd-oomd killed the whole tmux session under RAM pressure, wasting 10.5 h. Each stage now runs in its own systemd scope with 3 retries, and sandbox workers dropped from 8 to 4. No kills since.
20. The answer-forcing A/B (Section 6, item 4).
21. Reviewer finding: matched-compute self-consistency had too few samples. Fixed (Section 6, item 5).
22. A prompt over the context limit aborted a whole vLLM batch (`VLLMValidationError`). Bad requests are now isolated one by one.
23. systemd-oomd also kills `systemd-run --user` scopes (it watches all of user@1000.service). Jobs now start with `launch.sh` (nohup + setsid from the ssh session), outside that pressure-kill policy. Retries remain.

---

## 8. Testing and verification

- **73 automated tests**, run with `pytest` on CPU in about 2 minutes:
  - scorer byte-identity with the notebooks, plus every gold answer scored correct;
  - extraction edge cases;
  - sandbox: timeout, memory cap, blocked imports and calls, output cap, prelude replay, scipy;
  - PoT loop: fence handling, fallback, max blocks;
  - critic parsing and verdict forcing; correction stopping rules;
  - verifier on synthetic data (AUC > 0.9) and CV-fold indexing; RAG filters and threshold;
  - pilot stratification;
  - leakage assertions;
  - a full end-to-end pipeline run with a fake VLM, checking that self-consistency over 1 sample equals the mean Pass@1, and that cache re-runs produce no model calls.
- **Manual checks on real outputs:**
  - hand-checked scored answers (exact letter sets; numeric values inside gold ranges);
  - the Qwen3-VL chat template keeps the code fence when continuing a message;
  - the InternVL and Qwen2.5-VL engines produce sane text.
- **Reproduction of a paper number** (5.1), which validates the whole harness.

---

## 9. Plan for the final phase

1. **Full run** on the lab, about 45 GPU-hours, under the supervisor with a watcher:
   - CoT 20 × 190; PoT 8 × 190; same-model and cross-model correction;
   - PoT on all 1,270 training questions (about 10k verifier solutions, KB of about 600 entries);
   - RAG-PoT; full framework;
   - verifier training and selection on 2024; evaluation; report.
2. Final SOP Table I on 190 questions, with breakdowns, matched-compute self-consistency and significance tests. Correction statistics compared with the paper's EP/EC numbers.
3. Contamination check (2019–24 vs 2025 accuracy of the substrate).
4. Final IEEE-style report (auto-generated LaTeX tables and figures, plus written discussion).

## 10. Limitations

- One substrate model and one exam year (190 questions; CIs about ±6 points).
- 4-bit weights and a 4,096-token budget; about 35% of baseline answers are truncated (the same budget for every configuration).
- The strict, unchanged base-paper scoring (no partial credit; first `\boxed{}`).
- Possible pre-training exposure of Qwen3-VL to the 2025 exam.
