# Phase 4 pre-registration (written 2026-10-19, before any Phase-4 method is scored on 2025/2026)

The v3 framework is frozen (`V3_FREEZE.md`) and its 2026 clean test result stays the project's headline. 2025 and 2026 have both been looked at, so **every Phase-4 number on them is post-hoc**. This file fixes in advance what will be scored, how each choice is made, and what counts as a positive result. Choices use 2019–2024 only (leave-one-year-out, LOYO) or the 2024 dev year.

## 1. What is scored on 2025 and 2026 (W5, one shot)
The test pools are the existing frozen v3 pools: `v3_full` = 8 `v3_pot` originals + 8 corrected (2026: 204 questions; 2025: 190). No new test generations are made for the verifier work.

| # | Selector | How it is chosen (training data only) |
|---|---|---|
| S0 | verifier v3 (frozen: xgb_ranker / handcrafted / max_prob) | reference |
| S1 | majority vote | reference |
| S2 | verifier v3 refit **without length features** (`log_chars`, `log_tokens`, `n_steps`, `truncated`) | refit on the v3 training pools |
| S3 | neural verifier (ModernBERT-base cross-encoder) alone | the final model trained on 2019–2024 v3 pools |
| S4 | xgb_ranker with the neural score as an extra feature (stacking) | the neural OOF score as the training feature |
| S5 | S3 / S4 retrained with the Phase-4 v3-style pools (`v4_pot_train_*`, `v4_train_corr_*`) | the same, with the extra pools |

**Selection method** (max_prob / weighted_vote / log_odds_vote) per selector: whichever is best on the LOYO 16-pools.

**Gate:** a selector S2–S5 is scored on 2025/2026 only if its LOYO selection accuracy beats the frozen v3 ranker's LOYO score on the same pools:
- on the v3 16-pools: 62.0;
- for S5, also on the Phase-4 pools, against the v3 ranker's LOYO score there.

Otherwise it is reported as a negative LOYO result and not scored.

**Primary statistic:** selector minus majority, and selector minus S0, on 2026 (paired cluster bootstrap over EN/HI twins, 95% CI; McNemar). 2025 is reported alongside. A "positive result" means the lower CI bound against S0 is above 0 on 2026. Anything less is reported as no evidence of improvement.

**Known caveat (S4):** the stacking feature for training rows comes from the neural LOYO models. Those were trained on the other training years, including the year a stacking fold holds out, so the stacking LOYO estimate is slightly optimistic.

## 2. RAG (W4)
- **W4a/W4b, 2024 dev (no test data):**
  - `v4_rag_dev_sol` (v3 solution exemplars) and `v4_rag_dev_card` (method cards), both with matched seeds, each paired against `pot_train` 2024.
  - The effect is reported on all 204 questions and on the questions where RAG fires.
  - **Cards are adopted** if card minus sol > 0 and the paired-bootstrap lower bound is > 0 (the v3 inclusion rule).
- **W4c:**
  - kb_v4 = kb_v3 sources + OlympiadBench (Apache-2.0, text-only, numeric).
  - τ by the same LOYO proxy as v3.
  - Report LOYO coverage at proxy precision ≥ 0.70 against kb_v3 (11.7%).
  - **Post-hoc 2026:** `v4_pot_kb4` (the v3_pot recipe with kb_v4, same seeds) vs `v3_pot`: paired, differing only where retrieval changed. Reported as PoT accuracy overall and on the changed questions. There is no correction re-run, and the framework number is not replaced.
- **W4d (mixed RAG / no-RAG pools):** already measured on existing runs (CHANGELOG_v4). No GPU follow-up.

## 3. What will not happen
- No Phase-4 component replaces the frozen v3 headline (2026: 58.8 vs matched SC 52.5).
- No threshold, method or model is chosen on 2025/2026.
- No re-run of a selector on 2026 after seeing its result.
