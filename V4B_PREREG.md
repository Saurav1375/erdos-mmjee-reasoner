# Phase 4b pre-registration (written 2026-10-22, before any Phase-4b run)

Phase 4b is a separate, small follow-up to Phase 4. **Phase 4 is closed and stays as it is:** its pre-registration (`V4_PREREG.md`), outputs (`artifacts_v4/`, unchanged and checked against a SHA-256 manifest), results (`results/v4/`), report (`report/final_v4/`) and branch (`phase-4`, c829c41). Phase 4b runs with `configs/v4b.yaml` after the Phase-4 configs, writes only to `artifacts_v4b/` (seeded with a copy of `artifacts_v4/`), uses experiment names `v4b_*`, and lives on branch `phase-4b`.

Phase 4 ended with every extension null or negative (CHANGELOG_v4, `docs/phases/PHASE4.md`). The 2026 errors split into 42 selector misses and 42 questions with no right answer in the pool (W1). Phase 4b runs two small checks aimed at these two halves, plus one analysis of existing results. The rules of `V4_PREREG.md` apply here too: choices use 2019–2024 only, the v3 headline (2026: 58.8 vs matched SC 52.5) stays, and every Phase-4b number on 2025/2026 is post-hoc. **Each track is reported whatever its outcome.** Experiment freeze: 2026-10-25, 23:59 IST; anything unfinished then is reported as not done.

### A. Answer-level verifier, chosen by its worst year (CPU, existing pools only)
- **Rows:** one per distinct answer (`answer_key`) in each pool of the Phase-4 training frame (`load_oof(cfg, "extra", with_extra=True)`: v3 pools + Phase-4 v3-style pools, 2019–2024). Label: the answer is correct.
- **Features (fixed):**
  - vote share, vote rank (0 = most votes, scaled to [0, 1]), share gap to the top answer, number of distinct answers / pool size;
  - share of the pool's originals and of its corrected candidates that give this answer (−1 if the pool has no corrected candidates);
  - means over the answer's supporters of `code_ok_frac`, `truncated`, `log_tokens`, `n_steps`, `hedges_per_kchar`, `valid_format`;
  - means over supporters with a chain of `corr_critic_flagged_first` and `corr_answer_changed` (−1 if none);
  - question type (4 one-hot), `requires_image`, `lang_hindi`.
- **Models (5 configurations):** logistic regression (standardised, C = 1); XGBoost classifier with depth {2, 3} × trees {100, 300}, learning rate 0.05, other settings default.
- **Selection rule:** the answer with the highest predicted probability in each pool (ties: the earliest answer).
- **Model choice:** LOYO over 2019–2024; the configuration with the best **worst-year** pick accuracy on the v3 16-pools (ties: best mean).
- **Gate:** on the v3 16-pools the LOYO pick must be > **63.0** (the v3 ranker's best LOYO, 62.0, + 1.0 for row-order noise); on the Phase-4 pools > **61.1** (60.1 + 1.0); and its worst year must be ≥ the worst year of the v3 ranker (xgb_ranker / max_prob, base frame) on the v3 16-pools.
- **If it passes:** the chosen configuration is fit on all training rows and scored once on the 2025 and 2026 `v3_full` pools against S0 and majority, with the statistics of `V4_PREREG.md` §1. Otherwise only the LOYO table is reported.
- **Caveat:** the reported LOYO number of the chosen configuration is slightly optimistic (5 configurations were compared on the same folds).

### B. Diagram-first transcript as a second reading (GPU)
- **Second reading:** a new greedy transcription (seed 0, `rag.transcribe_sampling`) with this prompt, output in English for both languages (no translate stage):

  > Read this JEE Advanced exam question (it may be written in English or Hindi) and write it out in English. Do not solve it. Use exactly two parts:
  > QUESTION: the full question text and all answer options, with mathematics in LaTeX.
  > FIGURE: if there is a figure, describe it as a list with one item per line: every label and symbol, every given value with its unit, every arrow and its direction, how the parts are connected or placed relative to each other, and any axes, coordinates, curves or angles. Write "FIGURE: none" if there is no figure.
  > Output only these two parts.

- **Dev experiment (2024, 204 questions, config `configs/v3.yaml` + `configs/v4.yaml` + `configs/v4b.yaml`, i.e. the generation settings of `pot_tx_dev`):** `v4b_tx_diag_dev` = PoT with the diagram transcript in the transcript block, samples s4–s7 with the seeds of `pot_tx_dev` s4–s7 (same mode `pot_tx`, same sample index), so paired samples differ only in the transcript.
  - Reference pool: `pot_tx_dev` s0–s7. Multi-view pool: `pot_tx_dev` s0–s3 + `v4b_tx_diag_dev` s4–s7. Same number of samples.
  - **Gate:** majority vote at 8 samples, multi-view − reference > 0 **and** the paired cluster-bootstrap (EN/HI twins, 2,000 resamples) lower 95% bound > 0.
  - **Always reported, not gating:** matched per-sample accuracy (diagram vs current transcript on s4–s7), oracle@8, distinct answers per pool, tokens and truncation, each also split by `requires_image`.
- **If and only if the gate passes:** 2026 diagram transcription + `v4b_tx_diag_test` (the `v3_pot` recipe, `configs/v3_final.yaml` + `configs/v4b.yaml`, RAG unchanged, samples s4–s7 with `v3_pot`'s seeds), scored once: majority@8 and the frozen PoT-pool verifier v3 on `v3_pot` s0–s3 + diagram s4–s7, paired against `v3_pot` s0–s7. No correction re-run; the framework number is not replaced.

### C. Analysis of existing results (no new method, nothing tuned)
- Decomposition of the gain over matched SC on 2025 and 2026 into pool gain (majority on the v3 pool − matched SC) and selection gain (v3 pick − majority on the v3 pool).
- Candidate-level AUC vs pick accuracy, and the 2026 error budget (correct / selector miss / never in the pool, by diagram vs text), from the existing W1 outputs.
