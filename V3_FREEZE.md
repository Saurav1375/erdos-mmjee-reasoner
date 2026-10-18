# v3 freeze (2026-10-09)

This file fixes the **v3 framework before any final run**. Every choice below was made using
only 2019–2024 data (training years, leave-one-year-out (LOYO) CV, or the 2024 dev year).
Nothing here may change after the final runs start. A change needs a new freeze file and must
be reported as such.

- **Frozen config:** `configs/v3_final.yaml` (on top of `configs/v3.yaml`).
- **Experiments:** `cot_ext`, `v3_pot`, `v3_full`.
- **Evidence:** `CHANGELOG_v3.md`, `artifacts_v3/analysis/{dev_ab,rag_sweep_loyo,analysis}.json`, `artifacts_v3/verifier_v3/report.json`.

## Components

| Component | Frozen choice | Evidence (training years only) |
|---|---|---|
| Substrate | Qwen3-VL-8B AWQ (frozen), sampling unchanged from v2 (T 0.7, top-p 0.8, top-k 20, 4,096 tokens per call) | — |
| **Adaptive budget** | `sampling.extend_on_truncation: 4096`: a cut-off answer continues once for 4,096 more tokens. Applies to CoT, PoT, CoT fallback and the corrector, so it applies to *every* row, including baselines. | 2024 dev, PoT: **+2.33 [+0.98, +3.74]** (rule "gain > 0 and CI lower bound > 0": passes); CoT: +6.37 [+2.45, +10.78]. Cost: +65% tokens per PoT sample. |
| **Transcript block** | `transcript: true` in `v3_pot`: the question's English transcription (Hindi is translated) is added to the PoT prompt. Same VLM, extra calls counted in compute. | 2024 dev: **+4.90 [+1.78, +8.15]**; Hindi +8.7 [+3.9, +13.8]; no group significantly worse (rule passes). |
| **RAG v3** | KB `kb_v3`: verified `pot_train` solutions 2019–2024 + JEEBench 2016–2018 (`kb_ext_pot`). Hybrid retrieval (tag + problem text, w = 0.9), τ tuned LOYO (proxy precision 0.7), top-2, same subject, de-duplicated against 2025 and 2026. | LOYO coverage 11.7% vs 8.4% for the v2 recipe at the same proxy precision. 2024 dev accuracy ties v2 (+0.12 [−0.74, +1.04]). |
| Code sandbox | Unchanged from v2 | — |
| Correction | Unchanged from v2: cross-model critic InternVL3.5-8B, ≤3 rounds, patience 2. The corrector prompt does **not** get the transcript block (kept identical to the version the gate was trained on). | — |
| **Verifier v3** | `xgb_ranker` / handcrafted (+ chain and pool-half features) / `max_prob` over 8 originals + 8 corrected | LOYO 2019–2024, 16-answer pools: 62.0 vs majority 53.7 (oracle 78.5) |
| **Correction gate** | Keep a corrected answer only if score(corrected) − score(original) > δ, **δ = −0.08**. Used for the "+ correction (gated)" row; the framework's final answer is the verifier's pick from the full pool. | `train_corr_cross` (242 questions): 48.2% gated vs 47.6% ungated; right→wrong 32 → 14. Nested LOYO estimate 48.0%. |

## Final runs

| Batch | Data | Rows |
|---|---|---|
| `v3_2026` | **2026 clean test** (204 = 102 EN + 102 HI; Hindi crops built by us) | v2 Table I re-run (`cot`, `pot`, `pot_corr_cross`, `pot_rag` with the v2 KB, `full`, v2 verifier); v3 rows (`cot_ext` = baseline + matched self-consistency, `v3_pot`, `v3_full`, verifier v3 + gate) |
| `v3_2025` | 2025 | v3 rows (`cot_ext`, `v3_pot`, `v3_full`). **Post-hoc:** 2025 was used to pilot v1/v2 and Qwen3-VL may have seen it. |

## Known limitations, stated in advance

- **Distribution shift for the verifier.** It was trained on plain PoT solutions, without the transcript block or continuation. Length and truncation features shift at test time. No retraining is done, because there is no data without extra GPU time; the effect is reported as is.
- **Shared transcriber.** The transcript block and the KB problem texts come from the same VLM (no external OCR).
- **Our Hindi crops.** Hindi 2026 crops come from the official papers, in a different style from the dataset's images. Results are reported per language.
