# Phase 4b: final diagnostics (post-hoc): where the gain comes from, answer-level verifier, diagram-first transcript

| | |
|---|---|
| **Branch** | `phase-4b` (built on `phase-4`, which is not changed) |
| **Code** | Additive on top of Phase 4. All new options are off by default, so the v2, v3 and Phase-4 configs still give identical results. Phase-4b outputs go to `artifacts_v4b/` only (`configs/v4b.yaml`). |
| **Pre-registration** | [`V4B_PREREG.md`](../../V4B_PREREG.md), committed and pushed before any Phase-4b run |
| **Report** | [`report/final_v4b/final_report_v4b.pdf`](../../report/final_v4b/final_report_v4b.pdf): the Phase-4 report plus Sec. VIII, "Final diagnostics (Phase 4b)". [`report/final_v4/`](../../report/final_v4) is kept unchanged. |
| **Results** | [`results/v4b/`](../../results/v4b) |

**Status of the numbers.** The v3 clean 2026 result (58.8 vs 52.5 for matched SC) stays the headline. Every Phase-4b number on 2025 or 2026 is post-hoc. Each track has a gate on 2019–2024 only, and only a track that passes it would be scored on 2026, once. Each track is reported whatever its outcome.

**Isolation from Phase 4.** Phase 4b has its own pre-registration, config, output directory, experiment names (`v4b_*`) and branch. The Phase-4 outputs are checksummed (SHA-256 manifest) and read-only.

---

## 1. Goals

Phase 4 left two questions open (`PHASE4.md`):

1. **Why does selection fail on 2026?** Verifier v3 beats majority vote on 2025 (+6.3) but not on 2026 (−0.5), although its candidate AUC is high.
2. **Would a better reading of the figure help?** Most remaining 2026 errors are diagram questions, and all samples of a question share one transcript whose figure description is a single line.

## 2. The three tracks

| Track | What | Cost | Gate (fixed in advance) |
|---|---|---|---|
| **C** | Split the v3 gain over matched SC into pool gain and selection gain; candidate AUC overall vs inside mixed pools | CPU, existing results | none (analysis only) |
| **A** | Answer-level verifier: one row per distinct answer in a pool, pick the most probable answer; 5 models, chosen by the worst LOYO year | CPU, about 1 min | LOYO > 63.0 (v3 16-pools) and > 61.1 (Phase-4 pools), worst year ≥ the v3 ranker's |
| **B** | Diagram-first transcript: a second, itemised reading of the image (labels, values with units, arrows, connections, axes); half of each pool is solved from it with the same seeds | GPU, about 2 h on the 2024 dev year | majority@8 difference > 0 with lower 95% bound > 0 |

## 3. Results

### Track C: the gain comes from the pool (frozen v3 verifier, existing pools)

| | 2025 | 2026 |
|---|---|---|
| Matched self-consistency | 58.4 | 52.5 |
| Majority vote on the v3 pool | 65.8 | 59.3 |
| v3 pick | 72.1 | 58.8 |
| **Pool gain** (majority − SC) | +7.4 [0.0, +15.3] | **+6.9 [+2.0, +11.8]** |
| **Selection gain** (v3 pick − majority) | +6.3 [+1.6, +11.1] | **−0.5 [−2.9, +2.5]** |
| Candidate AUC, all candidates | 0.872 | 0.902 |
| Candidate AUC, inside mixed pools | 0.796 (129 pools) | 0.748 (121 pools) |

The pool gain is stable across years. The selection gain is what does not transfer. The global AUC rises on 2026 because it mostly separates easy from hard questions. Inside a pool, which is all that matters for picking, it falls.

### Track A: answer-level verifier (gate failed, not scored on test)

15,280 training candidates give 5,335 answer rows. LOYO pick accuracy on the v3 16-pools:

| Model | Mean | Worst year |
|---|---|---|
| **XGBoost depth 3, 300 trees (chosen)** | 57.9 | 50.0 |
| v3 ranker (per candidate) | 62.0 | 52.5 |
| Majority vote | 53.7 | |

Aggregating the supporters beats voting by about 4 points but loses about 4 to the per-candidate ranker, whose pick relies on the detail of single solutions.

### Track B: diagram-first transcript (gate failed, 2026 not run)

2024 dev year, 204 questions. Existing: PoT with the v3 transcript, samples 0–7. Mixed: samples 0–3 plus samples 4–7 solved from the diagram-first transcript.

| | All (204) | Diagram (59) | Text (145) |
|---|---|---|---|
| Majority@8, existing → mixed | 65.7 → 63.7 | 42.4 → 39.0 | 75.2 → 73.8 |
| **Difference (gate)** | **−2.0 [−5.4, +1.0]** | −3.4 [−8.5, 0.0] | −1.4 [−6.1, +2.8] |
| Oracle@8, existing → mixed | 79.9 → 78.4 | 61.0 → 59.3 | 87.6 → 86.2 |
| Per sample (s4–s7), current → diagram | 55.0 → 51.6 | 29.7 → 24.6 | 65.3 → 62.6 |
| Truncated (s4–s7), current → diagram | 37.3% → 42.6% | 61.0% → 67.8% | 27.6% → 32.4% |

The second reading adds distinct answers but no right ones, and it hurts most on diagram questions. The longer transcript leaves less of the fixed token budget, so more solutions are cut off. The diagram gap is therefore not a transcription-format problem: the 8B solver misreads or mis-reasons about the figure whichever description it gets.

## 4. Takeaway

On clean data the framework's gain comes from the candidate pool, and that part is stable across years. Selection is fragile. It helps on 2025 and not on 2026 despite a high candidate AUC, and neither a stronger per-candidate verifier (Phase 4) nor answer-level aggregation (Phase 4b) fixes it. A more detailed figure description does not help the solver either.

## 5. Reproduce

```bash
C="--config configs/v3.yaml --config configs/v4.yaml --config configs/v4b.yaml"   # outputs: artifacts_v4b/
mmjee --config configs/v3.yaml --config configs/v3_final.yaml --config configs/v4.yaml \
      --config configs/v4b.yaml train-verifier-v4b          # track A: LOYO table + gate
mmjee $C transcribe --view diagram --years 2024              # track B (GPU)
mmjee $C solve --exp v4b_tx_diag_dev                         # track B dev run (GPU, about 2 h)
python scripts/v4b_analysis.py                               # track C + track-B gate
python scripts/final_report_v4b.py && cd report/final_v4b && tectonic final_report_v4b.tex
```

| Code | Purpose |
|---|---|
| `prompts/templates.py` (`TRANSCRIBE_DIAGRAM_PROMPT`), `rag/tagger.py` (`view="diagram"`) | Diagram-first transcription, its own output file |
| `pipeline/solve.py` (`transcript_view`, `sample_offset`) | Solve from the diagram transcript with the seeds of samples 4–7 |
| `verifier/v4b.py`, CLI `train-verifier-v4b` | Answer-level verifier (track A) |
| `scripts/v4b_analysis.py` | Gain decomposition, within-pool AUC, track-B gate |
| `scripts/final_report_v4b.py` | Report Sec. VIII tables, macros and figure |
| `tests/test_v4b.py` | Seed offset, default unchanged, separate diagram file, view selection, answer rows |
