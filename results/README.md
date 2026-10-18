# Published results (`results/`)

These are small, curated copies of the evaluation outputs. The full artifacts are about 840 MB
(dataset images, the generation cache, 8–20 samples per question, correction chains, verifier
models), so they are not in git. They can be regenerated from the cache-backed pipeline.

## `phase2/`: full run on the 2025 held-out set (190 questions)

| File | Contents |
|---|---|
| `results.json` | Everything `mmjee evaluate` computes: Table I (with CIs, costs and breakdowns), full vs. matched SC (paired bootstrap and McNemar), the SC curve, correction statistics, the verifier on 2025, the contamination check, and the Qwen2.5-VL reproduction |
| `per_question.csv` | Per-question correctness for every Table I row |
| `results.lab_baseline_only.json` | Pre-full-run evaluation (16-sample baselines), used by the Phase-2 report |
| `results.pilot.json`, `per_question.pilot.csv` | 40-question pilot subset (20 EN/HI twin pairs of 2025), latest pilot evaluation |
| `verifier_report.json` | Verifier model selection: CV and 2024-dev metrics for each model × feature set, and the primary choice |
| `kb_meta.json` | RAG knowledge-base build: entry counts, de-duplication, tuned τ |
| `prepare_report.json` | Dataset preparation: split sizes and applied data fixes |

## `v3/`: Phase 3, clean 2026 test and 2025 post-hoc

| File | Contents |
|---|---|
| `results.y2026.v3.json`, `per_question.y2026.v3.csv` | v3 Table I on the clean 2026 test (204 questions), with SC matching, correction, gate and verifier statistics |
| `results.y2026.v2fw.json`, `per_question.y2026.v2fw.csv` | The frozen v2 framework re-run on 2026 |
| `results.v3.json`, `per_question.v3.csv` | v3 Table I on 2025 (**post-hoc**) |
| `results.q25.json`, `per_question.q25.csv` | Bridge: the framework with Qwen2.5-VL-7B on 2025 (clean for that model) |
| `verifier_v3_report.json` | Verifier v3 LOYO model selection, primary choice and correction-gate δ |
| `analysis.json` | Paper-style metacognition (EP/EC), pass@k, critic gating, memorisation probes |
| `dev_ab.json` | 2024 dev A/B tests behind the freeze (transcript block, adaptive budget, RAG v2 vs v3) |
| `rag_sweep_loyo.json` | RAG v3 LOYO sweep (coverage at fixed proxy precision) |

