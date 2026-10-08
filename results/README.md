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
