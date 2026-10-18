# Configuration (`configs/`)

Plain YAML. `load_config` (in [`src/mmjee_reasoner/config.py`](../src/mmjee_reasoner/config.py))
deep-merges `base.yaml` and `models.yaml`, then any `--config <file>` overrides, in the order
given. Relative paths resolve against the project root.

| File | Contents |
|---|---|
| `base.yaml` | Paths, seed, year split, pilot subset size, sampling defaults, and the settings of each module (sandbox, correction, verifier, rag). Also the evaluation settings, including the definition of **SOP Table I**. |
| `models.yaml` | Model roles and vLLM engine settings: `solver` (Qwen3-VL-8B AWQ), `critic_cross` (InternVL3.5-8B AWQ), `repro` (Qwen2.5-VL-7B AWQ). |
| `experiments/*.yaml` | One file per generation run. `kind: solve` (mode cot/pot, split, n_samples, rag, model) or `kind: correct` (source, critic). See [`pipeline/README.md`](../src/mmjee_reasoner/pipeline/README.md). |

## Key settings (phase 2)

| Setting | Value | Why |
|---|---|---|
| `sampling` | T = 0.7, top-p 0.8, top-k 20, max_tokens 4096 | Qwen3-VL Instruct recommended values; one budget for every configuration |
| `sandbox` | 10 s timeout, 1 GB memory, ≤3 code blocks, ≤8 turns, 4 workers, `force_answer: false` | Safety, and lab RAM (15 GB) |
| `correction` | R = 3 rounds, patience 2, greedy critic ≤1,536 tokens | SOP stopping rule |
| `verifier` | bge-small-en-v1.5, tail 2,000 chars, PCA 32, 5-fold CV, dev year 2024 | No 2025 data in selection |
| `rag` | top-k 2, τ auto (proxy precision 0.7), same subject only, dedup 0.90 | Retrieve nothing on a weak match |
| `eval` | 2,000 bootstrap resamples | Twin-cluster CIs |

`--mock` runs write to `artifacts_mock/`. `--limit N` and `--pilot` add a `.limitN` or
`.pilot` suffix to every output.

## v3 overrides (phase 3)

All v3 outputs go to `artifacts_v3/`. The v2 settings above stay the defaults, so `base.yaml` alone still reproduces the Phase-2 results.

| File | Contents |
|---|---|
| `v3.yaml` | Base v3 override: `artifacts_v3/`, verifier v3 and KB v3 directories, KB de-duplication against 2025 and 2026 |
| `v3_final.yaml` | **Frozen v3 framework** (`V3_FREEZE.md`): RAG v3 (hybrid w = 0.9, JEEBench, LOYO τ) and the adaptive budget |
| `v3_test2026.yaml` | Switches the test set to the clean 2026 year (204 questions); outputs get the `.y2026` tag |
| `v3_v2fw.yaml` | The frozen v2 framework (v2 KB and verifier) for the 2026 re-run |
| `v3_eval.yaml`, `v3_eval_v2fw.yaml`, `v3_eval_q25.yaml` | Table I definitions for v3, the v2 re-run and the Qwen2.5 bridge (evaluation only) |
| `v3_ext.yaml`, `v3_kbdev_v2.yaml`, `v3_kbdev_v3.yaml` | 2024 dev A/B settings (adaptive budget; dev KBs without 2024) |
| `experiments/v3_pot.yaml`, `v3_full.yaml`, `cot_ext.yaml` | The frozen v3 runs: PoT + RAG v3 + transcript, cross-critic correction, and CoT with the adaptive budget |
| `experiments/q25_*.yaml`, `probe_*.yaml`, `*_dev*.yaml`, `train_corr_cross.yaml`, `kb_ext_pot.yaml` | Qwen2.5 bridge, memorisation probes, dev A/Bs, verifier-v3 training pools, JEEBench solutions |
