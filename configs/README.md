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
