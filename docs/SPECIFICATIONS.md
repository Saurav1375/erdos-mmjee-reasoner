# Specifications

This file lists every hardware, software, model and hyper-parameter setting used in phases 1–2.
The authoritative values live in [`configs/`](../configs) and [`requirements*.txt`](..).

## 1. Hardware and OS

| | Lab GPU PC (all model generation) | Laptop (development, CPU stages) |
|---|---|---|
| GPU | NVIDIA GeForce **RTX 3060, 12 GB** (driver 595.91) | none (CPU-only PyTorch) |
| CPU | Intel Core i7-13700 | — |
| RAM | 15 GB (shared machine) | — |
| OS | Ubuntu 24.04.4 LTS | Linux |
| Python | 3.12.3 | 3.12.13 |
| Package manager | uv 0.10.9 | uv 0.11.14 |
| Access | `ssh ml-lab`; everything under `~/saurav_mmjee` (venv, HF cache, code, artifacts) | — |

LaTeX reports compile with **Tectonic 0.16.9**.

## 2. Software (exact versions)

| Purpose | Library | Version |
|---|---|---|
| Inference engine (lab) | vllm | 0.30.0 |
| | torch (CUDA) / torchvision / triton | 2.13.0 / 0.28.0 / 3.7.1 |
| | compressed-tensors (AWQ checkpoints) | 0.17.0 |
| | transformers / tokenizers | 5.18.0 / 0.23.2 |
| Data | datasets / pandas / pyarrow / numpy | 5.0.1 / 3.0.6 / 25.0.1 / 2.5.3 (lab 2.3.5) |
| ML | scikit-learn / xgboost / scipy | 1.9.1 / 3.4.1 / 1.18.1 |
| Embeddings and retrieval | sentence-transformers / faiss-cpu | 6.1.0 / 1.15.1 |
| Sandbox | sympy | 1.14.0 |
| Reports | matplotlib | 3.11.2 |
| Tests | pytest / ruff | 9.1.1 / 0.16.10 |

Full lists: [`requirements.txt`](../requirements.txt) (CPU) and
[`requirements-gpu.txt`](../requirements-gpu.txt) (lab). The resolved tree is in `uv.lock`.

## 3. Models

| Role | Model (Hugging Face) | Params / quantisation | Used for |
|---|---|---|---|
| `solver` | [`cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit) | 8B, AWQ 4-bit | Solver, Corrector, concept tagger, transcriber, same-model critic |
| `critic_cross` | [`cyankiwi/InternVL3_5-8B-AWQ-4bit`](https://huggingface.co/cyankiwi/InternVL3_5-8B-AWQ-4bit) | 8B, AWQ 4-bit | Cross-model Critic |
| `repro` | [`Qwen/Qwen2.5-VL-7B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ) | 7B, AWQ 4-bit | Reproduction of the base paper's number |
| embeddings | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | 33M | Verifier embedding features, RAG index |

No VLM is trained or fine-tuned. Only one model is on the GPU at a time.

### vLLM engine settings (`configs/models.yaml`)

| Setting | solver / critic_cross | repro |
|---|---|---|
| `max_model_len` | 12,288 | 12,288 |
| `gpu_memory_utilization` | 0.88 (0.92 OOMed) | 0.88 |
| `max_num_seqs` | 32 | 32 |
| `max_num_batched_tokens` | 4,096 | 4,096 |
| `enable_prefix_caching` | true | true |
| `kv_cache_dtype` | **fp8** (1.9× KV capacity, measured) | **auto** (fp8 produced garbage tokens) |
| `mm_processor_kwargs` | `max_pixels: 1.7 MP` (solver), default tiling (InternVL) | `max_pixels: 1.7 MP` |
| `limit_mm_per_prompt` | 1 image | 1 image |
| Sampler | PyTorch (`VLLM_USE_FLASHINFER_SAMPLER=0`; flashinfer JIT needed `ninja`) | same |

## 4. Data and protocol

| | Value |
|---|---|
| Dataset | HF `ArkaMukherjee/mmJEE-Eval` (MIT): JEE Advanced 2019–2025 (+2026 rows, excluded), English + Hindi, question images |
| Train / dev | 2019–2024: 1,270 questions; dev year 2024 |
| Test (held out) | 2025: **190** questions = 95 English + 95 Hindi twins |
| Scoring | Verbatim base-paper `extract_answer` / `is_answer_correct` / `calculate_statistics` |
| Metric | Pass@1 = mean over N independent samples |
| CIs | Cluster bootstrap over EN/HI twin pairs, 2,000 resamples |
| Tests | Paired bootstrap, McNemar |
| Seed | 1234; per-call seeds derived from (seed, mode, question, sample, turn) |

## 5. Hyper-parameters (`configs/base.yaml`)

| Component | Setting |
|---|---|
| Sampling (solver) | temperature 0.7, top-p 0.8, top-k 20, presence penalty 0, **max_tokens 4,096** per trajectory |
| Samples per question | CoT baseline **20** (raised from 16 after the pilot); PoT, RAG-PoT and pot_train **8**; repro 10; contamination 1 |
| Sandbox | timeout 10 s, memory 1,024 MB, output ≤ 2,000 chars, ≤ 3 executed blocks, ≤ 8 turns, 4 workers, `force_answer: false` |
| Correction | max 3 rounds, unchanged patience 2, critic greedy (T = 0) ≤ 1,536 tokens, verdict forcing ≤ 96 tokens |
| Verifier | models LogReg (C = 1, balanced) and XGBoost (400 trees, depth 4, lr 0.05, subsample 0.8, colsample 0.8); feature sets handcrafted / embedding / all; embedding tail 2,000 chars, PCA 32; 5-fold GroupKFold |
| RAG | top-k 2, τ auto (proxy precision 0.7 → τ = 0.885), same subject only, dedup cosine 0.90, solution ≤ 3,000 chars; tagger greedy ≤ 160 tokens, transcriber greedy ≤ 1,536 tokens |
| Pilot | 20 test twin pairs (40 questions, stratified by type), 75 train twin pairs |

## 6. Compute used (phase 2 full run)

About **54 GPU-hours**: 14 stages, no retries.

| Stage | Hours |
|---|---|
| CoT (20 × 190) | 1.5 |
| PoT (8 × 190) | 3.3 |
| Correction, same-model critic | 5.4 |
| Correction, cross-model critic | 5.9 |
| PoT on training set (8 × 1,270) | 23.9 |
| RAG-PoT | 3.8 |
| Full correction | 7.7 |
| CoT on training set | 2.8 |
