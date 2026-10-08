# mmJEE-Reasoner: improving scientific reasoning in VLMs on mmJEE-Eval

**Team Erdős**, IIT Bhilai, Machine Learning course project

> *Improving Scientific Reasoning in Vision-Language Models using Code Sandboxes, Agentic
> Correction, Learned Verification, and RAG on mmJEE-Eval*

| Member | Roll no. | Email | GitHub |
|---|---|---|---|
| Arpit Kumar | 12340350 | arpitk@iitbhilai.ac.in | [@arpitkumar0007](https://github.com/arpitkumar0007) |
| Saurav Gupta | 12341940 | sauravg@iitbhilai.ac.in | [@Saurav1375](https://github.com/Saurav1375) |

---

## Phase 1: proposal (this branch)

This branch holds **Phase 1** of the project: the proposal (SOP) and the base paper's code it
builds on. Later phases live on their own branches:

| Phase | Branch | Content |
|---|---|---|
| **1. Proposal and base-paper study** | `phase-1` | SOP, base-paper repository (submodule), code audit |
| 2. Baselines, pilot, full run | `phase-2` | Framework implementation, Qwen2.5/Qwen3 baselines, 40-question pilot, full run on 2025 |
| 3. v3 improvements | `phase-3` | In progress |

**Details: [`docs/phases/PHASE1.md`](docs/phases/PHASE1.md)** (goals, problem statement,
base-paper code audit, issues, and the hand-off to Phase 2).

## Problem in one paragraph

[mmJEE-Eval](https://mmjee-eval.github.io) (Mukherjee & Ghosh, Findings of IJCNLP-AACL 2025)
is a benchmark of 1,460 bilingual JEE Advanced questions, given as images. It shows two things
about open VLMs:
- they trail frontier models by about 40 points;
- they *detect* their own errors far more often than they *fix* them (1.1–5.2%).

The paper diagnoses these weaknesses but does not try to fix them. We propose a training-free,
inference-time framework around a frozen VLM, with four modules:
- **code sandbox** (Program-of-Thoughts);
- **agentic iterative correction** (Solver → Critic → Corrector);
- a **learned solution verifier**, the only trained part;
- **retrieval-augmented exemplars**.

The framework is evaluated only on the contamination-clean 2025 held-out set, against
self-consistency at matched compute.

## Contributions (as stated in the SOP)

- **Arpit Kumar**: Retrieval-Augmented Exemplar Prompting module (knowledge-base construction, retrieval, exemplar injection).
- **Saurav Gupta**: Agentic Iterative Correction module (Solver/Critic/Corrector orchestration).
- **Both**: the Code-Sandbox Tool Use module and the Learned Solution Verifier (data generation, feature design, training, evaluation).

## Repository contents (phase 1)

```
.
├── README.md
├── docs/
│   ├── proposal/SOP_12340350_12341940.pdf   # Statement of Purpose (Phase-1 deliverable)
│   └── phases/PHASE1.md                     # goals, base-paper audit, issues, next steps
└── third_party/mmJEE-Eval/                  # base-paper repo (git submodule @ 815ac89)
```

Fetch the base-paper code with:

```bash
git clone --recurse-submodules -b phase-1 https://github.com/Saurav1375/erdos-mmjee-reasoner.git
# or, inside an existing clone:
git submodule update --init
```

## References

1. A. Mukherjee and S. Ghosh, "mmJEE-Eval: A bilingual multimodal benchmark for evaluating scientific reasoning in vision-language models," *Findings of ACL: IJCNLP-AACL 2025*, pp. 2268–2290. [arXiv:2511.09339](https://arxiv.org/abs/2511.09339) · [code](https://github.com/ArkaMukherjee0/mmJEE-Eval) · [data](https://huggingface.co/datasets/ArkaMukherjee/mmJEE-Eval)
2. W. Chen et al., "Program of Thoughts prompting," *TMLR*, 2023.
3. A. Madaan et al., "Self-Refine: Iterative refinement with self-feedback," *NeurIPS*, 2023.
4. K. Cobbe et al., "Training verifiers to solve math word problems," arXiv:2110.14168, 2021.
5. X. Wang et al., "Self-consistency improves chain of thought reasoning in language models," *ICLR*, 2023.
6. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016.
7. P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," *NeurIPS*, 2020.
