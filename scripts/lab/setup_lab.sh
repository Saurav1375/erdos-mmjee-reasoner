#!/usr/bin/env bash
# One-time environment setup on the lab PC (run locally; executes over ssh).
# Creates $LAB_ROOT/.venv with vLLM (which pins torch/transformers) and installs
# this project into it in editable mode. No sudo, no system changes.
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$(dirname "$0")/sync_to_lab.sh"
ssh "$LAB_HOST" bash -s <<REMOTE
set -euo pipefail
cd $LAB_ROOT
[ -d .venv ] || uv venv --python 3.12 .venv
export UV_HTTP_TIMEOUT=600
uv pip install --python .venv/bin/python "vllm>=0.11.0"
uv pip install --python .venv/bin/python -e "$LAB_PROJECT[dev]"
.venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
REMOTE
