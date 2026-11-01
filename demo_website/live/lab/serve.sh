#!/usr/bin/env bash
# Runs ON A LAB PC: starts one model as an OpenAI-compatible vLLM server on 127.0.0.1:8000
# (reachable only through an ssh tunnel). Settings mirror configs/models.yaml and the offline
# engine (llm/engine.py); --generation-config vllm keeps the model's own default sampling
# settings out, as offline vLLM does when every request carries its sampling parameters.
#
# usage: bash serve.sh solver|critic [gpu_fraction] [port]   (defaults 0.88 as models.yaml, 8000)
#        bash serve.sh stop
# One model per PC: two 8B models do not fit one 12 GB GPU, and vLLM sleep mode (one model
# parked in CPU memory) ran the 15 GB-RAM lab PC out of memory (tested 2026-10-31).
set -euo pipefail
ROOT=~/saurav_mmjee
LOGDIR=$ROOT/demo_logs
mkdir -p "$LOGDIR"
ROLE="${1:?solver|critic|stop}"
FRAC="${2:-0.88}"
PORT="${3:-8000}"

if [ "$ROLE" = stop ]; then
  for f in "$LOGDIR"/*.pid; do
    [ -f "$f" ] || continue
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then kill -- -"$pid" 2>/dev/null || kill "$pid"; echo "stopped $(basename "$f" .pid) ($pid)"; fi
    rm -f "$f"
  done
  exit 0
fi

export HF_HOME=$ROOT/hf_cache VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PATH=$ROOT/.venv/bin:$PATH
# Both PCs already hold the two models in $HF_HOME: never wait on the network for them.
export HF_HUB_OFFLINE=1

case "$ROLE" in
  solver) ID=cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit; NAME=qwen3vl8b-awq
          EXTRA=(--mm-processor-kwargs '{"max_pixels": 1700000}') ;;
  critic) ID=cyankiwi/InternVL3_5-8B-AWQ-4bit; NAME=internvl3.5-8b-awq
          EXTRA=(--trust-remote-code) ;;
  *) echo "unknown role $ROLE"; exit 1 ;;
esac

if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "$(hostname): a model server is already running on port $PORT"; exit 0
fi

# Never squeeze out other people's jobs: start only if the requested share is free.
total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
need=$(python3 -c "print(int($total * $FRAC) + 256)")
free=$((total - used))
if [ "$free" -lt "$need" ]; then
  echo "$(hostname): GPU has $free MiB free, $need MiB needed for $ROLE at $FRAC - not starting"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  exit 1
fi

# --mm-processor-cache-gb 0: vLLM 0.30's image cache goes out of sync when a request with an image
# is rejected (prompt longer than max-model-len), and later requests with that image then fail
# with 500 "multi-modal cache drift" (seen 2026-10-31). Without the cache each request's image is
# processed again; the outputs are the same.
# On a shared GPU (share below 0.85) CUDA graphs are skipped: ~0.2 GB more for the KV cache.
if python3 -c "import sys; sys.exit(0 if $FRAC < 0.85 else 1)"; then EXTRA+=(--enforce-eager); fi

cd "$ROOT"
nohup setsid vllm serve "$ID" --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" --seed 0 --generation-config vllm \
  --max-model-len 12288 --gpu-memory-utilization "$FRAC" \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --enable-prefix-caching --kv-cache-dtype fp8 \
  --limit-mm-per-prompt '{"image": 1}' --mm-processor-cache-gb 0 "${EXTRA[@]}" \
  > "$LOGDIR/$ROLE.log" 2>&1 < /dev/null &
echo $! > "$LOGDIR/$ROLE.pid"
echo "$(hostname): starting $ROLE ($NAME) on port $PORT at gpu fraction $FRAC, pid $!, log $LOGDIR/$ROLE.log"
