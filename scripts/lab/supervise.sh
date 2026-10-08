#!/usr/bin/env bash
# Runs ON THE LAB PC. Executes `mmjee` stages one after another and retries a stage (up to 3
# attempts) if it dies; stages are resumable through the generation/execution caches.
#
# Start it with scripts/lab/launch.sh (nohup + setsid from the ssh session), NOT inside tmux or
# `systemd-run --user`: on this PC systemd-oomd applies a memory-PRESSURE kill policy to
# everything under user@1000.service (tmux, user scopes). Model swaps (Qwen <-> InternVL, 6-7 GB
# of weights) cause short page-cache pressure spikes although >13 GB RAM is available, and oomd
# killed the job twice. ssh session scopes are outside that policy (the kernel OOM killer still
# applies normally).
#
# usage: supervise.sh <name> "<mmjee args>" ["<mmjee args>" ...]
#   log of each stage: artifacts/logs/<name>.out ; progress: artifacts/logs/<name>.supervisor.out
set -uo pipefail
ROOT=~/saurav_mmjee; PROJ=$ROOT/mmjee-reasoner; LOG=$PROJ/artifacts/logs
NAME="$1"; shift
export HF_HOME=$ROOT/hf_cache VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
export PATH=$ROOT/.venv/bin:$PATH
cd "$PROJ"
mkdir -p "$LOG"
for stage in "$@"; do
  ok=0
  for attempt in 1 2 3; do
    echo "$(date) START [$attempt] $stage"
    # shellcheck disable=SC2086
    .venv/bin/python -m mmjee_reasoner $stage >> "$LOG/$NAME.out" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then echo "$(date) OK $stage"; ok=1; break; fi
    echo "$(date) EXIT $rc on attempt $attempt: $stage"
    sleep 90   # let the GPU / RAM settle before retrying
  done
  if [ $ok -ne 1 ]; then echo "$(date) FAILED $stage"; exit 1; fi
done
echo "$(date) ALL_DONE"
