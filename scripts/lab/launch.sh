#!/usr/bin/env bash
# Start a supervised stage chain on the lab PC, detached from ssh (survives logout; linger=yes).
#   scripts/lab/launch.sh <name> "<mmjee args>" ["<mmjee args>" ...]
# Progress: $RUN_ARTIFACTS/logs/<name>.supervisor.out (ends with ALL_DONE or FAILED)
# v3 runs: RUN_ARTIFACTS=artifacts_v3 scripts/lab/launch.sh <name> "--config configs/v3.yaml <stage> ..."
# Stop:     ssh ml-lab 'pkill -f "[s]upervise.sh <name>"; pkill -f "[m]mjee_reasoner"'
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME="$1"; shift
ARGS=""
for a in "$@"; do ARGS+=" '$a'"; done
ssh "$LAB_HOST" bash -s <<REMOTE
set -euo pipefail
cd $LAB_PROJECT
used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "\$used" -gt 1500 ]; then echo "GPU busy (\${used} MiB used) - not starting"; exit 1; fi
ln -sfn $LAB_ROOT/.venv .venv
mkdir -p $RUN_ARTIFACTS/logs
RUN_ARTIFACTS=$RUN_ARTIFACTS nohup setsid bash scripts/lab/supervise.sh $NAME $ARGS > $RUN_ARTIFACTS/logs/$NAME.supervisor.out 2>&1 < /dev/null &
sleep 2; cat $RUN_ARTIFACTS/logs/$NAME.supervisor.out
REMOTE
