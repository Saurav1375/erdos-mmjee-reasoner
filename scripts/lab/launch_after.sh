#!/usr/bin/env bash
# Queue a supervised stage chain on a lab PC behind a running one: a detached waiter starts
# `supervise.sh <name>` once the supervisor of <prev> has exited with ALL_DONE (not on FAILED).
#   scripts/lab/launch_after.sh <prev> <name> "<mmjee args>" ["<mmjee args>" ...]
# Progress: $RUN_ARTIFACTS/logs/<name>.supervisor.out (the waiter writes "WAITING ..." first).
# The waiter is written to $RUN_ARTIFACTS/logs/<name>.waiter.sh (no nested quoting, so each
# "<mmjee args>" stays one stage); args must not contain single quotes.
set -euo pipefail
source "$(dirname "$0")/env.sh"
PREV="$1"; NAME="$2"; shift 2
ARGS=""
for a in "$@"; do ARGS+=" '$a'"; done
ssh "$LAB_HOST" bash -s <<REMOTE
set -euo pipefail
cd $LAB_PROJECT
mkdir -p $RUN_ARTIFACTS/logs
OUT=$RUN_ARTIFACTS/logs/$NAME.supervisor.out
W=$RUN_ARTIFACTS/logs/$NAME.waiter.sh
cat > \$W <<'WAITER'
while pgrep -f "[s]upervise.sh $PREV " >/dev/null; do sleep 120; done
if tail -1 $RUN_ARTIFACTS/logs/$PREV.supervisor.out | grep -q ALL_DONE; then
  exec bash scripts/lab/supervise.sh $NAME $ARGS
else
  echo "\$(date) FAILED: $PREV did not finish with ALL_DONE - not starting"
fi
WAITER
bash -n \$W
echo "\$(date) WAITING for $PREV" > \$OUT
RUN_ARTIFACTS=$RUN_ARTIFACTS nohup setsid bash \$W >> \$OUT 2>&1 < /dev/null &
sleep 1; cat \$OUT
REMOTE
