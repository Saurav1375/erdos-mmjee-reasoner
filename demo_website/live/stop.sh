#!/usr/bin/env bash
# Stop everything start.sh started: the demo server, the ssh tunnels and both lab model servers.
# Nothing of the demo is left running on the lab PCs afterwards.
# usage: bash demo_website/live/stop.sh [--local]      (--local: laptop side only)
set -uo pipefail
LIVE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_HOST="${SOLVER_HOST:-ml-lab2-lan}"; SOLVER_JUMP="${SOLVER_JUMP:--J ml-lab}"
CRITIC_HOST="${CRITIC_HOST:-ml-lab}";      CRITIC_JUMP="${CRITIC_JUMP:-}"

if [ -f "$LIVE/.run/server.pid" ]; then
  kill "$(cat "$LIVE/.run/server.pid")" 2>/dev/null && echo "demo server stopped"
  rm -f "$LIVE/.run/server.pid"
fi
pkill -f '^ssh .*-L 8001:127.0.0.1:' && echo "solver tunnel closed"
pkill -f '^ssh .*-L 8002:127.0.0.1:' && echo "critic tunnel closed"
if [ "${1:-}" != "--local" ]; then
  # shellcheck disable=SC2086
  ssh -o ConnectTimeout=10 -o BatchMode=yes $SOLVER_JUMP "$SOLVER_HOST" 'bash ~/saurav_mmjee/demo/serve.sh stop' 2>/dev/null
  # shellcheck disable=SC2086
  ssh -o ConnectTimeout=10 -o BatchMode=yes $CRITIC_JUMP "$CRITIC_HOST" 'bash ~/saurav_mmjee/demo/serve.sh stop' 2>/dev/null
fi
true
