#!/usr/bin/env bash
# Laptop side: start the two model servers on the lab PCs, open ssh tunnels, start the demo
# server and open the page. Safe to run again (running parts are reused).
#
# usage (from anywhere):  bash demo_website/live/start.sh            # lab + demo
#                         bash demo_website/live/start.sh --check    # only: are both GPUs free?
#                         bash demo_website/live/start.sh --local    # demo only (lab offline)
#
# One model per PC, each at the frozen GPU share (configs/models.yaml):
#   solver (Qwen3-VL-8B)   on SOLVER_HOST, default ml-lab2 (through ml-lab on the campus LAN)
#   critic (InternVL3.5-8B) on CRITIC_HOST, default ml-lab
# Both GPUs must be free: serve.sh refuses to start next to someone else's job.
set -uo pipefail
LIVE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$LIVE/../.."
PORT="${PORT:-8765}"
SOLVER_HOST="${SOLVER_HOST:-ml-lab2-lan}"; SOLVER_JUMP="${SOLVER_JUMP:--J ml-lab}"
CRITIC_HOST="${CRITIC_HOST:-ml-lab}";      CRITIC_JUMP="${CRITIC_JUMP:-}"
FRAC="${FRAC:-0.88}"
RUN="$LIVE/.run"; mkdir -p "$RUN"

lab() {  # jump host command...
  local jump=$1 host=$2; shift 2
  # shellcheck disable=SC2086
  ssh -o ConnectTimeout=10 -o BatchMode=yes $jump "$host" "$@"
}
gpu_free() {  # jump host name -> prints a line, returns 1 if the frozen share is not free
  local out
  out=$(lab "$1" "$2" 'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader') \
    || { echo "  $3 ($2): not reachable"; return 1; }
  python3 - "$out" "$FRAC" "$3" "$2" <<'PY'
import sys
lines = sys.argv[1].strip().splitlines(); frac = float(sys.argv[2])
total, used = map(int, lines[0].split(","))
need = int(total * frac) + 256
ok = total - used >= need
print(f"  {sys.argv[3]} ({sys.argv[4]}): {used} MiB of {total} MiB in use -> "
      + ("free" if ok else f"NOT free (needs {need} MiB); processes: {'; '.join(lines[1:]) or '?'}"))
sys.exit(0 if ok else 1)
PY
}
start_role() {  # role jump host local_port
  # shellcheck disable=SC2086
  lab "$2" "$3" 'mkdir -p ~/saurav_mmjee/demo' && scp -q $2 "$LIVE/lab/serve.sh" "$3:saurav_mmjee/demo/serve.sh" &&
  lab "$2" "$3" "bash ~/saurav_mmjee/demo/serve.sh $1 $FRAC 8000"
}
tunnel() {  # jump host local_port
  curl -sf -m 2 "http://127.0.0.1:$3/health" >/dev/null && return 0
  pkill -f "^ssh .*-L $3:127.0.0.1:" 2>/dev/null
  # shellcheck disable=SC2086
  ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 $1 \
    -L "$3:127.0.0.1:8000" "$2" >/dev/null 2>&1 < /dev/null && echo "  tunnel localhost:$3 -> $2:8000"
}
wait_up() {  # local_port name
  for i in $(seq 1 100); do
    curl -sf -m 2 "http://127.0.0.1:$1/health" >/dev/null && { echo "  $2 ready after $((i * 5)) s"; return 0; }
    sleep 5
  done
  echo "!! $2 did not come up - see ~/saurav_mmjee/demo_logs on the lab PC"; return 1
}

if [ "${1:-}" != "--local" ]; then
  echo "== GPUs"
  up_s=$(curl -sf -m 2 http://127.0.0.1:8001/health >/dev/null && echo 1)
  up_c=$(curl -sf -m 2 http://127.0.0.1:8002/health >/dev/null && echo 1)
  ok=1
  [ -n "$up_s" ] && echo "  solver already running" || gpu_free "$SOLVER_JUMP" "$SOLVER_HOST" solver || ok=0
  [ -n "$up_c" ] && echo "  critic already running" || gpu_free "$CRITIC_JUMP" "$CRITIC_HOST" critic || ok=0
  [ "${1:-}" = "--check" ] && exit $((1 - ok))
  if [ "$ok" = 0 ]; then
    echo "!! A GPU is not free, so the live pipeline cannot start. The page still works without it:"
    echo "   replay, sandbox, verifier what-if, retrieval search and recorded runs (start.sh --local)."
    exit 1
  fi
  echo "== starting the models (about 3-4 min, both load in parallel)"
  [ -z "$up_s" ] && { start_role solver "$SOLVER_JUMP" "$SOLVER_HOST" || exit 1; }
  [ -z "$up_c" ] && { start_role critic "$CRITIC_JUMP" "$CRITIC_HOST" || exit 1; }
  sleep 15
  tunnel "$SOLVER_JUMP" "$SOLVER_HOST" 8001
  tunnel "$CRITIC_JUMP" "$CRITIC_HOST" 8002
  wait_up 8001 "solver (Qwen3-VL-8B)" && wait_up 8002 "critic (InternVL3.5-8B)" || exit 1
fi

if ! curl -sf -m 2 "http://127.0.0.1:$PORT/api/health" >/dev/null; then
  nohup "$PROJECT/.venv/bin/python" "$LIVE/server.py" --port "$PORT" > "$RUN/server.log" 2>&1 < /dev/null &
  echo $! > "$RUN/server.pid"
  for i in $(seq 1 60); do curl -sf -m 2 "http://127.0.0.1:$PORT/api/health" >/dev/null && break; sleep 1; done
fi
curl -s "http://127.0.0.1:$PORT/api/health"; echo
echo "demo: http://localhost:$PORT"
[ -z "${NO_OPEN:-}" ] && command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT" >/dev/null 2>&1 &
