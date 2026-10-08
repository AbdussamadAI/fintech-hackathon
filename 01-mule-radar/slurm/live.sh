#!/usr/bin/env bash
# Run the MuleRadar live demo on the A100 node and open it on this laptop.
#   1. needs a running GPU job (start one with ../slurm-notebook.sh -p A100 -g 1 -t 08:00:00)
#   2. syncs code, starts api/server.py inside that job, tunnels it to http://localhost:$LOCAL_PORT
# Ctrl-C stops the server and closes the tunnel (the GPU job keeps running).
# Usage: slurm/live.sh [-l local_port] [-r remote_port] [-j jobid]
set -euo pipefail
HOST=slurm
LOCAL_PORT=8050
REMOTE_PORT=8050
JOB=""
while getopts "l:r:j:h" opt; do
  case $opt in
    l) LOCAL_PORT=$OPTARG ;; r) REMOTE_PORT=$OPTARG ;; j) JOB=$OPTARG ;;
    *) echo "Usage: $0 [-l local_port] [-r remote_port] [-j jobid]"; exit 1 ;;
  esac
done
cd "$(dirname "$0")/.."
REMOTE_DIR=fintech-hackathon/01-mule-radar

if ! ssh -O check "$HOST" 2>/dev/null; then
  echo "Opening SSH connection to $HOST (enter your cluster password)..."
  ssh -fN "$HOST"
fi

if [[ -z "$JOB" ]]; then
  read -r JOB NODE <<<"$(ssh "$HOST" "squeue -h -u \$USER -t R -p A100 -o '%i %N' | head -1")"
else
  NODE=$(ssh "$HOST" "squeue -h -j $JOB -o %N")
fi
if [[ -z "${JOB:-}" ]]; then
  echo "No running A100 job found. Start one first:  ./slurm-notebook.sh -p A100 -g 1 -t 08:00:00"
  exit 1
fi
echo "Using job $JOB on $NODE"

./slurm/sync.sh >/dev/null
python3 dashboard/build.py results/demo.json >/dev/null && rsync -az dashboard/index.html "$HOST:$REMOTE_DIR/dashboard/"
ssh "$HOST" "~/envs/mule13/bin/python -c 'import fastapi, uvicorn' 2>/dev/null || ~/envs/mule13/bin/pip install -q fastapi uvicorn"

echo "Starting the live server on $NODE:$REMOTE_PORT (loads data and compiles kernels, ~30-60 s)..."
SPID=$(ssh "$HOST" "cd $REMOTE_DIR && JOB=$JOB nohup slurm/run.sh python -u api/server.py --port $REMOTE_PORT > live.log 2>&1 & echo \$!")

FORWARD=""
cleanup() {
  trap - EXIT INT TERM
  [[ -n "$FORWARD" ]] && ssh -O cancel -L "$FORWARD" "$HOST" 2>/dev/null || true
  echo; echo "Stopping the live server..."
  ssh "$HOST" "kill $SPID 2>/dev/null" || true
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

while :; do
  LOG=$(ssh "$HOST" "cat $REMOTE_DIR/live.log 2>/dev/null")
  if grep -q "^ready" <<<"$LOG"; then break; fi
  if grep -q "Traceback" <<<"$LOG" || ! ssh "$HOST" "kill -0 $SPID 2>/dev/null"; then
    echo "Server failed to start:"; echo "$LOG" | grep -v -i warning | tail -20; exit 1
  fi
  printf '.'; sleep 3
done
echo; grep "^ready" <<<"$LOG"

while lsof -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; do LOCAL_PORT=$((LOCAL_PORT + 1)); done
FORWARD="$LOCAL_PORT:$NODE:$REMOTE_PORT"
ssh -O forward -L "$FORWARD" "$HOST"
URL="http://localhost:$LOCAL_PORT"
echo
echo "MuleRadar Live is running on the A100 ($NODE, job $JOB)."
echo "  Live console:  $URL"
echo "  Results page:  $URL/results"
echo "Press Ctrl-C to stop the server."
command -v open >/dev/null && open "$URL"

while ssh "$HOST" "kill -0 $SPID 2>/dev/null"; do sleep 10 & wait $!; done
echo "Server stopped. Last log lines:"; ssh "$HOST" "tail -5 $REMOTE_DIR/live.log"
