#!/usr/bin/env bash
# Start Jupyter Lab on a Slurm compute node and tunnel it to this machine.
# Ctrl-C closes the tunnel and cancels the job.
set -euo pipefail

HOST=slurm            # ssh alias from ~/.ssh/config
PARTITION=interactive # interactive (2h max) or RTX3090 / A4500 / A5000 / A6000 / A100 (2 days)
GPUS=1
CPUS=8
MEM=32G
TIME=02:00:00
LOCAL_PORT=8888
ROOT_DIR='$HOME'      # expanded on the cluster

usage() {
  cat <<EOF
Usage: $0 [-p partition] [-g gpus] [-c cpus] [-m mem] [-t time] [-l local_port] [-d remote_dir]
Defaults: -p $PARTITION -g $GPUS -c $CPUS -m $MEM -t $TIME -l $LOCAL_PORT -d \$HOME
EOF
  exit 1
}

while getopts "p:g:c:m:t:l:d:h" opt; do
  case $opt in
    p) PARTITION=$OPTARG ;;
    g) GPUS=$OPTARG ;;
    c) CPUS=$OPTARG ;;
    m) MEM=$OPTARG ;;
    t) TIME=$OPTARG ;;
    l) LOCAL_PORT=$OPTARG ;;
    d) ROOT_DIR=$OPTARG ;;
    *) usage ;;
  esac
done

# Pick a free local port
while lsof -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; do
  LOCAL_PORT=$((LOCAL_PORT + 1))
done

GRES=""
[[ "$GPUS" -gt 0 ]] && GRES="#SBATCH --gres=gpu:$GPUS"

# The cluster only allows password login, so everything goes through one
# ControlMaster connection (see ~/.ssh/config). Open it if it isn't up yet.
if ! ssh -O check "$HOST" 2>/dev/null; then
  echo "Opening SSH connection to $HOST (enter your cluster password)..."
  ssh -fN "$HOST"
fi

echo "Submitting Jupyter job (partition=$PARTITION gpus=$GPUS cpus=$CPUS mem=$MEM time=$TIME)..."
JOBID=$(ssh "$HOST" "mkdir -p ~/.slurm-jupyter && chmod 700 ~/.slurm-jupyter && sbatch --parsable" <<EOF
#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=$PARTITION
$GRES
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=.slurm-jupyter/%j.log

umask 077
export PATH=\$HOME/bin:/opt/anaconda3/bin:\$PATH  # ~/bin has nsys (not installed on every node)
PORT=\$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
TOKEN=\$(python3 -c 'import secrets; print(secrets.token_hex(24))')
echo "\$(hostname) \$PORT \$TOKEN" > \$HOME/.slurm-jupyter/\$SLURM_JOB_ID.info
exec jupyter lab --no-browser --ip=0.0.0.0 --port=\$PORT \
  --IdentityProvider.token=\$TOKEN --ServerApp.root_dir=$ROOT_DIR
EOF
)
JOBID=${JOBID%%;*}
echo "Job $JOBID submitted."

FORWARD=""
cleanup() {
  trap - EXIT INT TERM
  [[ -n "$FORWARD" ]] && ssh -O cancel -L "$FORWARD" "$HOST" 2>/dev/null || true
  echo
  echo "Cancelling job $JOBID..."
  ssh "$HOST" "scancel $JOBID; rm -f ~/.slurm-jupyter/$JOBID.info" || true
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

# Wait for the job to start and Jupyter to come up
while :; do
  STATE=$(ssh "$HOST" "squeue -h -j $JOBID -o '%T %R'" || true)
  [[ -z "$STATE" ]] && { echo "Job ended unexpectedly. Log:"; ssh "$HOST" "cat ~/.slurm-jupyter/$JOBID.log"; exit 1; }
  if [[ "$STATE" == RUNNING* ]]; then
    INFO=$(ssh "$HOST" "cat ~/.slurm-jupyter/$JOBID.info 2>/dev/null; grep -q 'is running at' ~/.slurm-jupyter/$JOBID.log 2>/dev/null && echo READY" || true)
    [[ "$INFO" == *READY* ]] && break
  fi
  printf '\r  %-60s' "Waiting: $STATE"
  sleep 3
done
read -r NODE PORT TOKEN <<<"$(head -n1 <<<"$INFO")"
echo

FORWARD="$LOCAL_PORT:$NODE:$PORT"
ssh -O forward -L "$FORWARD" "$HOST"

URL="http://localhost:$LOCAL_PORT/lab?token=$TOKEN"
cat <<EOF

Jupyter Lab is running on $NODE (job $JOBID).

  Browser:  $URL
  VS Code:  "Select Kernel" -> "Existing Jupyter Server" -> http://localhost:$LOCAL_PORT/?token=$TOKEN

Press Ctrl-C to stop (cancels the Slurm job).
EOF
command -v open >/dev/null && open "$URL"

# Stay up while the job runs; exits (and cleans up) when it hits its time limit
while [[ -n "$(ssh "$HOST" "squeue -h -j $JOBID" 2>/dev/null)" ]]; do
  sleep 30 & wait $!  # backgrounded so Ctrl-C is handled immediately
done
echo "Job $JOBID has ended."
