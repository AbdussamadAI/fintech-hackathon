#!/usr/bin/env bash
# Push the local repo to the cluster (code only; data lives on the cluster).
# Requires the ControlMaster connection opened by slurm-notebook.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
rsync -az --delete \
  --exclude '.git' --exclude 'data/' --exclude 'node_modules/' --exclude '__pycache__/' \
  --exclude '.ipynb_checkpoints/' --exclude 'bench/results/' --exclude '*.log' \
  "$ROOT/" slurm:fintech-hackathon/
echo "synced $ROOT -> slurm:~/fintech-hackathon"
