#!/usr/bin/env bash
# Run a command inside the running Jupyter job's allocation on the GPU node.
# Usage (on the login node): slurm/run.sh python scripts/train.py --name HI-Small
set -euo pipefail
JOB=${JOB:-$(squeue -h -u "$USER" -n jupyter -o %i | head -1)}
cd "$(dirname "$0")/.."
exec srun --jobid="$JOB" --overlap env CUDA_PATH="$HOME/cuda_shim" PATH="$HOME/envs/mule13/bin:$PATH" \
  PYTHONWARNINGS=ignore "$@"
