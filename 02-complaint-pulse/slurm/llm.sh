#!/usr/bin/env bash
# Start the on-prem model servers inside the running GPU job (run on the login node):
#   N-ATLaS 8B (Q4_K_M) chat on :18081 (8081 is taken by another tenant on server02) with 16 parallel slots;  BGE-M3 embeddings on :8082
# Idempotent: servers already answering /health are left alone.
set -uo pipefail
JOB=${JOB:-$(squeue -h -u "$USER" -t R -n jupyter -o %i | head -1)}
NODE=$(squeue -h -j "$JOB" -o %N)
BIN=$HOME/models/llama.cpp-master/build/bin/llama-server
up() { srun --jobid="$JOB" --overlap curl -s "http://127.0.0.1:$1/health" 2>/dev/null | grep -q '"ok"'; }
start() { # port model extra-args...
  local port=$1 model=$2; shift 2
  if up "$port"; then echo "port $port already serving"; return; fi
  nohup srun --jobid="$JOB" --overlap "$BIN" -m "$HOME/models/$model" --host 0.0.0.0 --port "$port" -ngl 99 "$@" \
    > "$HOME/models/server-$port.log" 2>&1 &
  echo "starting $model on $NODE:$port"
}
start 18081 N-ATLaS.Q4_K_M.gguf -c 65536 --parallel 32 -fa on --jinja   # one process: several processes just contend for the GPU
start 8082 bge-m3-q8_0.gguf --embedding --pooling cls -c 16384 -b 8192 -ub 8192 --parallel 8
for p in 18081 8082; do
  for i in $(seq 1 90); do up $p && { echo "port $p ready"; break; }; sleep 2; done
  up $p || { echo "port $p FAILED:"; tail -5 "$HOME/models/server-$p.log"; }
done
