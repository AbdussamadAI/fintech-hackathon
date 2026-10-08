#!/usr/bin/env bash
# Everything that needs N-ATLaS, in order (run on the login node once ~/models/N-ATLaS.Q4_K_M.gguf is complete).
set -uo pipefail
cd "$(dirname "$0")/.."
R=slurm/run.sh
slurm/llm.sh
echo "== sanity"
$R python -u - <<'PY'
import sys, json; sys.path.insert(0, ".")
from pipeline import genai
print(genai.health())
print(json.dumps(genai.extract_one("Abeg, I send 50k since morning, una don debit me but e no reach."), ensure_ascii=False))
print(json.dumps(genai.extract_one("Na tura N20,000 amma bai isa ba, kuma an cire kudin daga asusuna."), ensure_ascii=False))
PY
echo "== rewritten-complaint evaluation + briefs"
$R python -u pipeline/eval_llm.py --data data/synthetic --per-lang 60 --workers 32
echo "== normalise all complaints with N-ATLaS"
$R python -u pipeline/normalize.py --data data/synthetic --workers 32
echo "== topics from N-ATLaS English summaries"
$R python -u pipeline/topics.py --data data/synthetic --source summary
echo "== detection on summary topics"
$R python -u pipeline/detect.py --data data/synthetic --ops gpu --topics data/synthetic/topics_summary.parquet --out data/synthetic/detect_gpu_summary.json
echo AFTER_LLM_DONE
