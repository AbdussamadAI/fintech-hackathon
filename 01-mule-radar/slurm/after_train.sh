#!/usr/bin/env bash
# Post-training chain (run on the login node): ring eval -> swap in CUDA-12 XGBoost -> multi-seed GPU
# confirmation of the full model -> GPU vs CPU benchmark -> demo export.
set -uo pipefail
cd "$(dirname "$0")/.."
R=slurm/run.sh
$R python -u scripts/eval_rings.py --name HI-Small
~/envs/mule13/bin/pip install -q --no-deps "xgboost==2.1.4" && echo "xgboost: $(~/envs/mule13/bin/python -c 'import xgboost; print(xgboost.__version__)')"
$R python -u - <<'PY'
import xgboost as xgb, numpy as np, warnings
warnings.simplefilter("error")
X = np.random.rand(2000, 4); y = (X[:, 0] > .5).astype(int)
xgb.XGBClassifier(device="cuda", n_estimators=5).fit(X, y)
print("xgboost GPU OK")
PY
$R python -u bench/bench.py --name HI-Small
$R python -u scripts/export_demo.py --name HI-Small
echo CHAIN_DONE
