#!/usr/bin/env bash
# =============================================================================
# diabetes_mlp — MLP for 3-class Diabetes-Risk prediction. One-shot execution.
#
#   Usage:  ./run.sh            (or: PYTHON=python3.11 OMP_NUM_THREADS=4 ./run.sh)
#   Log :   logs/pipeline_<timestamp>.log  (verbose, unbuffered; also on console)
#
# Pipeline:
#   [1/3] src/data.py        – EDA + leakage-audited preprocessing + 70/15/15
#                               stratified splits -> artifacts/{eda_report.json,
#                               pipeline_state.json, arrays.npz}
#   [2/3] src/tune.py        – Stage-1 hyper-parameter search (Phase A arch x
#                               dropout grid, Phase B lr + loss refinement),
#                               selection on validation macro-F1 only.
#   [3/3] src/final_train.py – Stage-2: best config as 3-seed ensemble +
#                               logistic-regression & gradient-boosting
#                               baselines + full test evaluation + plots +
#                               final_report.json.
#
# Resume: outputs checkpoint to artifacts/; re-running skips nothing but is
# deterministic (fixed seeds). If a stage fails, send back the log file.
# Requirements: python>=3.9, numpy pandas scikit-learn torch matplotlib
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
mkdir -p logs
LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== diabetes_mlp pipeline ==="
echo "started : $(date -Iseconds)"
echo "host    : $(hostname) | python: $($PY -V 2>&1)"
$PY -c "import numpy,pandas,sklearn,torch,matplotlib;print('deps OK')" || {
  echo "MISSING DEPS — pip install numpy pandas scikit-learn torch matplotlib"; exit 1; }
[ -f ../diabetes_risk_prediction_dataset.csv ] || { echo "CSV not found next to repo root"; exit 1; }

rc=0
echo "--- [1/3] data preparation + EDA ---"; $PY -u src/data.py        || rc=1
[ $rc -eq 0 ] && { echo "--- [2/3] hyper-parameter search (tune.py) ---"; $PY -u src/tune.py || rc=1; }
[ $rc -eq 0 ] && { echo "--- [3/3] final training + baselines + report ---"; $PY -u src/final_train.py || rc=1; }
echo "--- artifacts ---"; ls -l artifacts/ || true
echo "finished: $(date -Iseconds) | exit=$rc"
exit $rc
