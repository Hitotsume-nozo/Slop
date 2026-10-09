#!/usr/bin/env bash
# =============================================================================
# catdog_audio_classifier — separation classifier for cat vs dog vocalization
# noise, SINGLE-channel and MULTI-channel (mel + delta + delta-delta) inputs.
#
#   Usage:  ./run.sh                 full pipeline
#           SKIP_PREP=1 ./run.sh     resume after an interrupted run
#   Log :   logs/pipeline_<timestamp>.log  (verbose, unbuffered; also console)
#
# Experiments (each in its own subprocess = memory-safe, crash-resumable):
#   E1_single          SpectroCNN, 1-channel log-mel
#   E2_multi           SpectroCNN, 3-channel (mel+Δ+ΔΔ) + separation loss
#   E3_multi_noSepLoss ablation: multi-channel WITHOUT separation loss
#   E4_waveform        RawWaveNet on padded waveform (no mel front-end)
# Each run does 5-fold CV inside train/, then final fit + test metrics
# (acc/F1/ROC-AUC/confusion) + per-run t-SNE of test embeddings.
# Results checkpoint incrementally to artifacts/results.json.
#
# Requirements: python>=3.9, numpy scipy pandas librosa soundfile torch
#               scikit-learn matplotlib
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
mkdir -p logs
LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== catdog_audio_classifier pipeline ==="
echo "started : $(date -Iseconds)"
echo "host    : $(hostname) | python: $($PY -V 2>&1)"
$PY -c "import numpy,scipy,pandas,librosa,soundfile,torch,sklearn,matplotlib;print('deps OK')" || {
  echo "MISSING DEPS — pip install numpy scipy pandas librosa soundfile torch scikit-learn matplotlib"; exit 1; }

rc=0
if [ "${SKIP_PREP:-0}" != "1" ]; then
  echo "--- [stage] feature extraction (log-mel single/multi channels) ---"
  $PY -u -c "import sys;sys.path.insert(0,'src');from features import extract_all;extract_all('single');extract_all('multi');print('features cached')" || rc=1
fi
for e in E1_single E2_multi E3_multi_noSepLoss E4_waveform; do
  [ $rc -ne 0 ] && break
  echo "--- [$e] 5-fold CV + final fit + test eval ---"
  attempt=1
  until $PY -u src/train_one.py "$e"; do
    if [ $attempt -ge 4 ]; then echo "[$e] FAILED after $attempt attempts"; rc=1; break; fi
    attempt=$((attempt+1)); echo "[$e] crashed (likely OOM); retry $attempt/4 in 5s"; sleep 5
  done
done
[ $rc -eq 0 ] && { echo "--- summary ---"; cat artifacts/results.json; }
echo "finished: $(date -Iseconds) | exit=$rc"
exit $rc
