#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}

# 1. Utilize all available CPU threads for feature extraction and BLAS math
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export OMP_NUM_THREADS="$CORES" \
       MKL_NUM_THREADS="$CORES" \
       OPENBLAS_NUM_THREADS="$CORES" \
       NUMEXPR_NUM_THREADS="$CORES"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p logs artifacts
LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== catdog_audio_classifier pipeline ==="
echo "started : $(date -Iseconds)"
echo "host    : $(hostname) | python: $($PY -V 2>&1)"

# 2. Strict check: confirm CUDA is enabled in PyTorch
$PY -c "
import torch
assert torch.cuda.is_available(), 'CRITICAL: CUDA not detected by PyTorch! Check nvidia-driver and torch CUDA build.'
print(f'CUDA OK: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f}MB VRAM)')
" || exit 1

rc=0
if [ "${SKIP_PREP:-0}" != "1" ]; then
  echo "--- [stage] feature extraction (parallel CPU) ---"
  $PY -u -c "
import sys; sys.path.insert(0,'src')
from features import extract_all
extract_all('single')
extract_all('multi')
print('features cached')
" || rc=1
fi

for e in E1_single E2_multi E3_multi_noSepLoss E4_waveform; do
  [ $rc -ne 0 ] && break
  echo "--- [$e] 5-fold CV + final fit + test eval ---"
  attempt=1
  until $PY -u src/train_one.py "$e"; do
    if [ $attempt -ge 3 ]; then
      echo "[$e] FAILED after $attempt attempts"; rc=1; break;
    fi
    attempt=$((attempt+1))
    echo "[$e] crashed; retrying ($attempt/3) in 3s..."
    sleep 3
  done
done

[ $rc -eq 0 ] && { echo "--- summary ---"; cat artifacts/results.json; }
echo "finished: $(date -Iseconds) | exit=$rc"
exit $rc
