#!/usr/bin/env bash
# =============================================================================
# handwritten_cnn — CNN for MyHandwrittenDigits (28x28 grayscale, 10 classes).
#
#   Usage:  ./run.sh            (GPU auto-detected by the script itself)
#   Log :   logs/pipeline_<timestamp>.log  (verbose, unbuffered; also console)
#
# Single entry point runs the whole protocol:
#   * ModernConvNet (DoubleConv stages, global avg+max pooling)
#   * augmentation: pad+random-crop, rotation, affine jitter, random erasing,
#     mixup on targets
#   * stratified 5-fold cross-validation (honest estimate; only 40 imgs/class)
#   * final model trained on all data with CV-fixed hyper-parameters
#   * Grad-CAM explanations + t-SNE embedding plot + confusion matrix +
#     training curves -> artifacts/
#
# If a stage fails, send back the log file.
# Requirements: python>=3.9, numpy torch torchvision pillow matplotlib sklearn
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p logs
LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== handwritten_cnn pipeline ==="
echo "started : $(date -Iseconds)"
echo "host    : $(hostname) | python: $($PY -V 2>&1)"
$PY -c "import numpy,torch,torchvision,PIL,sklearn,matplotlib;print('deps OK'); print('cuda:', torch.cuda.is_available())" || {
  echo "MISSING DEPS — pip install numpy torch torchvision pillow matplotlib scikit-learn"; exit 1; }
[ -d ../MyHandwrittenDigits/0 ] || { echo "dataset folder MyHandwrittenDigits/ not found"; exit 1; }

rc=0
echo "--- training + evaluation (src/model.py) ---"
$PY -u src/model.py || rc=1
echo "--- artifacts ---"; ls -l artifacts/ || true
echo "finished: $(date -Iseconds) | exit=$rc"
exit $rc
