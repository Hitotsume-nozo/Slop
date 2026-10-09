# handwritten_cnn — CNN for MyHandwrittenDigits

Small-data (400 images: 10 classes × 40 samples, 28×28 grayscale) handwritten
digit classifier with an advanced-but-appropriate training recipe.

## Quick start
```bash
./run.sh          # full pipeline; verbose log in logs/pipeline_<ts>.log
```
Requires: `pip install numpy torch torchvision pillow matplotlib scikit-learn`
(GPU auto-detected; runs fine on CPU.)

## Layout
```
run.sh              one-shot execution script (tee'd verbose logging)
src/model.py        dataset, ModernConvNet, mixup+augmentation, CV protocol,
                    Grad-CAM + t-SNE explanations, plots, results.json
DESIGN.md           design document (architecture & protocol rationale)
PROGRESS.md         running progress/status log
artifacts/          generated outputs (gitignored)
```
