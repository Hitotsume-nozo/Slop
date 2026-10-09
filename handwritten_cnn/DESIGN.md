# DESIGN — handwritten_cnn

## 1. Dataset reality check
`../MyHandwrittenDigits/{0..9}` — exactly **40 PNGs per class (400 total)**,
28×28 grayscale (loaded as L). This is a *tiny-data* regime: the dominant risk
is overfitting, so the design priorities are regularization and honest
evaluation, not raw capacity.

## 2. Architecture — ModernConvNet (~500k params, deliberately modest)
* 4 conv stages of `DoubleConv`: [Conv3×3 → BN → ReLU] × 2, channels
  32 → 64 → 128 with MaxPool2 between stages, dropout between stages.
* Head: **global concat(avg-pool, max-pool)** → Linear(256→10). Global pooling
  instead of flatten-FC removes the largest overfitting-prone parameter block
  and makes the net translation-tolerant.
* Kaiming init; BN in train mode only during fitting.

## 3. Regularization stack (the "advanced" part)
| Technique | Where | Why |
|---|---|---|
| Pad-4 + random 28×28 crop | T.ToTensor pipeline | shift tolerance (biggest win on digits) |
| Random rotation ±12°, affine jitter | training TF | writer style variance |
| RandomErasing(p=0.25) | training TF | occlusion robustness, acts like cutout |
| **Mixup** α=0.2 (images AND one-hot targets) | batch level | strongest known regularizer for tiny image sets; smooths decision boundaries |
| Label smoothing 0.05 | CE loss | prevents overconfident memorization |
| AdamW wd=5e-4, cosine LR + warmup, grad clip | optimizer | stable small-batch training |
| Early stopping on fold-val accuracy, best-state restore | per run | no wasted epochs, no last-epoch variance |

## 4. Evaluation protocol (honest numbers for n=400)
1. **Stratified 5-fold CV** (per-class folds) → mean ± std accuracy = the
   headline generalization estimate. Each fold trains its own model from
   scratch with identical hyper-parameters.
2. Fold-0 additionally re-evaluated as a clean holdout (`results.holdout`).
3. **Final model** trained on all 400 images with CV-fixed hyper-parameters;
   its apparent accuracy is reported separately (`final_model_train_acc`) and
   explicitly labeled optimistic — never conflated with generalization.
4. Full per-class precision/recall/F1 + confusion matrix + Grad-CAM grid (one
   explanation per class) + t-SNE of penultimate embeddings.
Everything serialized to `artifacts/results.json`.

## 5. Determinism
Seeds fixed for python/numpy/torch; fold construction deterministic
(`stratified_kfold(seed=0)`); DataLoader shuffled with seeded generator.

## 6. Limitations / follow-ups
* 40/class means fold-val accuracy has ±~5% noise; treat CV mean as interval,
  not point estimate.
* Follow-ups: test-time augmentation averaging, snapshot ensembling, ProtoNet
  metric baseline (interesting alternative at this data scale).
