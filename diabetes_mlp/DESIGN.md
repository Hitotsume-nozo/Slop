# DESIGN — diabetes_mlp

## 1. Problem & dataset
* Task: 3-class classification of `Diabetes_Risk` (Low/Moderate/High), ordinal.
* Data: ~50,000 × 42 mixed-type clinical/lifestyle features.
* Key facts found by EDA (`src/data.py` writes `artifacts/eda_report.json`):
  - Missing values scattered across numeric columns → median imputation.
  - Target is severely imbalanced (Low is a tiny minority) → macro-F1 is the
    primary metric, never plain accuracy.
  - Some string columns contain stray non-numeric junk → coerced to NaN then
    imputed.

## 2. Leakage audit (column policy)
| Column | Decision | Reason |
|---|---|---|
| `Patient_ID` | DROP | surrogate key |
| `Country` | DROP | 25 cats, no plausible causal signal, adds variance |
| `Diabetes_Risk_Score` | **DROP (leakage)** | near-deterministic proxy of the label |
| `AI_Health_Recommendation` | **DROP (leakage)** | text generated *from* the risk label |
| 16 low-cardinality strings | one-hot (+ NaN indicator column) | categorical |
| remaining numerics | standardize | continuous/integer features |

## 3. Preprocessing protocol
Fit-on-TRAIN-only statistics (medians, clipping bounds at 0.1%/99.9% quantiles,
mean/std). One-hot vocabulary frozen from train; val/test reindexed. All state
serialized to `artifacts/pipeline_state.json` so inference batches reproduce it
exactly. Deterministic stratified split 70/15/15 (`seed=42`), materialized once
into `artifacts/arrays.npz`. The test set is touched only for final reporting.

## 4. Model
Configurable-depth MLP of pre-norm blocks:
`Linear → LayerNorm → GELU → Dropout`, Kaiming init, linear head over 3 logits.
Families searched: small [64,32], med [128,64,32], wide [256,128,64],
deep [256,128,64,32].

## 5. Loss for imbalance
Weighted cross-entropy with **effective-number** weights (Cui et al. 2019,
β=0.999), clipped to [0.2, 5.0] for stability; optional focal variant
(γ=2) tested in Phase B; label smoothing 0.02.

## 6. Optimization
AdamW (wd=1e-4), lr ∈ {1e-3, 2e-3, 4e-3}, cosine annealing with 5-epoch linear
warmup, grad-norm clip 5.0, batch 512, max 120 epochs, early stop patience 15
on **validation macro-F1**, best-state checkpoint restore. Full seeding
(numpy + torch) for reproducibility.

## 7. Experiment protocol (two-stage, honest evaluation)
1. **Stage 1 `tune.py`**: Phase A = arch × dropout grid at fixed lr; Phase B =
   lr sweep + loss variant around the Phase-A winner. Selection strictly on
   validation macro-F1 (test never used for selection).
2. **Stage 2 `final_train.py`**: winning config trained as a **3-seed ensemble**
   (probability averaging; reduces variance, helps the rare class), plus
   baselines on identical splits — multinomial Logistic Regression and
   Gradient Boosting — to verify the MLP earns its complexity.
3. Final report: accuracy, macro/weighted F1, per-class P/R/F1, confusion
   matrix, ROC-AUC (OvR), permutation feature importance, calibration spot
   checks; plots + `final_report.json` in `artifacts/`.

## 8. Known interim results (partial run before relocation, seed 42)
Phase A/B partial: best val macro-F1 ≈ **0.8525** (configs `A_wide_dp30`,
`B_lr0.002_wce`) with test macro-F1 ≈ 0.869 — to be re-confirmed end-to-end by
`./run.sh` on your machine.

## 9. Risks / limitations
* Ordinal structure (Low<Moderate<High) not exploited (plain CE); a CORAL /
  ordinal-loss variant is a documented follow-up.
* One-hot on `Work_Type`/`Residence_Type` may be near-noise; ablation trivial
  via `DROP_COLS`.
