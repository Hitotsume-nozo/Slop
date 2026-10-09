# diabetes_mlp — MLP for Diabetes Risk Prediction (3-class)

Multilayer perceptron on `../diabetes_risk_prediction_dataset.csv` (~50k rows,
42 columns) predicting `Diabetes_Risk` ∈ {Low, Moderate, High} (severe class
imbalance: Low ≈ 1%).

## Quick start
```bash
./run.sh          # full pipeline; verbose log in logs/pipeline_<ts>.log
```
Requires: `pip install numpy pandas scikit-learn torch matplotlib`

## Layout
```
run.sh                 one-shot execution script (tee'd verbose logging)
src/data.py            EDA + leakage-audited preprocessing + stratified splits
src/model.py           MLP architecture, weighted-CE/focal loss, AdamW+cosine,
                       warmup, early stopping on val macro-F1
src/tune.py            Stage-1 hyper-parameter search (Phase A grid, Phase B refine)
src/final_train.py     Stage-2 final 3-seed ensemble + LR/GB baselines + report
DESIGN.md              full design document (data audit, model, protocol)
PROGRESS.md            running progress/status log
artifacts/             generated outputs (gitignored)
```

See `DESIGN.md` for the rigorous rationale and `PROGRESS.md` for status.
