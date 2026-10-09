# PROGRESS — diabetes_mlp

## Status: READY FOR EXECUTION (scripts complete; training deferred to user machine)

| Step | State | Notes |
|---|---|---|
| Data pipeline (`src/data.py`) | ✅ implemented | EDA JSON, leakage audit, fit-on-train transforms, 70/15/15 stratified |
| Model/trainer (`src/model.py`) | ✅ implemented | LN+GELU+dropout blocks, effective-number class weights, focal option, AdamW+warmup+cosine, early stop on val macro-F1 |
| Tuning (`src/tune.py`) | ✅ implemented | Phase A grid (4 archs × 2 dropouts), Phase B (3 lrs × {wCE, focal}) |
| Final stage (`src/final_train.py`) | ✅ implemented | 3-seed ensemble + LogReg/GB baselines + plots + final_report.json |
| `run.sh` | ✅ created | verbose tee'd logging to `logs/pipeline_<ts>.log` |
| Partial local run (before move) | ⚠️ interrupted | 8/10 tune configs finished; best val macro-F1 ≈ 0.8525, test ≈ 0.869 |
| Full end-to-end run | ⏳ pending | execute `./run.sh` on your machine, send back the log |

## To reproduce / continue
```bash
cd diabetes_mlp && ./run.sh
```
Runtime estimate: CPU ~1–3 h (12-config search + 3 ensemble fits + baselines);
GPU dramatically faster (torch code is device-agnostic where relevant).

## Artifacts produced (gitignored)
`artifacts/{eda_report.json, pipeline_state.json, arrays.npz, tuning_results.json,
final_report.json, *.pt, *_proba.npz, confusion_matrix.png, training_curves.png,
per_class_metrics.png, feature_importance.png}`; console mirror in `logs/`.
