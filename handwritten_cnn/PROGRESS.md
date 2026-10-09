# PROGRESS — handwritten_cnn

## Status: READY FOR EXECUTION (code complete; training deferred to user machine)

| Step | State | Notes |
|---|---|---|
| Dataset loader (folder-per-class, indexed subsets) | ✅ | `DigitDataset` |
| ModernConvNet (DoubleConv stages, avg+max global pool) | ✅ | ~500k params |
| Augmentation: crop/rotate/affine/RandomErasing + Mixup | ✅ | mixup applied on batch |
| Trainer (AdamW, warmup+cosine, label smoothing, early stop) | ✅ | best-state restore |
| Stratified 5-fold CV + holdout + final-full-data model | ✅ | `main()` |
| Grad-CAM per-class grid + t-SNE embeddings | ✅ | bugfixed this session (CV-history collection, explanation steps) |
| Plots (training curves, confusion) + results.json | ✅ | artifacts/ |
| Duplicate `__main__` guard bug | ✅ fixed | double `main()` call removed |
| Missing `torchvision` dependency (local sandbox) | ✅ installed | your machine: see requirements in README |
| `run.sh` | ✅ created | verbose tee'd logging |
| End-to-end run | ⏳ pending | execute `./run.sh`, send back log |

Runtime estimate: CPU-only 5-fold + final ≈ 30–90 min on a laptop; GPU ≈ minutes.

## Artifacts produced (gitignored)
`artifacts/{results.json, final_cnn.pt, training_curves.png, confusion_matrix.png,
gradcam_grid.png, tsne_embeddings.png}`; console mirror in `logs/`.
