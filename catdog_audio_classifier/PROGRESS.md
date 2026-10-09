# PROGRESS — catdog_audio_classifier

## Status: READY FOR EXECUTION (code complete; training deferred to user machine)

| Step | State | Notes |
|---|---|---|
| Feature extraction (`src/features.py`) | ✅ | log-mel single + 3-ch dynamic (Δ, ΔΔ); caching verified locally (features npz built for 277 clips) |
| Dataset quirks handling | ✅ | mono-16k resample path, folder-name inconsistencies normalized, length crop/pad to 251 frames |
| SpectroCNN (GroupNorm/GELU/global pool) | ✅ | works for 1- and 3-channel input through same code path |
| RawWaveNet control model | ✅ | E4 |
| Separation loss (weighted CE + triplet margin) | ✅ | λ=0.1, m=2.0; E3 ablation λ=0 |
| 5-fold CV inside train + final fit + test eval | ✅ | `train_one.py` |
| Per-run t-SNE of test embeddings | ✅ | `artifacts/tsne_<run>.png` |
| Incremental crash-safe checkpointing | ✅ | `artifacts/results.json` written after each experiment |
| OOM issue diagnosis & revert | ✅ | old in-process driver removed; subprocess-per-experiment scheme kept; `runner.py` retained as alternative |
| `run.sh` | ✅ | verbose tee'd logging, auto-retry ×4 per experiment, SKIP_PREP resume flag |
| End-to-end run | ⏳ pending | E1 had started locally but was stopped before any fold finished; execute `./run.sh` on your machine and send back the log |

Runtime estimate: ~5–20 min per experiment on CPU (features precomputed;
E4 raw-waveform is the slowest). Total < 1–2 h.

## Artifacts produced (gitignored)
`artifacts/{features_single.npz, features_multi.npz, feature_meta.json,
results.json, <RUN>.pt, emb_<RUN>.npz, tsne_<RUN>.png}`; console mirror in `logs/`.
