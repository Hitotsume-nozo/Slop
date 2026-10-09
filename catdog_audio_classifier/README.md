# catdog_audio_classifier — Cat vs Dog vocalization ("noise") separation classifier

A separation-oriented CNN that recognizes the *sound texture* of cats vs dogs,
supporting **single-channel** (log-mel spectrogram) and **multi-channel**
(mel + Δ + ΔΔ dynamic features) inputs, plus a raw-waveform control model.

## Quick start
```bash
./run.sh                 # full pipeline; verbose log in logs/pipeline_<ts>.log
SKIP_PREP=1 ./run.sh     # resume after interruption (features already cached)
```
Requires: `pip install numpy scipy pandas librosa soundfile torch scikit-learn matplotlib`

## Layout
```
run.sh                    one-shot execution script (tee'd verbose logging)
src/features.py           WAV -> fixed-size log-mel (single) / 3-ch (multi)
src/model.py              SpectroCNN (GroupNorm+GELU, SpecAugment-ready) and
                          RawWaveNet; CE + triplet-style separation loss
src/train.py              shared training engine + 5-fold CV protocol
src/train_one.py          runs ONE experiment, checkpoints to results.json
src/runner.py             optional sequential runner (subprocess-per-experiment)
DESIGN.md                 design document (audio front-end, losses, protocol)
PROGRESS.md               running progress/status log
artifacts/                generated outputs (gitignored)
```

## Experiment matrix
| Run | Input | Channels | Separation loss | Purpose |
|---|---|---|---|---|
| E1_single | log-mel | 1 | yes | baseline single-channel |
| E2_multi | mel+Δ+ΔΔ | 3 | yes | does temporal dynamics help? |
| E3_multi_noSepLoss | mel+Δ+ΔΔ | 3 | no | ablation of the separation term |
| E4_waveform | raw padded wave | 1 | n/a | domain check without mel front-end |
