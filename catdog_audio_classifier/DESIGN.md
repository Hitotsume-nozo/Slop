# DESIGN — catdog_audio_classifier

## 1. Task & data
Binary separation of **cat vs dog vocalization noise** from
`../Cats and Dogs/cats_dogs/`: train ≈ 210 clips (125 cat / 85 dog — imbalanced),
test ≈ 67 clips. Files are mono 16 kHz WAVs of variable length (1–18 s).
Note: the dataset folder names are inconsistent (`train/cat`, `test/cats`);
`features.list_clips()` normalizes them.

## 2. Audio front-end (single vs multi channel)
* STFT (n_fft=1024, hop=256, Hann) → 96-band mel filterbank → log compression.
* Length normalization: center-crop/random-pad to 251 frames (~4 s window).
* Per-clip z-score normalization of the spectrogram.
* **Single-channel**: the log-mel image itself (1×96×251).
* **Multi-channel**: 3 pseudo-RGB channels `[mel | Δmel | ΔΔmel]`
  (HTK-style dynamic features) — captures the time-derivative *texture*
  (purr periodicity vs bark transients) that static spectra miss.
* Features extracted once, cached as npz (`artifacts/features_{single,multi}.npz`).

## 3. Architectures
**SpectroCNN** (used for E1–E3): 4 blocks of
Conv3×3 → GroupNorm → GELU → Conv3×3 → GroupNorm → GELU → MaxPool, base width 16
(→32→64→128), dropout, global concat(avg,max) pooling → linear head.
*GroupNorm* chosen over BatchNorm because (a) tiny batches/210 samples make BN
statistics unstable, and (b) the identical code path handles 1- and 3-channel
inputs. Training-time **SpecAugment-lite**: frequency masking, time masking,
random time-shift.

**RawWaveNet** (E4): strided-conv frontend on padded waveform (no mel), as a
front-end ablation/domain check.

## 4. Separation objective
Combined loss = weighted cross-entropy (class imbalance 125:85) **+ λ·L_sep**,
where L_sep is a batch-mined **triplet-style margin loss** on the penultimate
embedding: pulls same-class embeddings together, pushes cat/dog prototypes apart
by margin m=2.0 (λ=0.1). The embedding space is therefore explicitly optimized
for *separation*, not just discrimination — visualized per run with t-SNE of
test embeddings. E3 ablates λ→0 to measure its contribution.

## 5. Evaluation protocol
* Official split respected: model selection never sees `test/`.
* Because train is tiny: **stratified 5-fold CV inside train** (mean val acc)
  for selection; final fit uses an 80/20 stratified holdout inside train for
  checkpointing; then ONE pass on the official test set reporting accuracy,
  F1, ROC-AUC, confusion matrix, and per-run t-SNE.
* AdamW lr=1e-3 wd=1e-4, cosine schedule, grad clip, 60 epochs (40 for E4),
  deterministic seeds. Results checkpoint incrementally to
  `artifacts/results.json` (crash-safe).

## 6. Execution-safety architecture
The original all-in-process driver was OOM-killed in constrained environments
(~2 GB RSS containers). It was reverted/re-architected so that **each
experiment runs in its own subprocess** (`train_one.py` via `run.sh` loop or
`runner.py`): memory returns to the OS between runs, finished experiments are
never re-run (resume by presence in results.json), up to 4 automatic retries
per experiment, everything echoed into a tee'd verbose log.

## 7. Limitations / follow-ups
* 67 test clips ⇒ ±~6% CI on test accuracy; trust CV mean more.
* Follow-ups: VGGish/PANNs transfer features, harmonic-constant-cepstrum
  (HCC) front-end for pitch-related cues, Grad-CAM-on-spectrogram inspection.
