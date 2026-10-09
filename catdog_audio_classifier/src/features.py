"""Feature extraction for the cat/dog audio (noise) classifier.

All files are mono 16 kHz WAV, variable length (1–18 s). We convert each clip
into fixed-size log-Mel spectrogram images, which gives:
  * single-channel representation : 1-channel log-mel image
  * multi-channel representation  : 3 "pseudo-RGB" channels stacked as
        [ log-mel | delta(log-mel) | delta-delta(log-mel) ]
    (standard HTK-style dynamic features — captures time-derivative "texture"
     of the noise, exactly what distinguishes purring vs. growling spectra.)

Implementation notes
--------------------
* Pure numpy STFT + mel filterbank (no torchaudio/librosa dependency).
* Deterministic frame padding so every clip yields a (C, n_mels, T_fixed)
  tensor; T is cropped/center-padded to `target_frames`.
* Waveform loader also returns raw samples for a waveform-domain model.
* Caches extracted features in artifacts/features_<mode>.npz per split.
"""
from __future__ import annotations

import os
import wave
import glob
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "..", "Cats and Dogs", "cats_dogs")
ARTIFACTS = os.path.join(ROOT, "artifacts")

SR = 16_000
N_FFT = 1024
HOP = 256
N_MELS = 96
FMIN, FMAX = 50.0, 7600.0
TARGET_FRAMES = 251          # ~4 s window at hop 256/16k ; center-crop/pad


# ---------------------------------------------------------------- waveform I/O
def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1, path
        assert w.getframerate() == SR, path
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x


def list_clips() -> list[tuple[str, int, str]]:
    """Returns (path, label, split). label: 0=cat, 1=dog."""
    out = []
    for sub, lab, split in [("train/cat", 0, "train"), ("train/dog", 1, "train"),
                            ("test/cats", 0, "test"), ("test/test", 1, "test")]:
        for p in sorted(glob.glob(os.path.join(DATA, sub, "*.wav"))):
            out.append((p, lab, split))
    return out


# ------------------------------------------------------------------ DSP blocks
def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l: c = l + 1
        if r == c: r = c + 1
        fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_FB = None
_WIN = np.hanning(N_FFT).astype(np.float32)


def stft_logmel(x: np.ndarray) -> np.ndarray:
    """(samples,) -> (n_mels, T) log-mel spectrogram."""
    global _FB
    if _FB is None:
        _FB = _mel_filterbank(SR, N_FFT, N_MELS, FMIN, FMAX)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP] * _WIN
    mag = np.abs(np.fft.rfft(frames.astype(np.float32), axis=1)).T          # (F, T)
    mel = _FB @ mag                                                          # (M, T)
    return np.log1p(mel * 1e3).astype(np.float32)                           # log-compressed


def _diff_feature(s: np.ndarray, order: int) -> np.ndarray:
    """HTK-style deltas via centered windows (P * delta_t): output keeps the
    input's shape by edge-clipping indices; k-order applied recursively."""
    d = s
    for _ in range(order):
        T = d.shape[1]
        P = 2
        idx = np.clip(np.arange(T)[:, None] + np.arange(-P, P + 1)[None, :], 0, T - 1)
        num = d[:, idx] * np.arange(-P, P + 1)                # (M,T,2P+1)
        den = 2.0 * sum(p * p for p in range(1, P + 1))
        d = (num.sum(axis=2) / den).astype(np.float32)
        d = (d - d.mean()) / (d.std() + 1e-6)
    return d.astype(np.float32)


def make_channels(spec: np.ndarray, mode: str) -> np.ndarray:
    """mode 'single' -> (1,M,T); 'multi' -> (3,M,T) with delta & delta-delta."""
    if mode == "single":
        return spec[None]
    return np.stack([spec, _diff_feature(spec, 1), _diff_feature(spec, 2)])


def fit_length(spec: np.ndarray, target: int = TARGET_FRAMES) -> np.ndarray:
    M, T = spec.shape
    if T >= target:
        s = (T - target) // 2
        return spec[:, s:s + target]
    pad = target - T
    lo, hi = pad // 2, pad - pad // 2
    return np.pad(spec, ((0, 0), (lo, hi)), mode="edge")


def normalize_spec(spec: np.ndarray) -> np.ndarray:
    """Per-file standardization (robust to loudness differences)."""
    return (spec - spec.mean()) / (spec.std() + 1e-6)


# ------------------------------------------------------------------ extraction
def extract_all(mode: str = "multi", use_cache: bool = True) -> dict:
    os.makedirs(ARTIFACTS, exist_ok=True)
    cache = os.path.join(ARTIFACTS, f"features_{mode}.npz")
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    feats, labels, paths, splits = [], [], [], []
    for p, lab, split in list_clips():
        x = read_wav(p)
        spec = normalize_spec(stft_logmel(x))
        spec = fit_length(spec)
        feats.append(make_channels(spec, mode))
        labels.append(lab); paths.append(os.path.relpath(p, DATA)); splits.append(split)

    X = np.stack(feats).astype("float32")            # (N, C, M, T)
    y = np.array(labels, dtype="int64")
    out = {"X": X, "y": y,
           "paths": np.array(paths), "splits": np.array(splits)}
    np.savez_compressed(cache, **out)
    with open(os.path.join(ARTIFACTS, "feature_meta.json"), "w") as f:
        json.dump({"mode": mode, "shape": list(X.shape), "sr": SR,
                   "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS,
                   "target_frames": TARGET_FRAMES}, f, indent=2)
    return out


if __name__ == "__main__":
    for m in ("single", "multi"):
        d = extract_all(m)
        print(m, d["X"].shape, "train:", (d["splits"] == "train").sum(),
              "test:", (d["splits"] == "test").sum(),
              "cat:", (d["y"] == 0).sum(), "dog:", (d["y"] == 1).sum())
