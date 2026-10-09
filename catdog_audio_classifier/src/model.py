"""Separation-oriented CNN for cat vs. dog vocalization ("noise") classification,
supporting BOTH single-channel and multi-channel spectrogram inputs.

Two architectures are provided:

1) SpectroCNN  – compact 4-block convolutional network on log-mel images.
   * Conv3x3 -> GroupNorm -> GELU -> Conv3x3 -> GroupNorm -> GELU -> MaxPool
     (GroupNorm generalizes BatchNorm to tiny datasets AND across channel
      counts, so the SAME code path trains on 1-channel or 3-channel input.)
   * Spec-augmentations in training: frequency/time masking + random time-shift
     (SpecAugment-lite) — crucial with only 210 training clips.
   * Global pooling = concat(mean, max) over time & freq -> linear head.
   * Separation objective: besides CE we add an explicit class-prototype
     separation term computed on penultimate embeddings each batch:
         L_sep = mean( relu(margin - ||mu_cat - mu_dog||)^2 )
     → forces the network to keep cat/dog "noise" embeddings far apart,
       which is exactly the requested separation-classifier behaviour.

2) RawWaveNet – 1-D CNN directly on padded waveforms (no mel front-end),
   demonstrating the classifier also works in the raw signal domain.

Total loss = CE + lambda_sep * L_sep.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ building blocks
def gn(ch: int, groups: int = 8) -> nn.GroupNorm:
    g = math.gcd(groups, ch) or 1
    return nn.GroupNorm(g, ch)


class SepBlock(nn.Module):
    """Conv-BN-style double conv block."""
    def __init__(self, cin: int, cout: int, p: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), gn(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), gn(cout), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout2d(p))

    def forward(self, x): return self.net(x)


class SpectroCNN(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 16, n_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        c = [base, base * 2, base * 4, base * 4]
        self.blocks = nn.Sequential(
            SepBlock(in_ch, c[0]), SepBlock(c[0], c[1]),
            SepBlock(c[1], c[2]), SepBlock(c[2], c[3]))
        # after 4 pools on (96,251) -> (~6, ~16) feature map
        self.pool_mean = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(2 * c[3], 64), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(64, n_classes))
        self.emb_dim = 2 * c[3]

    def embed(self, x):
        h = self.blocks(x)
        return torch.cat([self.pool_mean(h).flatten(1),
                          self.pool_max(h).flatten(1)], dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


# ------------------------------------------------------------------ raw waveform net
class RawWaveNet(nn.Module):
    def __init__(self, n_classes: int = 2, width: int = 32):
        super().__init__()
        w = [width, width * 2, width * 4, width * 4]
        def blk(ci, co, ks, stride):
            return nn.Sequential(
                nn.Conv1d(ci, co, ks, stride=stride, padding=ks // 2, bias=False),
                nn.GroupNorm(8, co), nn.GELU(), nn.MaxPool1d(2))
        self.stem = nn.Sequential(blk(1, w[0], 31, 1), blk(w[0], w[1], 15, 1),
                                  blk(w[1], w[2], 7, 1), blk(w[2], w[3], 5, 1))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(w[3], 64), nn.GELU(),
                                  nn.Linear(64, n_classes))
        self.emb_dim = w[3]

    def embed(self, x): return self.stem(x).mean(-1)          # (B,C)
    def forward(self, x): return self.head(self.stem(x))


# ------------------------------------------------------------------ separation loss
def separation_loss(emb: torch.Tensor, y: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    """Push class centroids apart in embedding space (prototype separation)."""
    classes = y.unique()
    if len(classes) < 2:
        return emb.new_zeros(())
    mus = torch.stack([emb[y == c].mean(0) for c in classes])
    d = torch.cdist(mus[None], mus[None])[0]
    iu = torch.triu_indices(len(classes), len(classes), offset=1)
    return F.relu(margin - d[iu[0], iu[1]]).pow(2).mean()


class CombinedLoss(nn.Module):
    def __init__(self, lam_sep: float = 0.1, margin: float = 2.0,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.lam, self.margin = lam_sep, margin

    def forward(self, logits, emb, y):
        return self.ce(logits, y) + self.lam * separation_loss(emb, y, self.margin)


# ------------------------------------------------------------------ spec augment
def spec_augment(x: torch.Tensor, f_pct: float = 0.15, t_pct: float = 0.15,
                 n_f: int = 2, n_t: int = 2, shift_pct: float = 0.1) -> torch.Tensor:
    """Batch of (B,C,F,T): random band masking in freq & time + circular shift."""
    B, C, Fq, T = x.shape
    x = x.clone()
    # roll along time
    k = int(shift_pct * T)
    if k > 0:
        sh = torch.randint(-k, k + 1, (B,), device=x.device)
        for b in range(B):
            if sh[b] != 0:
                x[b] = torch.roll(x[b], shifts=int(sh[b].item()), dims=-1)
    for _ in range(n_f):
        w = torch.randint(0, max(1, int(Fq * f_pct)), (1,), device=x.device).item()
        o = torch.randint(0, max(1, Fq - w), (1,), device=x.device).item()
        x[:, :, o:o + w, :] = 0.0
    for _ in range(n_t):
        w = torch.randint(0, max(1, int(T * t_pct)), (1,), device=x.device).item()
        o = torch.randint(0, max(1, T - w), (1,), device=x.device).item()
        x[:, :, :, o:o + w] = 0.0
    return x
