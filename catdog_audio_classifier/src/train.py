"""Training / evaluation driver for the cat-vs-dog audio separation classifier."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_all, read_wav, list_clips, ARTIFACTS, SR  # noqa: E402
from model import SpectroCNN, RawWaveNet, CombinedLoss, separation_loss, spec_augment  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@dataclass
class Cfg:
    run: str = "E1_single"
    input: str = "multi"  # 'single' | 'multi' | 'wave'
    arch: str = "spectro"  # 'spectro' | 'wave'
    base: int = 16
    dropout: float = 0.3
    lam_sep: float = 0.1  # 0 -> ablation without separation loss
    margin: float = 2.0
    lr: float = 1e-3
    wd: float = 1e-4
    batch: int = 32  # Default batch size
    epochs: int = 60
    seed: int = 42
    augment: bool = True


# ------------------------------------------------------------------ wave dataset
def load_waveforms() -> np.ndarray:
    """Fixed-length raw waveforms: center-crop/pad to 4 s @16k."""
    L = 4 * SR
    out = []
    for p, lab, split in list_clips():
        x = read_wav(p)
        if len(x) >= L:
            s = (len(x) - L) // 2
            x = x[s : s + L]
        else:
            pad = L - len(x)
            x = np.pad(x, (pad // 2, pad - pad // 2))
        out.append(x)
    return np.stack(out).astype("float32")[:, None, :]  # (N,1,L)


# ------------------------------------------------------------------ train engine
def make_model(cfg: Cfg):
    if cfg.arch == "spectro":
        in_ch = 1 if cfg.input == "single" else 3
        return SpectroCNN(in_ch=in_ch, base=cfg.base, dropout=cfg.dropout)
    return RawWaveNet(width=cfg.base)


def get_data(cfg: Cfg):
    if cfg.arch == "spectro":
        d = extract_all(cfg.input)
        Xtr = d["X"][d["splits"] == "train"]
        ytr = d["y"][d["splits"] == "train"]
        Xte = d["X"][d["splits"] == "test"]
        yte = d["y"][d["splits"] == "test"]
    else:
        X = load_waveforms()
        d = extract_all("single")
        y, sp = d["y"], d["splits"]
        Xtr, ytr = X[sp == "train"], y[sp == "train"]
        Xte, yte = X[sp == "test"], y[sp == "test"]
    return Xtr, ytr, Xte, yte


@torch.no_grad()
def predict(model, X, bs=32):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i : i + bs], device=DEVICE, dtype=torch.float32)
        outs.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def get_embeddings(model, X, bs=32):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i : i + bs], device=DEVICE, dtype=torch.float32)
        outs.append(model.embed(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def fit_once(cfg: Cfg, Xtr, ytr, Xva, yva, verbose=False):
    set_seed(cfg.seed)
    model = make_model(cfg).to(DEVICE)

    counts = np.bincount(ytr, minlength=2).astype("float64")
    w = torch.tensor(
        (counts.sum() / (2 * np.maximum(counts, 1))), dtype=torch.float32, device=DEVICE
    )
    ce = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # Keep dataset on CPU; stream mini-batches on demand
    Xt = torch.from_numpy(Xtr)
    Yt = torch.from_numpy(ytr)
    has_val = Xva is not None and len(Xva) > 0

    steps = max(1, math.ceil(len(Xt) / cfg.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps
    )

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    best_acc, best_state, hist = -1.0, None, []

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot = 0.0
        perm = torch.randperm(len(Xt), generator=gen)

        for i in range(0, len(Xt), cfg.batch):
            idx = perm[i : i + cfg.batch]
            # Stream batch to GPU
            xb = Xt[idx].to(DEVICE, non_blocking=True)
            yb = Yt[idx].to(DEVICE, non_blocking=True)

            if cfg.augment and cfg.arch == "spectro":
                xb = spec_augment(xb)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if cfg.arch == "spectro":
                    emb = model.embed(xb)
                    logits = model.head(emb)
                else:
                    logits = model(xb)
                    emb = model.embed(xb)

                loss = ce(logits, yb)
                if cfg.lam_sep > 0:
                    loss = loss + cfg.lam_sep * separation_loss(emb, yb, cfg.margin)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            tot += float(loss.item())

        # Safe batched validation in torch.no_grad()
        if has_val:
            model.eval()
            with torch.no_grad():
                val_preds = predict(model, Xva, bs=cfg.batch)
                acc = float((val_preds == yva).mean())
        else:
            acc = None

        hist.append({"epoch": ep, "loss": tot / max(steps, 1), "val_acc": acc})

        # Save checkpoint to CPU RAM
        if has_val and acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and ep % 20 == 0:
            acc_str = f"{acc:.3f}" if acc is not None else "N/A"
            print(f"   ep{ep:02d} loss={tot / steps:.3f} val_acc={acc_str}")

    if has_val and best_state is not None:
        model.load_state_dict(best_state)

    return model, hist, best_acc


def cross_validate(cfg: Cfg, Xtr, ytr, k: int = 5) -> tuple[float, list]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    accs, fold_hists = [], []
    for f, (a, b) in enumerate(skf.split(Xtr, ytr)):
        model, hist, acc = fit_once(cfg, Xtr[a], ytr[a], Xtr[b], ytr[b])
        accs.append(acc)
        fold_hists.append(hist)
        print(f"  [{cfg.run}] fold{f}: {acc:.3f}")

        # Evict fold weights and clear cache
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return float(np.mean(accs)), fold_hists
