"""Training / evaluation driver for the cat-vs-dog audio separation classifier.

Experiment matrix (all runs share one engine, differ only by config):
  E1: SpectroCNN on SINGLE-channel log-mel
  E2: SpectroCNN on MULTI-channel (mel + delta + delta-delta)
  E3: Ablation – multi-channel WITHOUT the separation loss term
  E4: RawWaveNet on padded waveform (domain check, no mel front-end)

Protocol
--------
* Official split = folder structure (train/ vs test/). Because train is tiny
  (210 clips), model selection is done with stratified 5-fold CV *inside train*
  (mean val accuracy over folds); the official test set is touched exactly once
  per final configuration to report generalization.
* Class imbalance (125 cat / 85 dog) handled with weighted CE.
* AdamW, cosine LR, grad clipping, SpecAugment (spectrogram runs),
  deterministic seeds, best-epoch checkpointing on CV-fold mean accuracy.
Outputs: artifacts/results.json, embedding t-SNE plot, confusion matrices,
per-run training logs.
"""
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
from model import (SpectroCNN, RawWaveNet, CombinedLoss, separation_loss,
                   spec_augment)   # noqa: E402


def set_seed(s: int):
    np.random.seed(s); torch.manual_seed(s)


@dataclass
class Cfg:
    run: str = "E1_single"
    input: str = "multi"          # 'single' | 'multi' | 'wave'
    arch: str = "spectro"         # 'spectro' | 'wave'
    base: int = 16
    dropout: float = 0.3
    lam_sep: float = 0.1          # 0 -> ablation without separation loss
    margin: float = 2.0
    lr: float = 1e-3
    wd: float = 1e-4
    batch: int = 16
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
            s = (len(x) - L) // 2; x = x[s:s + L]
        else:
            x = np.pad(x, ((L - len(x)) // 2, 0))
        out.append(x)
    return np.stack(out).astype("float32")[:, None, :]      # (N,1,L)


# ------------------------------------------------------------------ train engine
def make_model(cfg: Cfg):
    if cfg.arch == "spectro":
        in_ch = 1 if cfg.input == "single" else 3
        return SpectroCNN(in_ch=in_ch, base=cfg.base, dropout=cfg.dropout)
    return RawWaveNet(width=cfg.base)


def get_data(cfg: Cfg):
    if cfg.arch == "spectro":
        d = extract_all(cfg.input)
        Xtr = d["X"][d["splits"] == "train"]; ytr = d["y"][d["splits"] == "train"]
        Xte = d["X"][d["splits"] == "test"];  yte = d["y"][d["splits"] == "test"]
    else:
        X = load_waveforms()
        d = extract_all("single")
        y, sp = d["y"], d["splits"]
        Xtr, ytr = X[sp == "train"], y[sp == "train"]
        Xte, yte = X[sp == "test"], y[sp == "test"]
    return Xtr, ytr, Xte, yte


@torch.no_grad()
def predict(model, X, bs=64):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs])
        outs.append(model(xb).argmax(1).numpy())
    return np.concatenate(outs)


@torch.no_grad()
def get_embeddings(model, X, bs=64):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        outs.append(model.embed(torch.from_numpy(X[i:i + bs])).numpy())
    return np.vstack(outs)


def fit_once(cfg: Cfg, Xtr, ytr, Xva, yva, verbose=False):
    set_seed(cfg.seed)
    model = make_model(cfg)
    counts = np.bincount(ytr, minlength=2).astype("float64")
    w = torch.tensor((counts.sum() / (2 * np.maximum(counts, 1))), dtype=torch.float32)
    ce = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    steps = math.ceil(len(Xt) / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr,
                                                total_steps=cfg.epochs * steps)
    gen = torch.Generator().manual_seed(cfg.seed)
    best_acc, best_state, hist = -1, None, []

    for ep in range(1, cfg.epochs + 1):
        model.train(); tot = 0.0
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), cfg.batch):
            idx = perm[i:i + cfg.batch]
            xb, yb = Xt[idx], Yt[idx]
            if cfg.augment and cfg.arch == "spectro":
                xb = spec_augment(xb)
            opt.zero_grad(set_to_none=True)
            if cfg.arch == "spectro":
                emb = model.embed(xb); logits = model.head(emb)
            else:
                logits = model(xb); emb = model.embed(xb)
            loss = ce(logits, yb)
            if cfg.lam_sep > 0:
                loss = loss + cfg.lam_sep * separation_loss(emb, yb, cfg.margin)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += float(loss.item())
        acc = (predict(model, Xva) == yva).mean() if Xva is not None else None
        hist.append({"epoch": ep, "loss": tot / max(steps, 1), "val_acc": acc})
        if Xva is not None and acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
        if verbose and ep % 20 == 0:
            print(f"   ep{ep} loss={tot/steps:.3f} val_acc={acc:.3f}")
    if Xva is not None:
        model.load_state_dict(best_state)
    return model, hist, best_acc


def cross_validate(cfg: Cfg, Xtr, ytr, k: int = 5) -> tuple[float, list]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    accs, fold_hists = [], []
    for f, (a, b) in enumerate(skf.split(Xtr, ytr)):
        _, hist, acc = fit_once(cfg, Xtr[a], ytr[a], Xtr[b], ytr[b])
        accs.append(acc); fold_hists.append(hist)
        print(f"  [{cfg.run}] fold{f}: {acc:.3f}")
    return float(np.mean(accs)), fold_hists


RESUME = os.environ.get("RESUME", "1") == "1"


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    results = {}
    embed_store = {}
    res_path = os.path.join(ARTIFACTS, "results.json")
    if RESUME and os.path.exists(res_path):
        try:
            results = json.load(open(res_path))
            print("resuming; already done:", list(results))
        except Exception:
            results = {}

    configs = [
        Cfg(run="E1_single", input="single", arch="spectro"),
        Cfg(run="E2_multi", input="multi", arch="spectro"),
        Cfg(run="E3_multi_noSepLoss", input="multi", arch="spectro", lam_sep=0.0),
        Cfg(run="E4_waveform", input="wave", arch="wave", augment=False, epochs=40),
    ]

    for cfg in configs:
        if cfg.run in results:
            continue
        import gc
        gc.collect()
        t0 = time.time()
        Xtr, ytr, Xte, yte = get_data(cfg)
        cv_mean, _ = cross_validate(cfg, Xtr, ytr)
        # final fit: full train, hold-out 20% of train for checkpointing
        n = len(Xtr); cut = int(.8 * n)
        rs = np.random.RandomState(cfg.seed); perm = rs.permutation(n)
        a, b = perm[:cut], perm[cut:]
        model, hist, holdout_acc = fit_once(cfg, Xtr[a], ytr[a], Xtr[b], ytr[b], verbose=True)
        pred_te = predict(model, Xte)
        te_acc = float((pred_te == yte).mean())
        from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
        cm = confusion_matrix(yte, pred_te).tolist()
        f1 = float(f1_score(yte, pred_te, average="binary"))
        with torch.no_grad():
            proba = np.vstack([torch.softmax(model(torch.from_numpy(Xte[i:i + 64])), 1)[:, 1].numpy()
                               for i in range(0, len(Xte), 64)])
        auc = float(roc_auc_score(yte, proba))
        # store only the small test-set embeddings (for the t-SNE plot script);
        # train embeddings would blow the 2 GB RAM budget when kept for all runs
        embed_store[cfg.run] = {"test": get_embeddings(model, Xte),
                                "y_test": yte.tolist()}
        np.savez(os.path.join(ARTIFACTS, f"emb_{cfg.run}.npz"),
                 test=embed_store[cfg.run]["test"],
                 y_test=np.asarray(embed_store[cfg.run]["y_test"]))
        results[cfg.run] = {
            "config": asdict(cfg), "cv_mean_acc": round(cv_mean, 4),
            "holdout_acc": round(holdout_acc, 4), "test_acc": round(te_acc, 4),
            "test_f1": round(f1, 4), "test_roc_auc": auc,
            "confusion_test": cm, "wall_s": round(time.time() - t0, 1),
            "final_history": hist[-10:],
        }
        torch.save(model.state_dict(), os.path.join(ARTIFACTS, f"{cfg.run}.pt"))
        print(f"== {cfg.run}: cv={cv_mean:.3f} holdout={holdout_acc:.3f} "
              f"TEST acc={te_acc:.3f} f1={f1:.3f} auc={auc}\n", flush=True)
        embed_store.clear()       # keep RAM low (container has ~2 GB limit)
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)

    with open(os.path.join(ARTIFACTS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE ->", os.path.join(ARTIFACTS, "results.json"))


if __name__ == "__main__":
    main()
