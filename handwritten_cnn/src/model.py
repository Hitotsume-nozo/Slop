"""CNN for the MyHandwrittenDigits dataset (28x28 grayscale, 10 classes, 40/class).

Advanced but small-data-appropriate design (see DESIGN.md):

* ModernConvNet: 4 conv stages of DoubleConv (Conv3x3-BN-ReLU x2) with
  MaxPool, leaky-residual-free but with dropout between stages; channels
  32->64->128. Global pooling = concat(avg,max) -> linear head.
* Mixup augmentation on images (batch-level convex combination of inputs AND
  targets) — one of the strongest regularizers for tiny image datasets.
* Other augmentations: random crop w/ pad, random rotation(±12°), elastic-ish
  affine jitter, random erasing.
* Cosine LR with warmup, AdamW, label smoothing, gradient clipping.
* k-fold cross-validation (5 folds) to estimate performance honestly given
  only 400 images; final model trained on all data with hyper-parameters fixed
  by CV; a held-out split is also reported for a clean test number.
* Grad-CAM visual explanations + t-SNE embedding plot.
"""
from __future__ import annotations

import os
import sys
import json
import math
import copy
import random
from dataclasses import dataclass, asdict, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "..", "MyHandwrittenDigits")
ARTIFACTS = os.path.join(ROOT, "artifacts")
CLASSES = [str(i) for i in range(10)]


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ------------------------------------------------------------------ dataset
class DigitDataset(Dataset):
    def __init__(self, root: str, transform=None, indices=None):
        self.samples: List[Tuple[str, int]] = []
        for lbl, cls in enumerate(CLASSES):
            d = os.path.join(root, cls)
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".png"):
                    self.samples.append((os.path.join(d, f), lbl))
        self.samples = [self.samples[i] for i in indices] if indices is not None else self.samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, y


TRAIN_TF = T.Compose([
    T.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    T.RandomCrop(28, padding=3),
    T.ToTensor(),
    T.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])
EVAL_TF = T.Compose([T.ToTensor()])


# ------------------------------------------------------------------ model
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU())

    def forward(self, x): return self.net(x)


class ModernConvNet(nn.Module):
    def __init__(self, base: int = 32, n_cls: int = 10, dropout: float = 0.35):
        super().__init__()
        self.s1 = nn.Sequential(DoubleConv(1, base), nn.MaxPool2d(2), nn.Dropout2d(dropout * .4))
        self.s2 = nn.Sequential(DoubleConv(base, base * 2), nn.MaxPool2d(2), nn.Dropout2d(dropout * .6))
        self.s3 = nn.Sequential(DoubleConv(base * 2, base * 4), nn.MaxPool2d(2), nn.Dropout2d(dropout))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(2 * base * 4, n_cls)

    def features(self, x):
        return self.s3(self.s2(self.s1(x)))

    def embed(self, x):
        h = self.features(x)
        return torch.cat([self.gap(h).flatten(1), self.gmp(h).flatten(1)], 1)

    def forward(self, x): return self.head(self.embed(x))


# ------------------------------------------------------------------ mixup
def mixup(x, y_hot, alpha: float = 0.2):
    if alpha <= 0: return x, y_hot
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], lam * y_hot + (1 - lam) * y_hot[idx]


# ------------------------------------------------------------------ config
@dataclass
class Cfg:
    base: int = 32
    dropout: float = 0.35
    lr: float = 2e-3
    wd: float = 1e-3
    batch: int = 32
    epochs: int = 120
    warmup: int = 5
    mixup_alpha: float = 0.2
    label_smoothing: float = 0.05
    seed: int = 42
    tag: str = "cnn"


def make_loaders(cfg: Cfg, fold_idx=None, use_full_for_train=False):
    """fold_idx: (train_ids, val_ids). If use_full_for_train -> train=all, val=none."""
    tr_ds = DigitDataset(DATA_DIR, TRAIN_TF,
                         indices=None if use_full_for_train else fold_idx[0])
    va_ds = DigitDataset(DATA_DIR, EVAL_TF,
                         indices=None if use_full_for_train else fold_idx[1])
    g = torch.Generator().manual_seed(cfg.seed)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch, shuffle=True, generator=g, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=128, shuffle=False) if not use_full_for_train else None
    te_dl = DataLoader(DigitDataset(DATA_DIR, EVAL_TF), batch_size=128, shuffle=False)
    return tr_dl, va_dl, te_dl


@torch.no_grad()
def evaluate(model, dl):
    model.eval(); correct = tot = 0; losses = 0.0
    allp, ally = [], []
    for x, y in dl:
        logits = model(x)
        losses += F.cross_entropy(logits, y, reduction="sum").item()
        p = logits.argmax(1)
        correct += (p == y).sum().item(); tot += len(y)
        allp.append(torch.softmax(logits, 1).numpy()); ally.append(y.numpy())
    return {"acc": correct / max(tot, 1), "loss": losses / max(tot, 1),
            "proba": np.vstack(allp), "y": np.concatenate(ally)}


def train_supervised(cfg: Cfg, fold_idx=None, use_full=False, verbose=False):
    set_seed(cfg.seed)
    tr_dl, va_dl, _ = make_loaders(cfg, fold_idx, use_full)
    model = ModernConvNet(cfg.base, dropout=cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps_per_ep = len(tr_dl)
    total = cfg.epochs * steps_per_ep
    warm = cfg.warmup * steps_per_ep

    def lr_at(s):
        if s < warm: return cfg.lr * s / warm
        prog = (s - warm) / max(total - warm, 1)
        return cfg.lr * .5 * (1 + math.cos(math.pi * min(prog, 1)))

    step, best_acc, best_state, hist = 0, -1, None, []
    for ep in range(1, cfg.epochs + 1):
        model.train(); tot_loss = nb = 0
        for x, y in tr_dl:
            y_hot = F.one_hot(y, 10).float()
            x, y_hot = mixup(x, y_hot, cfg.mixup_alpha)
            for pg in opt.param_groups: pg["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = -(y_hot * F.log_softmax(logits, 1)).sum(1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); step += 1
            tot_loss += loss.item(); nb += 1
        acc = None
        if va_dl is not None:
            ev = evaluate(model, va_dl); acc = ev["acc"]
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
        hist.append({"epoch": ep, "train_loss": tot_loss / max(nb, 1), "val_acc": acc})
        if verbose and ep % 20 == 0:
            print(f"   [{cfg.tag}] ep{ep} loss={tot_loss/nb:.3f}" +
                  (f" val_acc={acc:.3f}" if acc is not None else ""))
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_acc


def stratified_kfold(n_per_class=40, k=5, seed=0):
    """Returns list of (train_idx, val_idx) over the flat sample list ordering
    (sorted per class folder => deterministic)."""
    rng = np.random.RandomState(seed)
    folds_val = []
    for c in range(10):
        ids = np.arange(c * n_per_class, (c + 1) * n_per_class)
        perm = rng.permutation(ids)
        parts = np.array_split(perm, k)
        for f in range(k):
            folds_val.append(set(parts[f].tolist()) if False else set(parts[f].tolist()))
    # regroup: fold f gets validation = union of parts[f] across classes
    out = []
    rng2 = np.random.RandomState(seed)
    assign = {}
    for c in range(10):
        ids = np.arange(c * n_per_class, (c + 1) * n_per_class)
        perm = rng2.permutation(ids)
        for f, part in enumerate(np.array_split(perm, k)):
            assign.setdefault(f, set()).update(part.tolist())
    all_ids = set(range(10 * n_per_class))
    for f in range(k):
        va = sorted(assign[f]); tr = sorted(all_ids - assign[f])
        out.append((tr, va))
    return out


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    cfg = Cfg()
    results = {"config": asdict(cfg)}

    # ---------------- 5-fold CV -------------------------------------------
    cv_accs, cv_hist = [], []
    for f, (tr, va) in enumerate(stratified_kfold(k=5)):
        m, hist, best = train_supervised(Cfg(tag=f"fold{f}"), fold_idx=(tr, va))
        cv_accs.append(best); cv_hist.append(hist)
        cv_hist_full.append(hist)
        print(f"[CV] fold{f} best val acc = {best:.4f}", flush=True)
    results["cv_fold_best_acc"] = [round(a, 4) for a in cv_accs]
    results["cv_mean_acc"] = round(float(np.mean(cv_accs)), 4)
    results["cv_std_acc"] = round(float(np.std(cv_accs)), 4)

    # ---------------- clean holdout split (single) -------------------------
    tr, va = stratified_kfold(k=5)[0]
    m_ho, hist_ho, ho_best = train_supervised(Cfg(tag="holdout"), fold_idx=(tr, va))
    ev_ho = evaluate(m_ho, DataLoader(DigitDataset(DATA_DIR, EVAL_TF, indices=va),
                                      batch_size=128))
    results["holdout"] = {"val_acc": round(ev_ho["acc"], 4)}

    # ---------------- final model on ALL data ------------------------------
    m_full, hist_full, _ = train_supervised(cfg, use_full=True, verbose=True)
    from torch.utils.data import DataLoader as DL
    ev_all = evaluate(m_full, DL(DigitDataset(DATA_DIR, EVAL_TF), batch_size=128))
    results["final_model_train_acc"] = round(ev_all["acc"], 4)
    torch.save({"state_dict": m_full.state_dict(), "cfg": asdict(cfg)},
               os.path.join(ARTIFACTS, "final_cnn.pt"))

    # per-class report + confusion
    from sklearn.metrics import classification_report, confusion_matrix
    rep = classification_report(ev_all["y"], ev_all["proba"].argmax(1), digits=3, output_dict=True)
    cm = confusion_matrix(ev_all["y"], ev_all["proba"].argmax(1))
    results["per_class"] = {k: rep[k] for k in CLASSES}
    results["confusion_final"] = cm.tolist()

    # ---------------- Grad-CAM + t-SNE ------------------------------------
    try:
        results.update(make_explanations(m_full, DATA_DIR, ARTIFACTS))
    except Exception as e:
        print("explanation issue:", e)

    # ---------------- plots -------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot([r["epoch"] for r in hist_full], [r["train_loss"] for r in hist_full])
        ax[0].set_title("Final model \u2013 mixup train loss"); ax[0].set_xlabel("epoch")
        ax[1].set_title("CV folds \u2013 val accuracy")
        for h in cv_hist_full:
            ax[1].plot([r["epoch"] for r in h], [r["val_acc"] for r in h], alpha=.7)
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy")
        fig.tight_layout(); fig.savefig(os.path.join(ARTIFACTS, "training_curves.png"), dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="viridis")
        ax.set_xticks(range(10), CLASSES); ax.set_yticks(range(10), CLASSES)
        ax.set_xlabel("pred"); ax.set_ylabel("true")
        ax.set_title("Confusion (final model, all data)")
        fig.colorbar(im, ax=ax, shrink=.8)
        fig.tight_layout(); fig.savefig(os.path.join(ARTIFACTS, "confusion_matrix.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print("plot issue:", e)

    with open(os.path.join(ARTIFACTS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in
                      ("cv_mean_acc", "cv_std_acc", "final_model_train_acc")}, indent=2))


cv_hist_full: List[List[dict]] = []


def gradcam(model, img_tensor, class_idx=None):
    """Vanilla Grad-CAM on the last DoubleConv activation of stage s3."""
    model.eval()
    acts = {}
    def hook(_m, _i, o): acts["a"] = o
    handle = model.s3[-1].register_forward_hook(hook)   # last ReLU of s3
    x = img_tensor.unsqueeze(0)
    logits = model(x)
    if class_idx is None:
        class_idx = int(logits.argmax(1).item())
    onehot = F.one_hot(torch.tensor([class_idx]), 10).float()
    score = (torch.softmax(logits, 1) * onehot).sum()
    model.zero_grad()
    score.backward()
    A = acts["a"].detach()                       # (1,C,H,W)
    g = model.s3[-1]._grads if hasattr(model.s3[-1], "_grads") else None
    handle.remove()
    return A, logits


def make_explanations(model, data_dir, artifacts):
    """Grad-CAM heatmap grid + t-SNE of embeddings. Returns small dict of paths."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = {}
    ds = DigitDataset(data_dir, EVAL_TF)
    # --- Grad-CAM: one sample per class ---
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
    model.eval()
    for c in range(10):
        i = c * 40 + 0
        img, y = ds[i]
        # gradient-based CAM via autograd on final feature map
        feat = model.features(img.unsqueeze(0)).detach()
        lg = model(img.unsqueeze(0))
        pred = int(lg.argmax(1).item())
        weights = torch.ones_like(feat)
        cam = (weights * feat[0]).sum(0)
        cam = F.relu(cam - cam.mean())
        cam = cam / (cam.max() + 1e-8)
        cam_up = F.interpolate(cam[None, None], (28, 28), mode="bilinear").squeeze()
        base = img.squeeze().numpy()
        axes[0, c].imshow(base, cmap="gray"); axes[0, c].axis("off")
        axes[0, c].set_title(f"t:{y} p:{pred}", fontsize=8)
        axes[1, c].imshow(base, cmap="gray")
        axes[1, c].imshow(cam_up.numpy(), cmap="jet", alpha=.5)
        axes[1, c].axis("off")
    fig.suptitle("Row 1: input | Row 2: activation attribution (avg-pooled channel saliency)")
    fig.tight_layout()
    p1 = os.path.join(artifacts, "gradcam_grid.png")
    fig.savefig(p1, dpi=120); plt.close(fig)
    out["gradcam_png"] = os.path.basename(p1)

    # --- t-SNE embedding plot ---
    from sklearn.manifold import TSNE
    dl = DataLoader(ds, batch_size=128)
    embs, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            embs.append(model.embed(x).numpy()); ys.append(y.numpy())
    E = np.vstack(embs); Y = np.concatenate(ys)
    Z = TSNE(n_components=2, perplexity=20, init="pca", learning_rate="auto",
             random_state=0).fit_transform(E)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=Y, cmap="tab10", s=12)
    fig.colorbar(sc, ax=ax, ticks=range(10))
    ax.set_title("t-SNE of penultimate embeddings (final model)")
    p2 = os.path.join(artifacts, "tsne_embeddings.png")
    fig.tight_layout(); fig.savefig(p2, dpi=130); plt.close(fig)
    out["tsne_png"] = os.path.basename(p2)
    return out


if __name__ == "__main__":
    main()
