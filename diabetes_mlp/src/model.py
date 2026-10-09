"""MLP model + training engine for the Diabetes-Risk task (PyTorch).

Design highlights (see DESIGN.md §4–§7):
* Configurable-depth MLP with LayerNorm + GELU + dropout residual blocks.
* Class-imbalanced target (Low ≈ 1%) -> weighted CrossEntropy computed from
  train-set inverse frequencies (clipped), plus optional focal loss.
* AdamW + cosine-annealing-with-warmup LR schedule, gradient clipping.
* Early stopping on validation macro-F1 with checkpoint restore of best state.
* Deterministic seeding; CPU-friendly batch size.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
N_CLASSES = 3


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm residual-ish MLP block: Linear -> Norm -> GELU -> Dropout."""
    def __init__(self, din: int, dout: int, dropout: float, use_norm: str = "layernorm"):
        super().__init__()
        self.lin = nn.Linear(din, dout)
        if use_norm == "layernorm":
            self.norm: nn.Module = nn.LayerNorm(dout)
        elif use_norm == "batchnorm":
            self.norm = nn.BatchNorm1d(dout)
        else:
            self.norm = nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.lin(x))))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float = 0.3,
                 use_norm: str = "layernorm"):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers.append(Block(d, h, dropout, use_norm))
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, N_CLASSES)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.body(x))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def class_weights_from_counts(counts: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    """Effective-number reweighting (Cui et al., 2019), clipped for stability."""
    eff = 1.0 - np.power(beta, counts.astype(np.float64))
    w = (1.0 / eff)
    w = w / w.sum() * len(w)
    w = np.clip(w, 0.2, 5.0)
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, weight=self.weight, reduction="none")
        p_t = torch.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Training config / engine
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    hidden: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    use_norm: str = "layernorm"
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 120
    warmup_epochs: int = 5
    patience: int = 15
    label_smoothing: float = 0.02
    focal_gamma: Optional[float] = None      # None -> plain weighted CE
    seed: int = 42
    tag: str = "model"


def make_lr_fn(cfg: TrainConfig, steps_per_epoch: int):
    total = cfg.max_epochs * steps_per_epoch
    warm = max(1, cfg.warmup_epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        if step < warm:
            return cfg.lr * step / warm
        prog = (step - warm) / max(1, total - warm)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return lr_at


@torch.no_grad()
def predict_probs(model: nn.Module, X: np.ndarray, bs: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs].astype("float32"))
        out.append(torch.softmax(model(xb), dim=1).numpy())
    return np.vstack(out)


def evaluate_arrays(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    pred = proba.argmax(1)
    m = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
    }
    try:
        m["roc_auc_ovr"] = float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except ValueError:
        m["roc_auc_ovr"] = float("nan")
    return m


def train_one_cfg(cfg: TrainConfig, data: Dict[str, np.ndarray],
                  verbose: bool = True) -> Dict:
    set_seed(cfg.seed)
    Xtr, ytr, Xva, yva = data["Xtr"], data["ytr"], data["Xva"], data["yva"]
    in_dim = Xtr.shape[1]

    gts = torch.from_numpy(ytr)
    counts = np.bincount(gts.numpy(), minlength=N_CLASSES)
    w = class_weights_from_counts(counts)
    if cfg.focal_gamma:
        crit: nn.Module = FocalLoss(w, cfg.focal_gamma)
    else:
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

    model = MLP(in_dim, cfg.hidden, cfg.dropout, cfg.use_norm)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_at = make_lr_fn(cfg, math.ceil(len(Xtr) / cfg.batch_size))

    Xt = torch.from_numpy(Xtr.astype("float32"))
    Yt = torch.from_numpy(ytr.astype("int64"))
    gen = torch.Generator().manual_seed(cfg.seed)

    best_f1, best_state, best_epoch, since_best = -1.0, None, 0, 0
    history: List[Dict] = []
    step = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        tot_loss, nb = 0.0, 0
        for i in range(0, len(Xt), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = Xt[idx], Yt[idx]
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += float(loss.item()); nb += 1; step += 1

        proba = predict_probs(model, Xva)
        ev = evaluate_arrays(yva, proba)
        rec = {"epoch": epoch, "train_loss": tot_loss / max(nb, 1),
               "val_acc": ev["accuracy"], "val_macro_f1": ev["macro_f1"],
               "lr": lr_at(step - 1)}
        history.append(rec)
        if verbose and (epoch % 10 == 1 or epoch == cfg.max_epochs):
            print(f"[{cfg.tag}] ep{epoch:3d} loss={rec['train_loss']:.4f} "
                  f"val_acc={ev['accuracy']:.4f} val_f1={ev['macro_f1']:.4f}")

        if ev["macro_f1"] > best_f1 + 1e-5:
            best_f1, best_epoch, since_best = ev["macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
            if since_best >= cfg.patience:
                if verbose:
                    print(f"[{cfg.tag}] early stop @ep{epoch} (best ep{best_epoch})")
                break

    model.load_state_dict(best_state)
    va_proba = predict_probs(model, Xva)
    te_proba = predict_probs(model, data["Xte"])
    result = {
        "config": asdict(cfg),
        "n_params": sum(p.numel() for p in model.parameters()),
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "val_metrics": evaluate_arrays(yva, va_proba),
        "test_metrics": evaluate_arrays(data["yte"], te_proba),
        "history": history,
    }
    os.makedirs(ARTIFACTS, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg),
                "in_dim": in_dim}, os.path.join(ARTIFACTS, f"{cfg.tag}.pt"))
    np.savez(os.path.join(ARTIFACTS, f"{cfg.tag}_proba.npz"),
             va=va_proba, te=te_proba)
    return result, model
