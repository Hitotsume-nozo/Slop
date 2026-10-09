"""Run exactly ONE experiment from the cat/dog audio matrix and checkpoint it."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import ARTIFACTS  # noqa: E402
from train import (
    Cfg,
    get_data,
    cross_validate,
    fit_once,  # noqa: E402
    predict,
    get_embeddings,
)

# Batch 32 for spectrograms; Batch 16 for raw waveforms (to prevent 64k-sample activation OOM)
RUN_CONFIGS = {
    "E1_single": dict(run="E1_single", input="single", arch="spectro", batch=32),
    "E2_multi": dict(run="E2_multi", input="multi", arch="spectro", batch=32),
    "E3_multi_noSepLoss": dict(
        run="E3_multi_noSepLoss", input="multi", arch="spectro", lam_sep=0.0, batch=32
    ),
    "E4_waveform": dict(
        run="E4_waveform", input="wave", arch="wave", augment=False, epochs=40, batch=16
    ),
}


def main() -> None:
    name = sys.argv[1]
    cfg = Cfg(**RUN_CONFIGS[name])
    res_path = os.path.join(ARTIFACTS, "results.json")
    results = {}
    if os.path.exists(res_path):
        try:
            results = json.load(open(res_path))
        except Exception:
            results = {}
    if name in results:
        print(f"[{name}] already present in results.json – skipping")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    t0 = time.time()
    Xtr, ytr, Xte, yte = get_data(cfg)
    cv_mean, _ = cross_validate(cfg, Xtr, ytr)

    # 80/20 train/holdout split
    n = len(Xtr)
    rs = np.random.RandomState(cfg.seed)
    perm = rs.permutation(n)
    cut = int(0.8 * n)
    a, b = perm[:cut], perm[cut:]
    model, hist, holdout_acc = fit_once(
        cfg, Xtr[a], ytr[a], Xtr[b], ytr[b], verbose=True
    )

    model = model.to(device)
    model.eval()

    pred_te = predict(model, Xte, bs=cfg.batch)
    te_acc = float((pred_te == yte).mean())

    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    cm = confusion_matrix(yte, pred_te).tolist()
    f1 = float(f1_score(yte, pred_te, average="binary"))

    # Batched inference for probabilities (safe for 64k waveform samples)
    with torch.no_grad():
        proba_chunks = []
        for i in range(0, len(Xte), cfg.batch):
            batch_x = torch.as_tensor(
                Xte[i : i + cfg.batch], device=device, dtype=torch.float32
            )
            proba_chunks.append(
                torch.softmax(model(batch_x), dim=1)[:, 1].cpu().numpy()
            )
        proba = np.concatenate(proba_chunks)
    auc = float(roc_auc_score(yte, proba))

    emb_te = get_embeddings(model, Xte, bs=cfg.batch)
    np.savez(os.path.join(ARTIFACTS, f"emb_{name}.npz"), test=emb_te, y_test=yte)

    # t-SNE plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        Z = TSNE(
            n_components=2, init="pca", random_state=0, perplexity=15, n_jobs=-1
        ).fit_transform(emb_te)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=yte, cmap="coolwarm", s=26)
        fig.colorbar(sc, ax=ax, ticks=[0, 1], label="cat=0 / dog=1")
        ax.set_title(f"{name}: test embeddings (t-SNE)")
        fig.tight_layout()
        fig.savefig(os.path.join(ARTIFACTS, f"tsne_{name}.png"), dpi=130)
        plt.close(fig)
    except Exception as ex:
        print("tsne plot skipped:", ex)

    results[name] = {
        "config": {
            "run": cfg.run,
            "input": cfg.input,
            "arch": cfg.arch,
            "base": cfg.base,
            "dropout": cfg.dropout,
            "lam_sep": cfg.lam_sep,
            "margin": cfg.margin,
            "lr": cfg.lr,
            "wd": cfg.wd,
            "batch": cfg.batch,
            "epochs": cfg.epochs,
            "seed": cfg.seed,
            "augment": cfg.augment,
        },
        "cv_mean_acc": round(cv_mean, 4),
        "holdout_acc": round(holdout_acc, 4),
        "test_acc": round(te_acc, 4),
        "test_f1": round(f1, 4),
        "test_roc_auc": auc,
        "confusion_test": cm,
        "wall_s": round(time.time() - t0, 1),
        "final_history": hist[-10:],
    }
    torch.save(model.state_dict(), os.path.join(ARTIFACTS, f"{name}.pt"))
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"== {name}: cv={cv_mean:.3f} holdout={holdout_acc:.3f} "
        f"TEST acc={te_acc:.3f} f1={f1:.3f} auc={auc:.4f} "
        f"({results[name]['wall_s']}s)",
        flush=True,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
