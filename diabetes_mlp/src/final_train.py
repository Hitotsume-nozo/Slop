"""Stage 2 – Final training, evaluation and reporting for the Diabetes MLP.

Steps
-----
1. Rebuild/load arrays (stage-1 cache).
2. Train the winning configuration from artifacts/tuning_results.json as a
   3-seed ensemble (seed averaging reduces variance; ensembling further helps
   the rare "Low" class).
3. Baselines on identical splits: multinomial Logistic Regression and
   Gradient Boosting (sklearn) — to prove the MLP earns its complexity.
4. Full test metrics: accuracy, macro/weighted F1, per-class P/R/F1,
   confusion matrix, ROC-AUC(OvR), calibration spot-checks.
5. Feature-importance via permutation importance (on val set).
6. Artifacts: final_report.json, confusion_matrix.png, training_curves.png,
   per_class_metrics.png, feature_importance.png + this script's stdout log.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_dataset, ARTIFACTS, CLASS_ORDER          # noqa: E402
from model import TrainConfig, train_one_cfg, evaluate_arrays   # noqa: E402


def plot_history(histories, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for h in histories:
        ep = [r["epoch"] for r in h]
        ax[0].plot(ep, [r["train_loss"] for r in h], label=h[0]["tag"] if "tag" in h[0] else "")
        ax[1].plot(ep, [r["val_macro_f1"] for r in h])
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[0].set_title("Training loss")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("macro-F1"); ax[1].set_title("Val macro-F1")
    for a in ax: a.grid(alpha=.3)
    fig.suptitle("Diabetes MLP – training curves (ensemble members)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xticks(range(3), CLASS_ORDER); ax.set_yticks(range(3), CLASS_ORDER)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Test confusion matrix")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_per_class(per_class, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = CLASS_ORDER
    x = np.arange(len(labels)); w = .27
    ax.bar(x - w, per_class["precision"], w, label="precision")
    ax.bar(x, per_class["recall"], w, label="recall")
    ax.bar(x + w, per_class["f1"], w, label="F1")
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.05)
    ax.set_title("Test per-class metrics (MLP ensemble)"); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    t0 = time.time()
    data = build_dataset()
    Xtr, ytr, Xva, yva, Xte, yte = (data[k] for k in
                                    ("Xtr", "ytr", "Xva", "yva", "Xte", "yte"))

    # ---- recover best config from tuning ----------------------------------
    with open(os.path.join(ARTIFACTS, "tuning_results.json")) as f:
        tuned = json.load(f)
    best = tuned[0]["config"]
    print("Best config from tuning:", {k: best[k] for k in
                                       ("hidden", "dropout", "lr", "focal_gamma", "use_norm")})

    # ---- 3-seed ensemble ---------------------------------------------------
    ens_va = np.zeros_like(np.load(os.path.join(ARTIFACTS, f"{best['tag']}_proba.npz"))["va"])
    ens_te = None
    histories = []
    for seed in (42, 7, 2024):
        cfg = TrainConfig(**{k: v for k, v in best.items() if k != "tag"},
                          seed=seed, tag=f"final_seed{seed}")
        res, _ = train_one_cfg(cfg, data, verbose=True)
        p = np.load(os.path.join(ARTIFACTS, f"{cfg.tag}_proba.npz"))
        ens_va += p["va"]; 
        ens_te = p["te"] if ens_te is None else ens_te + p["te"]
        h = res["history"]
        h[0]["tag"] = cfg.tag
        histories.append(h)
    ens_va /= 3; ens_te /= 3

    mlp_val = evaluate_arrays(yva, ens_va)
    mlp_test = evaluate_arrays(yte, ens_te)
    print("ENSEMBLE val:", mlp_val)
    print("ENSEMBLE test:", mlp_test)

    # ---- baselines ----------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    base = {}
    lr_clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
    lr_clf.fit(Xtr, ytr)
    base["logreg"] = evaluate_arrays(yte, lr_clf.predict_proba(Xte))

    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                        random_state=42, class_weight="balanced")
    gb.fit(Xtr, ytr)
    base["hist_gbm"] = evaluate_arrays(yte, gb.predict_proba(Xte))
    print("Baselines (test):", json.dumps(base, indent=2))

    # ---- per-class report + confusion ---------------------------------------
    pred_te = ens_te.argmax(1)
    rep = classification_report(yte, pred_te, target_names=CLASS_ORDER, output_dict=True)
    cm = confusion_matrix(yte, pred_te)
    per_class = {
        "precision": [rep[c]["precision"] for c in CLASS_ORDER],
        "recall": [rep[c]["recall"] for c in CLASS_ORDER],
        "f1": [rep[c]["f1-score"] for c in CLASS_ORDER],
    }

    # ---- permutation importance (val, first 5k rows for speed) --------------
    from sklearn.inspection import permutation_importance
    import torch, copy
    from model import MLP
    ck = torch.load(os.path.join(ARTIFACTS, f"final_seed42.pt"), weights_only=False)
    m = MLP(ck["in_dim"], ck["cfg"]["hidden"], ck["cfg"]["dropout"], ck["cfg"]["use_norm"])
    m.load_state_dict(ck["state_dict"]); m.eval()

    def _torch_pred(model, Xb, idx):
        with torch.no_grad():
            logits = model(torch.from_numpy(Xb[idx].astype("float32")))
        return (logits.argmax(1).numpy() == yva[idx]).astype(float)

    rng_idx = np.random.default_rng(0).choice(len(Xva), size=min(5000, len(Xva)), replace=False)
    feat_names = data["state"].numeric_cols + data["state"].onehot_cols
    scores = []
    for j in range(Xva.shape[1]):
        orig = _torch_pred(m, Xva, rng_idx)
        backup = Xva[rng_idx, j].copy()
        rs = np.random.default_rng(j)
        accs = []
        for _ in range(3):
            Xva[rng_idx, j] = rs.permutation(Xva[rng_idx, j])
            accs.append(_torch_pred(m, Xva, rng_idx).mean())
        Xva[rng_idx, j] = backup
        scores.append(orig - float(np.mean(accs)))
    order = np.argsort(scores)[::-1][:20]
    fi = [(feat_names[j], round(scores[j], 5)) for j in order]

    # ---- plots ---------------------------------------------------------------
    plot_history(histories, os.path.join(ARTIFACTS, "training_curves.png"))
    plot_confusion(cm, os.path.join(ARTIFACTS, "confusion_matrix.png"))
    plot_per_class(per_class, os.path.join(ARTIFACTS, "per_class_metrics.png"))

    fig, ax = plt.subplots(figsize=(8, 6))
    names = [n for n, _ in fi][::-1]; vals = [v for _, v in fi][::-1]
    ax.barh(names, vals, color="steelblue")
    ax.set_xlabel("accuracy drop when feature shuffled (val)")
    ax.set_title("Top-20 permutation feature importance (MLP)")
    fig.tight_layout(); fig.savefig(os.path.join(ARTIFACTS, "feature_importance.png"), dpi=130)
    plt.close(fig)

    # ---- final report ---------------------------------------------------------
    report = {
        "wall_time_s": round(time.time() - t0, 1),
        "selected_config": best,
        "ensemble_seeds": [42, 7, 2024],
        "mlp_single_val": {"note": "see tuning_results.json top entry"},
        "mlp_ensemble_val": mlp_val,
        "mlp_ensemble_test": mlp_test,
        "baselines_test": base,
        "per_class_test": per_class,
        "confusion_matrix_test": cm.tolist(),
        "class_labels": CLASS_ORDER,
        "top_features": fi,
    }
    with open(os.path.join(ARTIFACTS, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("mlp_ensemble_test", "baselines_test")}, indent=2))


if __name__ == "__main__":
    main()
