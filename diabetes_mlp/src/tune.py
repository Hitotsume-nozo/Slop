"""Stage 1 – Hyper-parameter search for the Diabetes-Risk MLP.

Protocol (documented in DESIGN.md §7):
  Phase A: coarse grid over (depth/width family x dropout) at fixed lr=2e-3,
           weighted CE. Selection metric = validation macro-F1.
  Phase B: refine around the best config – learning-rate sweep + loss variant
           (weighted CE vs focal gamma=2).
Writes artifacts/tuning_results.json with full history of every run.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_dataset, ARTIFACTS          # noqa: E402
from model import TrainConfig, train_one_cfg       # noqa: E402

QUICK = "--quick" in sys.argv   # smoke-test mode: fewer epochs


def main() -> None:
    data = build_dataset()
    results = []

    def run(cfg: TrainConfig):
        t0 = time.time()
        res, _ = train_one_cfg(cfg, data, verbose=True)
        res["wall_s"] = round(time.time() - t0, 1)
        results.append(res)
        print(f"== {cfg.tag}: val_f1={res['val_metrics']['macro_f1']:.4f} "
              f"test_f1={res['test_metrics']['macro_f1']:.4f} ({res['wall_s']}s)\n")

    ep = 30 if QUICK else 120
    pat = 8 if QUICK else 15

    # ---------------- Phase A: architecture / dropout grid ----------------
    archs = {
        "small": [64, 32],
        "med":   [128, 64, 32],
        "wide":  [256, 128, 64],
        "deep":  [256, 128, 64, 32],
    }
    dropouts = [0.15, 0.3] if not QUICK else [0.3]
    best_a = None
    for aname, hidden in archs.items():
        for dp in dropouts:
            cfg = TrainConfig(hidden=hidden, dropout=dp, max_epochs=ep, patience=pat,
                              tag=f"A_{aname}_dp{int(dp*100)}")
            run(cfg)
            m = results[-1]["val_metrics"]["macro_f1"]
            if best_a is None or m > best_a[0]:
                best_a = (m, cfg)

    # ---------------- Phase B: lr + loss refinement -----------------------
    lrs = [1e-3, 2e-3, 4e-3] if not QUICK else [2e-3]
    for lr in lrs:
        for focal in ([None, 2.0] if not QUICK else [None]):
            cfg = TrainConfig(hidden=best_a[1].hidden, dropout=best_a[1].dropout,
                              lr=lr, max_epochs=ep, patience=pat,
                              focal_gamma=focal,
                              tag=f"B_lr{lr:g}_{'focal' + str(focal) if focal else 'wce'}")
            run(cfg)

    results.sort(key=lambda r: r["val_metrics"]["macro_f1"], reverse=True)
    with open(os.path.join(ARTIFACTS, "tuning_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    top = results[0]
    print("\nBEST:", top["config"]["tag"],
          "val_f1=%.4f test_f1=%.4f" % (top["val_metrics"]["macro_f1"],
                                        top["test_metrics"]["macro_f1"]))


if __name__ == "__main__":
    main()
