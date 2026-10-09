"""Memory-safe sequential runner for the cat/dog audio experiments.

The container has a hard ~2 GB RSS limit and three other training jobs may run
concurrently, so this runner:

  * executes ONE experiment at a time (subprocess per run => full memory release
    between runs),
  * resumes automatically: a run is skipped if artifacts/results.json already
    contains its key,
  * retries up to N times if a run is killed (OOM / external SIGKILL),
  * writes each finished run into artifacts/results.json immediately
    (crash-safe incremental checkpointing).

Usage:  python3 runner.py [run_name ...]     (default: E1..E4 in order)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "..", "artifacts")
RES_PATH = os.path.join(ARTIFACTS, "results.json")

ALL_RUNS = ["E1_single", "E2_multi", "E3_multi_noSepLoss", "E4_waveform"]
MAX_ATTEMPTS = 4


def done_runs() -> list[str]:
    if not os.path.exists(RES_PATH):
        return []
    try:
        return list(json.load(open(RES_PATH)).keys())
    except Exception:
        return []


def main() -> None:
    wanted = sys.argv[1:] or ALL_RUNS
    env = dict(os.environ, RESUME="1")
    for run in wanted:
        attempt = 0
        while run not in done_runs():
            attempt += 1
            if attempt > MAX_ATTEMPTS:
                print(f"[runner] {run}: giving up after {MAX_ATTEMPTS} attempts",
                      flush=True)
                break
            print(f"[runner] starting {run} (attempt {attempt})", flush=True)
            t0 = time.time()
            p = subprocess.run([sys.executable, "train_one.py", run],
                               cwd=HERE, env=env)
            dt = time.time() - t0
            if p.returncode != 0:
                print(f"[runner] {run} exited rc={p.returncode} after {dt:.0f}s; "
                      f"retrying", flush=True)
                time.sleep(5)
        if run in done_runs():
            print(f"[runner] {run} complete.", flush=True)
    print("[runner] ALL DONE ->", RES_PATH, flush=True)


if __name__ == "__main__":
    main()
