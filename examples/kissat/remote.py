#!/usr/bin/env python3
# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Pool-worker entry point for ``evaluator_pool.py``: score one candidate here.

    python3 remote.py STAGE PROGRAM METRICS_JSON      # STAGE: stage1 | stage2 | <set name>

The first task on a worker installs gcc/make if missing, builds the pinned
Kissat and fetches the instances into ``KISSAT_EVAL_HOME`` (default
``/var/tmp/oeb-kissat``, which outlives the per-task directory). Later tasks
on the same worker reuse it. Scoring uses the committed ``stock_ticks.json``
(tick metric), so nothing is re-measured per worker.
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("KISSAT_EVAL_HOME", "/var/tmp/oeb-kissat")
sys.path.insert(0, str(HERE))
import evaluator  # noqa: E402
import prepare  # noqa: E402


def ensure_toolchain() -> None:
    if shutil.which("gcc") and shutil.which("make"):
        return
    subprocess.run(["sudo", "-n", "dnf", "install", "-y", "-q", "gcc", "make"], check=True)


def prepared_home() -> Path:
    home = evaluator.home()
    home.mkdir(parents=True, exist_ok=True)
    with open(home / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ensure_toolchain()
        prepare.build_kissat(home)
        rows = list({r["hash"]: r for rows in evaluator.instance_sets().values() for r in rows}.values())
        prepare.fetch_instances(home, rows)
    return home


def main(stage: str, program: str, out: str) -> int:
    t0 = time.time()
    prepared_home()
    setup_s = time.time() - t0
    set_name = {"stage1": "stage1",
                "stage2": os.environ.get("KISSAT_EVAL_SPLIT", "train")}.get(stage, stage)
    metrics = evaluator.score(program, set_name, evaluator.cap())
    metrics.update({
        "set": set_name,
        "worker_machine": platform.machine(),
        "worker_cpus": os.cpu_count(),
        "worker_setup_s": round(setup_s, 1),
        "worker_eval_s": round(time.time() - t0 - setup_s, 1),
    })
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(metrics))
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
