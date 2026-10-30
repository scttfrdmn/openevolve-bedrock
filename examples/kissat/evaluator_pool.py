# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
OpenEvolve evaluator for the Kissat example that scores candidates on a
spore.host ``spawn pool`` (issue #1) instead of this machine.

Pass this file to ``oe-bedrock run`` in place of ``evaluator.py``. Each stage
runs ``remote.py`` on a pool worker, which scores the candidate exactly as
``evaluator.py`` would locally (same metrics, same tick baseline). The source
check runs here first, so a forbidden candidate isn't sent to the pool.

Needs ``OE_BEDROCK_POOL_RUN_ID`` and ``OE_BEDROCK_POOL_STAGING`` (see
``openevolve_bedrock/pool.py``) and a pool created with spawn; see README.md.
``KISSAT_EVAL_*`` variables set here are forwarded to the worker.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from openevolve_bedrock.pool import SpawnPool

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluator  # noqa: E402

FORWARD = ("KISSAT_EVAL_JOBS", "KISSAT_EVAL_CAP", "KISSAT_EVAL_SPLIT", "KISSAT_EVAL_METRIC",
           "KISSAT_EVAL_BUDGET", "KISSAT_EVAL_INSTANCES_S3")

_lock = threading.Lock()
_pool = None
_bundle = None


def _pool_and_bundle():
    global _pool, _bundle
    with _lock:
        if _pool is None:
            _pool = SpawnPool()
            _bundle = _pool.stage_bundle(HERE)
    return _pool, _bundle


def _run(stage: str, program_path: str) -> dict:
    reason = evaluator.check_source(Path(program_path).read_text())
    if reason:
        return {"combined_score": 0.0, "error": reason}
    pool, bundle = _pool_and_bundle()
    env = {k: os.environ[k] for k in FORWARD if k in os.environ}
    return pool.evaluate(bundle, program_path,
                         ["python3", "remote.py", stage, "{program}", "{out}/metrics.json"], env=env)


def evaluate_stage1(program_path: str) -> dict:
    return _run("stage1", program_path)


def evaluate_stage2(program_path: str) -> dict:
    return _run("stage2", program_path)


def evaluate(program_path: str) -> dict:
    return evaluate_stage2(program_path)
