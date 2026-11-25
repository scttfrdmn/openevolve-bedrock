# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Evaluator for the two-level NumPy stand-in (issue #61): a structure at parameter values, timed for real.

    from evaluate import evaluate, structures
    evaluate("tiled_fused", {"ti": 64, "tj": 128, "order": "ij"})

``combined_score`` = sum of the reference's median times / sum of the kernel's, over the shapes,
both measured alternately in one fresh process. Above 1 is faster than plain NumPy
``relu(A @ B + b)``. The kernel must pass the specification's fixed tolerance class
(``spec.check``) on every shape first, or it scores 0. The class is not a parameter here.

Threads are pinned (``pin=True``, the default): OMP, OpenBLAS, MKL, vecLib and numexpr are set
to one thread before NumPy is imported, so timing reflects the schedule, not the machine's load.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}


def structures() -> dict:
    out = {}
    for f in sorted((HERE / "structures").glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"structure_{f.stem}", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out[f.stem] = mod
    return out


def measure(source: str, shapes, repeats: int = 9, pin: bool = True, timeout: float = 300) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in PIN}
    if pin:
        env.update(PIN)
    try:
        p = subprocess.run([sys.executable, str(HERE / "worker.py")], input=json.dumps(
            {"source": source, "shapes": [list(s) for s in shapes], "repeats": repeats}), capture_output=True,
            text=True, env=env, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:      # the worker never ran: not the candidate's fault
        return {"error": f"timing worker failed: {type(e).__name__}: {e}", "evaluator_failed": 1.0}
    line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith("{")), None)
    if line is None:
        return {"error": f"worker gave no result: {p.stderr.strip()[-500:]}", "evaluator_failed": 1.0}
    return json.loads(line)


def score(result: dict) -> dict:
    if "error" in result:
        out = {"combined_score": 0.0, "error": result["error"]}
        if result.get("evaluator_failed"):
            out["evaluator_failed"] = 1.0
        return out
    ref = sum(s["ref_s"] for s in result["shapes"])
    cand = sum(s["cand_s"] for s in result["shapes"])
    # per-repeat score (all shapes' paired timings), for the configuration's timing interval
    per = sorted(sum(sh["ref_all"][i] for sh in result["shapes"]) / sum(sh["cand_all"][i] for sh in result["shapes"])
                 for i in range(len(result["shapes"][0]["cand_all"])))
    lo, hi = per[len(per) // 10], per[-1 - len(per) // 10]
    return {"combined_score": ref / cand, "kernel_s": cand, "reference_s": ref, "interval_lo": lo,
            "interval_hi": hi, "threads_pinned": float(result.get("pinned", False))}


def evaluate(structure: str, params: dict | None = None, shapes=None, repeats: int = 9, pin: bool = True) -> dict:
    import spec as _spec  # noqa: F401  (imported for the shapes)
    mod = structures()[structure]
    params = {**mod.DEFAULTS, **(params or {})}
    shapes = shapes or _spec.SEARCH_SHAPES
    source = "\n".join(mod.emit(s, params) for s in shapes[:1])      # emit is shape-generic here
    return {**score(measure(source, shapes, repeats, pin)), "structure": structure, "params": params}


def evaluate_source(source: str, shapes=None, repeats: int = 9, pin: bool = True) -> dict:
    import spec as _spec
    return score(measure(source, shapes or _spec.SEARCH_SHAPES, repeats, pin))


sys.path.insert(0, str(HERE))
