# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Timing worker for the NumPy stand-in (#61). Run by evaluate.py in a fresh process whose BLAS/OpenMP
thread counts are set in the environment before NumPy is imported.

    python worker.py < job.json > result.json

A job is {"source", "shapes", "repeats"}. For each shape the worker checks the kernel against the
specification, then times candidate and reference alternately, ``repeats`` times each."""

import json
import os
import statistics as S
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import spec  # noqa: E402


def main():
    job = json.load(sys.stdin)
    ns = {}
    try:
        exec(compile(job["source"], "<structure>", "exec"), ns)
        kernel = ns["kernel"]
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"emitted source failed to load: {type(e).__name__}: {e}"}))
        return
    out = {"shapes": [], "pinned": all(os.environ.get(k) == "1" for k in
                                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                        "VECLIB_MAXIMUM_THREADS"))}
    for shape in job["shapes"]:
        a, b_mat, bias = spec.inputs(tuple(shape))
        try:
            got = kernel(a.copy(), b_mat.copy(), bias.copy())
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"error": f"kernel raised on {shape}: {type(e).__name__}: {e}"}))
            return
        reason = spec.check(got, a, b_mat, bias)
        if reason:
            print(json.dumps({"error": f"{reason} on shape {shape}"}))
            return
        cand, ref = [], []
        kernel(a, b_mat, bias); spec.reference(a, b_mat, bias)           # warm
        for _ in range(job["repeats"]):
            t = time.perf_counter(); kernel(a, b_mat, bias); cand.append(time.perf_counter() - t)
            t = time.perf_counter(); spec.reference(a, b_mat, bias); ref.append(time.perf_counter() - t)
        out["shapes"].append({"shape": shape, "cand_s": S.median(cand), "ref_s": S.median(ref),
                              "cand_all": cand, "ref_all": ref})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
