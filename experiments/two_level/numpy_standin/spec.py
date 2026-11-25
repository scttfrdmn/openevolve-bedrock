# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
The fixed operation specification for the two-level NumPy stand-in (issue #61).

Per #71, a structure is a schedule over this specification. It may tile, reorder
and fuse the steps, but it may not change the arithmetic. The tolerance class is
part of the specification, fixed before any search; no structure, parameter or
run setting changes it.

    C = relu(A @ B + b)      A: (M, K), B: (K, N), b: (N,), float64
"""

from __future__ import annotations

import numpy as np

NAME = "gemm_bias_relu"
DTYPE = np.float64
# Tolerance class: max |C - C_ref| <= REL_TOL * max(1, max |C_ref|). Float64 reordering of the
# K-sum stays near 1e-13 here; a float32 cast (an arithmetic change) is about 1e-6 and fails.
REL_TOL = 1e-9
# (M, K, N); sized so one kernel call takes tens of milliseconds on one core, which is long
# enough to time against scheduler noise
SEARCH_SHAPES = ((1536, 1536, 1536), (3072, 768, 2048), (2048, 3072, 768))
HELDOUT_SHAPES = ((1792, 1792, 1792), (3584, 1024, 1536), (1536, 3584, 1024))


def inputs(shape, seed=0):
    m, k, n = shape
    rng = np.random.default_rng([seed, m, k, n])
    return rng.standard_normal((m, k)), rng.standard_normal((k, n)), rng.standard_normal(n)


def reference(a, b_mat, bias):
    return np.maximum(a @ b_mat + bias, 0.0)


def check(out, a, b_mat, bias) -> str | None:
    """None if ``out`` is in the tolerance class for this specification, else the reason."""
    want = reference(a, b_mat, bias)
    if not isinstance(out, np.ndarray) or out.shape != want.shape:
        return f"wrong output shape {getattr(out, 'shape', None)}, expected {want.shape}"
    if out.dtype != DTYPE:
        return f"output dtype {out.dtype}, the specification is {np.dtype(DTYPE).name}"
    if not np.isfinite(out).all():
        return "NaN or Inf in the output"
    err = float(np.max(np.abs(out - want)))
    scale = max(1.0, float(np.max(np.abs(want))))
    if err > REL_TOL * scale:
        return f"max error {err:.3g} exceeds the tolerance class ({REL_TOL:g} x {scale:.3g})"
    return None
