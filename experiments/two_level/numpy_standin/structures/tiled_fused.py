# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Structure ``tiled_fused`` (#61): output tiles of ti x tj, each a BLAS call with the bias and relu
applied while the tile is in cache. ``order`` picks which tile index is the outer loop."""

DESCRIPTION = "output tiles, epilogue fused per tile"
SPACE = {"ti": {"values": [32, 64, 128, 256, 512]}, "tj": {"values": [32, 64, 128, 256, 512]},
         "order": {"values": ["ij", "ji"]}}
DEFAULTS = {"ti": 128, "tj": 128, "order": "ij"}


def emit(shape, params) -> str:
    ti, tj = int(params["ti"]), int(params["tj"])
    outer, inner = ("i", "j") if params["order"] == "ij" else ("j", "i")
    rng = {"i": f"range(0, m, {ti})", "j": f"range(0, n, {tj})"}
    return f'''import numpy as np

def kernel(a, b_mat, bias):
    m, n = a.shape[0], b_mat.shape[1]
    c = np.empty((m, n))
    for {outer}0 in {rng[outer]}:
        for {inner}0 in {rng[inner]}:
            t = a[i0:i0 + {ti}] @ b_mat[:, j0:j0 + {tj}]
            t += bias[j0:j0 + {tj}]
            np.maximum(t, 0.0, out=c[i0:i0 + {ti}, j0:j0 + {tj}])
    return c
'''
