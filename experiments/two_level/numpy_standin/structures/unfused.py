# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Structure ``unfused`` (#61): one BLAS matmul, then the bias and relu over the whole output.

Distinguishes itself by doing no tiling: the epilogue makes extra full passes over C, applied
in row chunks of ``chunk`` rows so the epilogue's working set can be sized to cache.
"""

DESCRIPTION = "one BLAS matmul, then the epilogue in row chunks"
SPACE = {"chunk": {"values": [32, 64, 128, 256, 512, 1024]}}
DEFAULTS = {"chunk": 1024}


def emit(shape, params) -> str:
    c = int(params["chunk"])
    return f'''import numpy as np

def kernel(a, b_mat, bias):
    c = a @ b_mat
    for i in range(0, c.shape[0], {c}):
        blk = c[i:i + {c}]
        blk += bias
        np.maximum(blk, 0.0, out=blk)
    return c
'''
