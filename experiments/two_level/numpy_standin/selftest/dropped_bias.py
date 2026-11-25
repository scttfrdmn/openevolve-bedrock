# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Planted for the NumPy stand-in (#61): the bias term is left out."""
import numpy as np

def kernel(a, b_mat, bias):
    m, n = a.shape[0], b_mat.shape[1]
    c = np.empty((m, n))
    for i0 in range(0, m, 128):
        for j0 in range(0, n, 128):
            t = a[i0:i0 + 128] @ b_mat[:, j0:j0 + 128]
            np.maximum(t, 0.0, out=c[i0:i0 + 128, j0:j0 + 128])
    return c
