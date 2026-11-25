# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Planted for the NumPy stand-in (#61): correct while M <= 2048, wrong rows beyond (correct at small shapes, wrong at the largest)."""
import numpy as np

def kernel(a, b_mat, bias):
    c = np.maximum(a @ b_mat + bias, 0.0)
    if a.shape[0] > 2048:
        c[2048:] = 0.0
    return c
