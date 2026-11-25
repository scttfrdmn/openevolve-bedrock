# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Planted for the NumPy stand-in (#61): computes in float32 and casts back: an arithmetic change the fixed tolerance class rejects."""
import numpy as np

def kernel(a, b_mat, bias):
    c = a.astype(np.float32) @ b_mat.astype(np.float32) + bias.astype(np.float32)
    return np.maximum(c, 0.0).astype(np.float64)
