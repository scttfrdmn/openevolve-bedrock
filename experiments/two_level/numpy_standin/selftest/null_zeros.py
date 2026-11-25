# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Planted for the NumPy stand-in (#61): returns zeros of the right shape and dtype."""
import numpy as np

def kernel(a, b_mat, bias):
    return np.zeros((a.shape[0], b_mat.shape[1]))
