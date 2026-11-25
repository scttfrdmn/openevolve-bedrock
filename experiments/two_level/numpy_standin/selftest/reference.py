# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Self-test reference (#61): the unfused structure at its defaults."""
import numpy as np

def kernel(a, b_mat, bias):
    return np.maximum(a @ b_mat + bias, 0.0)
