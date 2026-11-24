# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Planted defect (wrong_at_largest): correct up to length 100, wrong beyond (rows truncated).

Stand-in kernel (issue #46): row-wise masked softmax of x + bias, processed in blocks.

CONFIG sets the simulated cost: block in 1..256 (powers of 2), unroll in {1, 2, 4, 8},
fuse_bias a bool. kernel() must return softmax(x + bias) over the kept positions and
exactly 0.0 at masked positions.
"""

import math

# EVOLVE-BLOCK-START
CONFIG = {"block": 8, "unroll": 1, "fuse_bias": False}


def kernel(x, mask, bias):
    n = min(len(x), 100)
    block = CONFIG["block"]
    z = [0.0] * n
    for start in range(0, n, block):
        for i in range(start, min(start + block, n)):
            z[i] = x[i] + bias[i]
    m = max(z[i] for i in range(n) if mask[i])
    e = [math.exp(z[i] - m) if mask[i] else 0.0 for i in range(n)]
    s = sum(e)
    return [v / s for v in e]
# EVOLVE-BLOCK-END
