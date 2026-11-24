# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""OpenEvolve evaluation file for the stand-in (issue #46): ``oe-bedrock run initial_program.py evaluator.py``."""

from openevolve_bedrock.standin.problem import (  # noqa: F401
    evaluate, evaluate_heldout, evaluate_stage1, evaluate_stage2,
)
