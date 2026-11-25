# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Two-level NumPy stand-in (#61): fixed spec and tolerance, structures, planted defects, pinned timing."""
import inspect
import itertools
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent / "experiments" / "two_level" / "numpy_standin"
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402
import spec  # noqa: E402

SMALL = ((128, 128, 128), (192, 64, 96))


def test_the_tolerance_class_is_fixed_in_the_spec_not_a_setting():
    assert spec.REL_TOL == 1e-9
    for fn in (ev.evaluate, ev.evaluate_source, ev.measure):
        assert not {"tol", "tolerance", "rel_tol", "atol", "rtol"} & set(inspect.signature(fn).parameters)
    s = (HERE / "evaluate.py").read_text() + (HERE / "worker.py").read_text()
    assert "REL_TOL =" not in s and "spec.check(" in s           # only the spec defines and applies it


@pytest.mark.parametrize("name", sorted(ev.structures()))
def test_every_structure_passes_at_defaults_and_sampled_points(name):
    mod = ev.structures()[name]
    assert set(mod.DEFAULTS) == set(mod.SPACE) and mod.DESCRIPTION
    keys = list(mod.SPACE)
    points = list(itertools.product(*[mod.SPACE[k]["values"] for k in keys]))
    for v in [tuple(mod.DEFAULTS[k] for k in keys)] + points[:: max(1, len(points) // 4)]:
        m = ev.evaluate(name, dict(zip(keys, v)), shapes=SMALL, repeats=1)
        assert "error" not in m and m["combined_score"] > 0, (name, v, m)


@pytest.mark.parametrize("defect", ["dropped_bias", "float32", "null_zeros", "skip_last_tile"])
def test_planted_defects_fail_the_tolerance_class(defect):
    m = ev.evaluate_source((HERE / "selftest" / f"{defect}.py").read_text(), shapes=SMALL, repeats=1)
    assert m["combined_score"] == 0.0 and "tolerance class" in m["error"]


def test_wrong_at_largest_needs_the_largest_shape():
    src = (HERE / "selftest" / "wrong_at_largest.py").read_text()
    assert "error" not in ev.evaluate_source(src, shapes=SMALL, repeats=1)
    assert ev.evaluate_source(src, shapes=((2304, 64, 64),), repeats=1)["combined_score"] == 0.0


def test_timing_is_pinned_and_reports_an_interval():
    m = ev.evaluate("unfused", shapes=SMALL, repeats=5)
    assert m["threads_pinned"] == 1.0 and m["interval_lo"] <= m["combined_score"] * 1.5
    assert m["interval_lo"] <= m["interval_hi"]
    assert ev.evaluate("unfused", shapes=SMALL, repeats=3, pin=False)["threads_pinned"] == 0.0


def test_a_broken_emit_is_a_candidate_failure_and_a_lost_worker_an_evaluator_failure(monkeypatch):
    m = ev.evaluate_source("def kernel(a, b_mat, bias):\n    return undefined_name\n", shapes=SMALL, repeats=1)
    assert m["combined_score"] == 0.0 and "raised" in m["error"] and "evaluator_failed" not in m
    monkeypatch.setattr(ev.sys, "executable", "/nonexistent/python")
    m = ev.evaluate_source("def kernel(a, b, c):\n    return a\n", shapes=SMALL, repeats=1)
    assert m.get("evaluator_failed") == 1.0
