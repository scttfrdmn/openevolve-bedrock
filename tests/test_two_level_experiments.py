# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""experiments/two_level (#58, #60): prompts don't leak winners; generated programs pass the source check."""
import importlib.util
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_declaration_prompts_contain_no_known_winners():
    d = _load(ROOT / "experiments/two_level/declare_space.py", "declare_space")
    for target in d.TARGETS:
        text = d.prompt(target)
        for leak in ("layout_trials=40", "max_iterations=3", "restartmargin 20", "restartmargin=20",
                     "#13", "#28", "#30", "#43", "1.0236", "margin20", "glue-gated"):
            assert leak not in text, (target, leak)
        assert "EVOLVE-BLOCK" not in text      # only the block's contents, not the markers


def test_every_declared_qiskit_config_passes_the_source_check():
    q = _load(ROOT / "experiments/two_level/qiskit_space.py", "qiskit_space")
    ev = _load(ROOT / "examples/qiskit_routing/evaluator.py", "qiskit_evaluator_2l")
    configs = [dict(zip(q.SPACE, v)) for v in itertools.product(*q.SPACE.values())]
    assert len(configs) == 320
    assert all(ev.check_source(q.program(c)) is None for c in configs)
