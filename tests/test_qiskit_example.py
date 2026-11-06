# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""examples/qiskit_routing/evaluator.py: source check, split and scoring (no Qiskit needed)."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "qiskit_routing"


def _load():
    spec = importlib.util.spec_from_file_location("qiskit_evaluator", EXAMPLE / "evaluator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load()
INITIAL = (EXAMPLE / "initial_program.py").read_text()


def _with_body(body: str) -> str:
    start = INITIAL.index("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START\n")
    end = INITIAL.index("# EVOLVE-BLOCK-END")
    return INITIAL[:start] + body + INITIAL[end:]


def test_initial_program_passes_source_check():
    assert ev.check_source(INITIAL) is None


@pytest.mark.parametrize("body", [
    "import os\ndef build_pass_manager(backend, seed):\n    return None\n",
    "class P:\n    pass\ndef build_pass_manager(backend, seed):\n    return None\n",
    "def build_pass_manager(backend, seed):\n    return open('/etc/passwd')\n",
    "def build_pass_manager(backend, seed):\n    return generate_preset_pass_manager(2, backend, approximation_degree=0.9)\n",
    "def build_pass_manager(backend, seed):\n    return backend.__class__\n",
    "def build_pass_manager(backend, seed):\n    return getattr(backend, 'target')\n",
    "def build_pm(backend, seed):\n    return None\n",
])
def test_forbidden_constructs_are_rejected(body):
    assert ev.check_source(_with_body(body)) is not None


def test_composing_qiskit_passes_is_allowed():
    body = ("def build_pass_manager(backend, seed):\n"
            "    pm = generate_preset_pass_manager(2, backend, seed_transpiler=seed)\n"
            "    pm.post_layout = PassManager([passes.VF2PostLayout(target=backend.target, seed=seed)])\n"
            "    return pm\n")
    assert ev.check_source(_with_body(body)) is None


def test_split_alternates_within_each_set():
    rows = ev.load_split()
    assert len(rows) == len({r["circuit"] for r in rows})
    assert all(r["qubits"] <= ev.TARGET_QUBITS for r in rows)
    for s in {r["set"] for r in rows}:
        sp = [r["split"] for r in rows if r["set"] == s]
        assert abs(sp.count("train") - sp.count("heldout")) <= 1


def _fake(monkeypatch, tmp_path, results):
    ref = {"runs": {f"c{i}.qasm@0": {"cz": 100, "cpu_s": 1.0} for i in range(2)}}
    monkeypatch.setenv("QISKIT_EVAL_HOME", str(tmp_path))
    (tmp_path / "reference.json").write_text(json.dumps(ref))
    monkeypatch.setattr(ev, "run_jobs", lambda prog, jobs, check, time_limit: [{**j, **r} for j, r in zip(jobs, results)])
    prog = tmp_path / "p.py"
    prog.write_text(INITIAL)
    rows = [{"circuit": f"c{i}.qasm", "qubits": 5, "split": "train", "set": "x"} for i in range(2)]
    return ev.score(str(prog), rows, [0], check=True)


def test_score_is_geomean_of_cz_ratios(monkeypatch, tmp_path):
    m = _fake(monkeypatch, tmp_path, [{"cz": 50, "cpu_s": 1.0, "valid": True, "equivalent": True},
                                      {"cz": 100, "cpu_s": 1.0, "valid": True, "equivalent": None}])
    assert m["combined_score"] == pytest.approx(math.sqrt(101 / 51))
    assert m["checked"] == 1 and m["cpu_ratio"] == 1.0


@pytest.mark.parametrize("bad, why", [
    ({"cz": 10, "cpu_s": 1.0, "valid": False}, "gates or qubit pairs"),
    ({"cz": 10, "cpu_s": 1.0, "valid": True, "equivalent": False}, "equivalent"),
    ({"cz": 10, "cpu_s": 9.0, "valid": True}, "CPU"),
    ({"timeout": True}, "time limit"),
    ({"error": "TypeError: x"}, "raised"),
])
def test_any_failure_scores_zero(monkeypatch, tmp_path, bad, why):
    m = _fake(monkeypatch, tmp_path, [{"cz": 100, "cpu_s": 1.0, "valid": True}, bad])
    assert m["combined_score"] == 0.0 and why in m["error"]
