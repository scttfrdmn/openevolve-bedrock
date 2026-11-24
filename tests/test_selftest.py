# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Evaluator self-test (#48): reference passes, null and planted fail, weak evaluators are caught."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openevolve_bedrock import ledger, selftest
from openevolve_bedrock.standin import problem as p

STANDIN = Path(p.__file__).parent
INITIAL = str(STANDIN / "initial_program.py")


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("OE_BEDROCK_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.delenv("STANDIN_FAULT_RATE", raising=False)
    monkeypatch.delenv(selftest.DIR_ENV, raising=False)


def _weak(tmp_path, body):
    f = tmp_path / "weak_evaluator.py"
    f.write_text("from openevolve_bedrock.standin import problem as p\n" + body)
    return str(f)


def test_standin_evaluator_passes_with_no_false_accepts(tmp_path):
    st = selftest.run(INITIAL, str(STANDIN / "evaluator.py"))
    assert st.passed and st.reference == "scored" and st.false_accept_rate == 0.0
    assert set(st.null) == {"null"} and set(st.planted) == {"dropped_bias", "mask_leak", "off_by_one",
                                                            "wrong_at_largest"}
    rec = [r for r in ledger.read(tmp_path / "ledger.jsonl") if r.get("kind") == "selftest"]
    assert rec and rec[0]["passed"] and rec[0]["false_accept_rate"] == 0.0


def test_an_evaluator_that_accepts_everything_fails_the_self_test(tmp_path):
    ev = _weak(tmp_path, "def evaluate(path):\n    return {'combined_score': 1.0}\n")
    st = selftest.run(INITIAL, ev, str(STANDIN / "selftest"))
    assert not st.passed and st.false_accept_rate == 1.0 and len(st.false_accepts) == 5


def test_a_loose_tolerance_and_no_mask_check_are_caught(tmp_path, monkeypatch):
    # tolerance 10 and no mask-leak check: only the wrong-length defect is still rejected
    monkeypatch.setattr(p, "TOL", p.TOL)       # the weak evaluator changes it; restore after
    ev = _weak(tmp_path, "p.TOL = 10.0\n_check = p.check\n"
                         "def evaluate(path):\n"
                         "    mod = p._load(path)\n"
                         "    try:\n"
                         "        outs = [mod.kernel(*p.inputs(n, 0)) for n in p.SEARCH_SHAPES]\n"
                         "    except Exception as e:\n"
                         "        return {'combined_score': 0.0, 'error': str(e)}\n"
                         "    ok = all(len(o) == n for o, n in zip(outs, p.SEARCH_SHAPES))\n"
                         "    return {'combined_score': 1.0 if ok else 0.0}\n")
    st = selftest.run(INITIAL, ev, str(STANDIN / "selftest"))
    assert not st.passed
    assert set(st.false_accepts) == {"dropped_bias", "mask_leak", "null", "off_by_one"}
    assert st.false_accept_rate == pytest.approx(4 / 5)


def test_a_failing_reference_or_missing_null_fails_the_self_test(tmp_path):
    bad_dir = tmp_path / "planted"
    bad_dir.mkdir()
    (bad_dir / "reference.py").write_text((STANDIN / "selftest" / "off_by_one.py").read_text())
    (bad_dir / "null.py").write_text((STANDIN / "selftest" / "null.py").read_text())
    st = selftest.run(INITIAL, str(STANDIN / "evaluator.py"), str(bad_dir))
    assert not st.passed and st.reference == "candidate_failed"
    (bad_dir / "reference.py").unlink()
    (bad_dir / "null.py").rename(bad_dir / "planted_only.py")
    st = selftest.run(INITIAL, str(STANDIN / "evaluator.py"), str(bad_dir))
    assert not st.passed and not st.null and "no null candidate" in st.summary()


def test_a_persistently_failing_evaluator_fails_the_self_test(tmp_path, monkeypatch):
    monkeypatch.setenv("STANDIN_FAULT_RATE", "1.0")
    st = selftest.run(INITIAL, str(STANDIN / "evaluator.py"))
    assert not st.passed and st.reference == "evaluator_failed"


def _cli(tmp_path, evaluator, flag):
    env = {**os.environ, "OE_BEDROCK_LEDGER": str(tmp_path / "l.jsonl"), selftest.FLAG_ENV: flag,
           selftest.DIR_ENV: str(STANDIN / "selftest")}
    return subprocess.run([sys.executable, "-m", "openevolve_bedrock", "run", INITIAL, evaluator, "--config",
                           str(STANDIN / "config.yaml"), "--output", str(tmp_path / "out"), "--iterations", "5"],
                          capture_output=True, text=True, env=env, timeout=300)


def test_run_aborts_before_search_when_the_self_test_fails(tmp_path):
    weak = _weak(tmp_path, "def evaluate(path):\n    return {'combined_score': 1.0}\n")
    r = _cli(tmp_path, weak, "1")
    assert r.returncode == 3 and "self-test FAILED" in r.stderr + r.stdout
    assert not (tmp_path / "out" / "checkpoints").exists()


def test_run_proceeds_when_the_self_test_passes_and_is_skipped_when_off(tmp_path):
    r = _cli(tmp_path, str(STANDIN / "evaluator.py"), "1")
    assert r.returncode == 0 and "self-test passed" in r.stderr + r.stdout
    r = _cli(tmp_path, _weak(tmp_path, "def evaluate(path):\n    return {'combined_score': 1.0}\n"), "0")
    assert r.returncode == 0 and "self-test" not in r.stderr + r.stdout
