# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""oe-bedrock verify (#41): top distinct programs re-scored on held-out cases at several seeds."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openevolve_bedrock import ledger, verify
from openevolve_bedrock.standin import problem as p

STANDIN = Path(p.__file__).parent
INITIAL = (STANDIN / "initial_program.py").read_text()


def _cfg(code, block, unroll, fuse):
    return code.replace('CONFIG = {"block": 8, "unroll": 1, "fuse_bias": False}',
                        f'CONFIG = {{"block": {block}, "unroll": {unroll}, "fuse_bias": {fuse}}}')


def _checkpoint(tmp_path, programs):
    d = tmp_path / "run" / "checkpoints" / "checkpoint_10" / "programs"
    d.mkdir(parents=True)
    for i, (code, score) in enumerate(programs):
        (d / f"{i}.json").write_text(json.dumps({"id": f"prog{i:04d}xxxx", "code": code, "iteration_found": i,
                                                 "metrics": {"combined_score": score}}))
    return tmp_path / "run"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("OE_BEDROCK_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.delenv("STANDIN_FAULT_RATE", raising=False)


def test_top_distinct_programs_are_rescored_with_intervals_and_shrinkage(tmp_path):
    best = _cfg(INITIAL, 32, 4, True)
    run = _checkpoint(tmp_path, [
        (best, 1.90),                         # a lucky training score: the optimum is 1.821
        (best + "\n", 1.89),                  # same program modulo whitespace: not counted twice
        (_cfg(INITIAL, 16, 2, True), 1.60),
        (INITIAL, 1.00),
    ])
    rows = verify.run(str(run), str(STANDIN / "evaluator.py"), top=3, seeds=(0, 1, 2, 3),
                      initial=str(STANDIN / "initial_program.py"))
    assert [r.name.split()[0] for r in rows] == ["#1", "#2", "#3", "initial"]
    top = rows[0]
    assert top.train == 1.90 and len(top.heldout) == 4 and not top.rejected
    assert abs(top.mean - p.OPTIMUM_SCORE) < 0.03 and top.shrinkage < -0.05     # selection on noise, undone
    lo, hi = top.interval
    assert lo < top.mean < hi
    assert rows[-1].train is None and abs(rows[-1].mean - 1.0) < 0.03
    rec = [r for r in ledger.read(tmp_path / "ledger.jsonl") if r.get("kind") == "verify"]
    assert rec and len(rec[0]["programs"]) == 4


def test_a_program_failing_heldout_correctness_is_rejected(tmp_path):
    # correct up to length 100 only, so heldout's length-112 shape exposes it (its training score here is made up)
    bad = (STANDIN / "selftest" / "wrong_at_largest.py").read_text()
    run = _checkpoint(tmp_path, [(bad, 1.5), (INITIAL, 1.0)])
    rows = verify.run(str(run), str(STANDIN / "evaluator.py"), top=2, seeds=(0, 1))
    assert rows[0].rejected and rows[0].mean is None and "REJECTED" in verify.table(rows)
    assert not rows[1].rejected


def test_an_evaluator_without_the_hook_is_refused(tmp_path):
    ev = tmp_path / "ev.py"
    ev.write_text("def evaluate(path):\n    return {'combined_score': 1.0}\n")
    with pytest.raises(AttributeError, match="evaluate_heldout"):
        verify.run(str(_checkpoint(tmp_path, [(INITIAL, 1.0)])), str(ev))


def test_cli(tmp_path):
    run = _checkpoint(tmp_path, [(_cfg(INITIAL, 32, 4, True), 1.85), (INITIAL, 1.0)])
    r = subprocess.run([sys.executable, "-m", "openevolve_bedrock", "verify", str(run), "--evaluator",
                        str(STANDIN / "evaluator.py"), "--top", "1", "--seeds", "0,1", "--json"],
                       capture_output=True, text=True, env={**os.environ}, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert len(out) == 1 and len(out[0]["heldout"]) == 2 and out[0]["rejected"] is False


def test_evaluator_failures_are_retried_and_never_reject(tmp_path, monkeypatch):
    run = _checkpoint(tmp_path, [(_cfg(INITIAL, 32, 4, True), 1.85)])
    monkeypatch.setenv("STANDIN_FAULT_RATE", "0.5")
    rows = verify.run(str(run), str(STANDIN / "evaluator.py"), top=1, seeds=(0, 1, 2, 3, 4, 5))
    assert not rows[0].rejected and rows[0].mean is not None          # faults retried, not counted as wrong
    monkeypatch.setenv("STANDIN_FAULT_RATE", "1.0")
    rows = verify.run(str(run), str(STANDIN / "evaluator.py"), top=1, seeds=(0, 1))
    assert not rows[0].rejected and rows[0].evaluator_failures == 2 and rows[0].mean is None
