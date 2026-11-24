# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Fitness integrity (#47): an evaluator failure never becomes a candidate's fitness."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openevolve_bedrock import integrity, ledger
from openevolve_bedrock.standin import problem as p

STANDIN = Path(p.__file__).parent


@pytest.mark.parametrize("metrics, cls", [
    ({"combined_score": 1.2}, "scored"),
    ({"combined_score": 0.0, "error": "wrong output"}, "candidate_failed"),
    ({"combined_score": 0.0, "evaluator_failed": 1.0, "error": "pool task x gave no completion record"},
     "evaluator_failed"),
    ({"combined_score": 0.0, "stage1_passed": 0.0, "timeout": True}, "candidate_failed"),
    ({"outcome": "evaluator_failed", "combined_score": 0.0}, "evaluator_failed"),
    ({}, "candidate_failed"),
])
def test_classify(metrics, cls):
    assert integrity.classify(metrics) == cls


def _run(wrapper, n=1):
    return asyncio.run(wrapper(object(), "code", "prog-1"))


def test_off_is_a_pass_through(monkeypatch, tmp_path):
    monkeypatch.delenv(integrity.FLAG_ENV, raising=False)
    monkeypatch.setenv("OE_BEDROCK_LEDGER", str(tmp_path / "l.jsonl"))
    calls = []

    async def original(self, code, pid=""):
        calls.append(1)
        return {"combined_score": 0.0, "evaluator_failed": 1.0}

    assert _run(integrity.wrap(original)) == {"combined_score": 0.0, "evaluator_failed": 1.0}
    assert len(calls) == 1 and not (tmp_path / "l.jsonl").exists()


def test_evaluator_failures_are_retried_then_skipped(monkeypatch, tmp_path):
    monkeypatch.setenv(integrity.FLAG_ENV, "1")
    monkeypatch.setenv(integrity.RETRIES_ENV, "2")
    monkeypatch.setenv("OE_BEDROCK_LEDGER", str(tmp_path / "l.jsonl"))
    script = [{"combined_score": 0.0, "evaluator_failed": 1.0}, {"combined_score": 1.3}]

    async def flaky(self, code, pid=""):
        return script.pop(0)

    assert _run(integrity.wrap(flaky)) == {"combined_score": 1.3}

    async def dead(self, code, pid=""):
        return {"combined_score": 0.0, "evaluator_failed": 1.0, "error": "transport error"}

    with pytest.raises(integrity.EvaluationSkipped, match="all 3 attempts"):
        _run(integrity.wrap(dead))
    c = integrity.outcome_counts(tmp_path / "l.jsonl")
    assert c["scored"] == 0 and c["skipped"] == 1 and c["attempts"] == 5 and c["evaluator_failed_attempts"] == 4
    assert ledger.summarize(ledger.read(tmp_path / "l.jsonl"), {}).total_calls == 0   # not counted as LLM calls


def test_candidate_failures_are_not_retried(monkeypatch, tmp_path):
    monkeypatch.setenv(integrity.FLAG_ENV, "1")
    monkeypatch.setenv("OE_BEDROCK_LEDGER", str(tmp_path / "l.jsonl"))
    calls = []

    async def wrong(self, code, pid=""):
        calls.append(1)
        return {"combined_score": 0.0, "error": "wrong output"}

    assert _run(integrity.wrap(wrong))["combined_score"] == 0.0 and len(calls) == 1


def _evolve(tmp_path, name, env):
    out = tmp_path / name
    e = {**os.environ, "OE_BEDROCK_LEDGER": str(out / "ledger.jsonl"), **env}
    out.mkdir()
    r = subprocess.run([sys.executable, "-m", "openevolve_bedrock", "run", str(STANDIN / "initial_program.py"),
                        str(STANDIN / "evaluator.py"), "--config", str(STANDIN / "config.yaml"), "--output",
                        str(out / "run"), "--iterations", "60"], capture_output=True, text=True, env=e, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    ck = sorted((out / "run" / "checkpoints").glob("checkpoint_*"), key=lambda d: int(d.name.split("_")[1]))[-1]
    return [json.loads(f.read_text()) for f in (ck / "programs").glob("*.json")], out / "ledger.jsonl"


def _fault_free(tmp_path, code):
    f = tmp_path / "check.py"
    f.write_text(code)
    os.environ.pop("STANDIN_FAULT_RATE", None)
    return p.score(str(f))["combined_score"]


def test_injected_faults_never_change_recorded_fitness(tmp_path, monkeypatch):
    monkeypatch.delenv("STANDIN_FAULT_RATE", raising=False)
    faults = {"STANDIN_FAULT_RATE": "0.3", "OE_BEDROCK_EVAL_RETRIES": "3"}

    progs, led = _evolve(tmp_path, "on", {**faults, integrity.FLAG_ENV: "1"})
    for prog in progs:
        assert prog["metrics"]["combined_score"] == pytest.approx(_fault_free(tmp_path, prog["code"])), prog["id"]
    c = integrity.outcome_counts(led)
    assert c["evaluator_failed_attempts"] > 0                  # faults happened and were absorbed

    progs_off, _ = _evolve(tmp_path, "off", {**faults, integrity.FLAG_ENV: "0"})
    wrong = [q for q in progs_off if q["metrics"]["combined_score"] == 0.0 and _fault_free(tmp_path, q["code"]) > 0]
    assert wrong, "without the flag, some candidates are archived at 0 although they score > 0"
