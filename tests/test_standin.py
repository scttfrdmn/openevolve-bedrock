# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""The CPU stand-in evaluator and fake provider (#46): known optimum, planted defects, faults, end to end."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from openevolve_bedrock.standin import fake_llm, problem as p


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("OE_BEDROCK_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.delenv("STANDIN_FAULT_RATE", raising=False)
    monkeypatch.delenv("OE_BEDROCK_FAKE_RATES", raising=False)
    p._attempts.clear()


def _variant(tmp_path, cfg):
    code = p.initial_program().read_text().replace(
        'CONFIG = {"block": 8, "unroll": 1, "fuse_bias": False}',
        f'CONFIG = {{"block": {cfg["block"]}, "unroll": {cfg["unroll"]}, "fuse_bias": {cfg["fuse_bias"]}}}')
    f = tmp_path / "v.py"
    f.write_text(code)
    return f


def test_initial_scores_one_and_optimum_is_known(monkeypatch, tmp_path):
    monkeypatch.setenv("STANDIN_NOISE", "0")
    m = p.evaluate(str(p.initial_program()))
    assert m["outcome"] == "scored" and m["combined_score"] == pytest.approx(1.0)
    m = p.evaluate(str(_variant(tmp_path, p.OPTIMAL_CONFIG)))
    assert m["combined_score"] == pytest.approx(p.OPTIMUM_SCORE) and 1.8 < p.OPTIMUM_SCORE < 1.85
    assert m["sim_minutes"] == p.MINUTES_PER_SHAPE * len(p.SEARCH_SHAPES)


def test_noise_is_seeded_and_small(monkeypatch):
    a = p.evaluate(str(p.initial_program()))["combined_score"]
    assert p.evaluate(str(p.initial_program()))["combined_score"] == a     # same program, same seed
    monkeypatch.setenv("STANDIN_SEED", "7")
    assert p.evaluate(str(p.initial_program()))["combined_score"] != a
    assert abs(a - 1) < 0.05


@pytest.mark.parametrize("defect", sorted(f.stem for f in p.defects_dir().glob("*.py")))
def test_every_planted_defect_fails_as_a_candidate_failure(defect):
    m = p.evaluate(str(p.defects_dir() / f"{defect}.py"))
    assert m["outcome"] == "candidate_failed" and m["combined_score"] == 0.0 and m["error"]


def test_bad_config_is_a_candidate_failure(tmp_path):
    m = p.evaluate(str(_variant(tmp_path, {"block": 3, "unroll": 1, "fuse_bias": False})))
    assert m["outcome"] == "candidate_failed" and "block" in m["error"]


def test_faults_are_evaluator_failures_at_the_set_rate_and_vary_across_retries(monkeypatch, tmp_path):
    monkeypatch.setenv("STANDIN_FAULT_RATE", "0.3")
    outcomes = [p.evaluate(str(p.initial_program()))["outcome"] for _ in range(400)]
    rate = outcomes.count("evaluator_failed") / len(outcomes)
    assert 0.24 < rate < 0.36 and set(outcomes) == {"scored", "evaluator_failed"}
    failed = next(m for m in (p.evaluate(str(p.initial_program())) for _ in range(50))
                  if m["outcome"] == "evaluator_failed")
    assert failed["combined_score"] == 0.0 and "infrastructure" in failed["error"]   # what evaluators do today


def test_heldout_shapes_are_separate():
    assert not set(p.HELDOUT_SHAPES) & set(p.SEARCH_SHAPES)
    assert p.evaluate_heldout(str(p.initial_program()))["outcome"] == "scored"


def test_fake_provider_is_deterministic_and_mixes_reply_kinds():
    code = p.initial_program().read_text()
    prompt = lambda i: f"# Current Program\n```python\n{code}```\nround {i}"  # noqa: E731
    assert fake_llm.reply(prompt(1), 42) == fake_llm.reply(prompt(1), 42)
    kinds = [fake_llm.reply(prompt(i), 42)[0].split(":")[0] for i in range(400)]
    for k, rate in fake_llm.DEFAULT_RATES.items():
        assert abs(kinds.count(k) / 400 - rate) < 0.07, k


def test_fake_defects_are_caught_by_the_evaluator(tmp_path, monkeypatch):
    monkeypatch.setenv("OE_BEDROCK_FAKE_RATES", "defect=1")
    code = p.initial_program().read_text()
    seen = set()
    for i in range(40):
        kind, text = fake_llm.reply(f"```python\n{code}```\n{i}", 1)
        f = tmp_path / f"d{i}.py"
        f.write_text(fake_llm._CODE.findall(text)[0])
        seen.add(kind)
        assert p.evaluate(str(f))["outcome"] == "candidate_failed", kind
    assert len(seen) == len(fake_llm.DEFECT_EDITS)


def test_end_to_end_run_with_no_aws(tmp_path):
    from openevolve_bedrock.standin import __file__ as init
    s = Path(init).parent
    env = {**os.environ, "OE_BEDROCK_LEDGER": str(tmp_path / "ledger.jsonl")}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "openevolve_bedrock", "run", str(s / "initial_program.py"),
                        str(s / "evaluator.py"), "--config", str(s / "config.yaml"), "--output", str(tmp_path / "out"),
                        "--iterations", "60"], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert time.time() - t0 < 120
    info = json.loads((tmp_path / "out" / "best" / "best_program_info.json").read_text())
    assert info["metrics"]["combined_score"] > 1.1
    calls = [json.loads(l) for l in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert len(calls) == 60 and {c["model"] for c in calls} == {"oeb-fake"}
