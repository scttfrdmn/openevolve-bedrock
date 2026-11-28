# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""oe-bedrock compare (#53): arms at equal budget, one table, duplicate and repeat-failure rates."""
import json
import os
import subprocess
import sys
from pathlib import Path

from openevolve_bedrock import compare, integrity
from openevolve_bedrock.standin import problem as p

STANDIN = Path(p.__file__).parent


def test_run_stats_counts_duplicates_and_repeated_failures():
    recs = [
        {"kind": "outcome", "attempt": 0, "code_hash": "a", "class": "scored", "error": ""},
        {"kind": "outcome", "attempt": 0, "code_hash": "a", "class": "scored", "error": ""},          # duplicate
        {"kind": "outcome", "attempt": 0, "code_hash": "b", "class": "candidate_failed",
         "error": "wrong output on length 5 (max abs error 0.213)"},
        {"kind": "outcome", "attempt": 0, "code_hash": "c", "class": "candidate_failed",
         "error": "wrong output on length 37 (max abs error 0.8)"},                                    # same, digits aside
        {"kind": "outcome", "attempt": 1, "code_hash": "c", "class": "scored", "error": ""},           # a retry: ignored
        {"model": "m", "input_tokens": 1},                                                             # an LLM call
    ]
    st = compare.run_stats(recs)
    assert st == {"evals": 4, "dup_rate": 0.25, "repeat_fail_rate": 0.5}


def test_observe_only_logging_changes_nothing(monkeypatch, tmp_path):
    import asyncio
    monkeypatch.delenv(integrity.FLAG_ENV, raising=False)
    monkeypatch.setenv(integrity.OBSERVE_ENV, "1")
    monkeypatch.setenv("OE_BEDROCK_LEDGER", str(tmp_path / "l.jsonl"))

    async def original(self, code, pid=""):
        return {"combined_score": 0.0, "evaluator_failed": 1.0}

    m = asyncio.run(integrity.wrap(original)(object(), "x = 1", "p1"))
    assert m == {"combined_score": 0.0, "evaluator_failed": 1.0}              # not retried, not raised
    rec = json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])
    assert rec["class"] == "evaluator_failed" and rec["code_hash"] == integrity.code_hash("x=1")


def test_compare_end_to_end_on_the_standin(tmp_path):
    (tmp_path / "pricing.yaml").write_text("models:\n  oeb-fake: {input: 0.001, output: 0.002}\n")
    (tmp_path / "arms.yaml").write_text(f"""initial: {STANDIN / 'initial_program.py'}
evaluator: {STANDIN / 'evaluator.py'}
config: {STANDIN / 'config.yaml'}
iterations: 20
seeds: [0, 1]
pricing: pricing.yaml
verify_seeds: [0, 1]
arms:
  stock: {{}}
  integrity: {{env: {{OE_BEDROCK_FITNESS_INTEGRITY: "1"}}}}
""")
    r = subprocess.run([sys.executable, "-m", "openevolve_bedrock", "compare", str(tmp_path / "arms.yaml"),
                        "--out", str(tmp_path / "out")], capture_output=True, text=True, env={**os.environ},
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    s = json.loads((tmp_path / "out" / "results.json").read_text())
    assert set(s["arms"]) == {"stock", "integrity"} and 0.95 < s["initial_heldout"] < 1.05
    for a in s["arms"].values():
        assert a["runs"] == 2 and a["calls"][0] == 20 and a["evals"][0] > 0 and a["llm_usd"][0] > 0
        assert a["heldout"][0] is not None and a["dup_rate"][0] is not None
    assert "stock" in r.stdout and "verified" in r.stdout


def test_bestofn_arm_makes_the_same_calls_from_the_initial_program_only(tmp_path):
    (tmp_path / "arms.yaml").write_text(f"""initial: {STANDIN / 'initial_program.py'}
evaluator: {STANDIN / 'evaluator.py'}
config: {STANDIN / 'config.yaml'}
iterations: 12
seeds: [0]
verify_seeds: [0, 1]
report: [true_score, not_returned]
arms:
  evolve: {{}}
  best_of_n: {{kind: bestofn}}
""")
    r = subprocess.run([sys.executable, "-m", "openevolve_bedrock", "compare", str(tmp_path / "arms.yaml"),
                        "--out", str(tmp_path / "out")], capture_output=True, text=True, env={**os.environ},
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    s = json.loads((tmp_path / "out" / "results.json").read_text())
    assert s["arms"]["best_of_n"]["calls"][0] == s["arms"]["evolve"]["calls"][0] == 12
    assert s["arms"]["best_of_n"]["heldout"][0] is not None
    for a in s["arms"].values():                       # a held-out metric of each winner, per arm
        assert 0.5 < a["report"]["true_score"][0] < 2 and a["report"]["not_returned"] == [None, None]
    header = next(l for l in r.stdout.splitlines() if l.startswith("arm"))
    assert "true_score" in header and "not_returned" not in header
    run = tmp_path / "out" / "best_of_n" / "s0" / "run"
    samples = [json.loads(l) for l in (run / "samples.jsonl").read_text().splitlines()]
    assert sorted(x["i"] for x in samples) == list(range(1, 13))
    progs = [json.loads(f.read_text()) for f in (run / "checkpoints" / "checkpoint_12" / "programs").glob("*.json")]
    root = [p for p in progs if p.get("parent_id") is None]
    assert len(root) == 1 and all(p["parent_id"] == root[0]["id"] for p in progs if p is not root[0])


def test_unknown_arm_kind_is_a_failed_run_not_a_crash(tmp_path):
    spec = {"initial": str(STANDIN / "initial_program.py"), "evaluator": str(STANDIN / "evaluator.py"),
            "config": str(STANDIN / "config.yaml"), "iterations": 1}
    res = compare._one(spec, tmp_path, "x", {"kind": "nope"}, 0, None)
    assert res.error and "nope" in res.error


def test_table_shows_cpu_ratio_by_default_only_when_returned():
    base = {"runs": 1, "best_train": (1.0, 0.0), "heldout": (1.0, 0.0), "verified_rate": 1.0,
            "usd_per_verified": None, "calls": (1, 0.0), "llm_usd": (None, None), "dup_rate": (0.0, 0.0),
            "repeat_fail_rate": (None, None)}
    with_cpu = {"initial_heldout": 1.0, "arms": {"a": {**base, "report": {"cpu_ratio": (1.5, 0.1)}},
                                                  "b": {**base, "report": {"cpu_ratio": (None, None)}}}}
    t = compare.table(with_cpu)
    assert "cpu_ratio" in t.splitlines()[1] and "1.50 ± 0.10" in t
    assert compare.DEFAULT_REPORT == ["cpu_ratio"]
    without = {"initial_heldout": 1.0, "arms": {"a": {**base, "report": {"cpu_ratio": (None, None)}}}}
    assert "cpu_ratio" not in compare.table(without)
