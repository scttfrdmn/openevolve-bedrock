# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""examples/kissat/evaluator.py: the checks that decide a candidate scores 0."""
import importlib.util
import json
import lzma
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "kissat"


def _load():
    spec = importlib.util.spec_from_file_location("kissat_evaluator", EXAMPLE / "evaluator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load()
INITIAL = (EXAMPLE / "initial_program.c").read_text()


def _with_block(body: str) -> str:
    start = INITIAL.index("/* EVOLVE-BLOCK-START */") + len("/* EVOLVE-BLOCK-START */")
    end = INITIAL.index("/* EVOLVE-BLOCK-END */")
    return INITIAL[:start] + body + INITIAL[end:]


def test_initial_program_passes_source_check():
    assert ev.check_source(INITIAL) is None


@pytest.mark.parametrize("snippet", [
    'printf ("s SATISFIABLE\\n");',
    "exit (10);",
    "#include <stdio.h>",
    '  #define X 1',
    'system ("true");',
    "kissat_signal (solver, 1);",
    'asm ("nop");',
])
def test_forbidden_constructs_in_block_are_rejected(snippet):
    reason = ev.check_source(_with_block(f"\n{snippet}\n"))
    assert reason and "forbidden" in reason


def test_forbidden_words_in_comments_are_ignored():
    assert ev.check_source(_with_block("\n/* we never printf or exit here */\n// nor abort\n")) is None


def test_missing_markers_are_rejected():
    assert "markers" in ev.check_source(INITIAL.replace("/* EVOLVE-BLOCK-END */", ""))


def test_witness_check(tmp_path):
    cnf = tmp_path / "f.cnf.xz"
    with lzma.open(cnf, "wt") as f:
        f.write("c comment\np cnf 3 3\n1 -2 0\n2 3\n0\n-1 -3 0\n")
    assert ev.satisfies(cnf, [1, 2, -3])
    assert not ev.satisfies(cnf, [1, 2, 3])        # violates -1 -3
    assert not ev.satisfies(cnf, [-1, -2, -3])     # violates 2 3 (split across lines)


def test_par2_counts_unsolved_twice_the_cap():
    rows = [{"status": "sat", "wall_s": 10.0}, {"status": "unsat", "wall_s": 30.0},
            {"status": "unknown", "wall_s": 300.0}, {"status": "error", "wall_s": 1.0}]
    assert ev.par2(rows, 300) == (10 + 30 + 600 + 600) / 4


def test_tick_par2_doubles_unsolved_ticks():
    rows = [{"status": "sat", "ticks": 100}, {"status": "unsat", "ticks": 300},
            {"status": "unknown", "ticks": 500}]
    assert ev.tick_par2(rows) == (100 + 300 + 1000) / 3


def test_conflict_budget_is_a_multiple_of_stock(monkeypatch):
    monkeypatch.setenv("KISSAT_EVAL_BUDGET", "3")
    assert ev.conflict_budget({"conflicts": 163375}) == 490125
    assert ev.conflict_budget({"conflicts": 10}) == 1000          # floor for trivial instances


def test_split_is_stratified_and_disjoint():
    rows = [json.loads(l) for l in (EXAMPLE / "hamiltonian_split.jsonl").read_text().splitlines()]
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "heldout", "frontier")}
    assert [len(by[s]) for s in by] == [24, 10, 6]
    assert len({r["hash"] for r in rows}) == 40
    sat_share = {s: sum(r["expected"] == "sat" for r in by[s]) / len(by[s]) for s in ("train", "heldout")}
    assert abs(sat_share["train"] - sat_share["heldout"]) < 0.05
    assert all(r["status"] == "unknown" for r in by["frontier"])


@pytest.mark.skipif(not (ev.home() / "baseline.json").is_file(), reason="run examples/kissat/prepare.py")
def test_stage1_scores_stock_and_rejects_a_wrong_policy(tmp_path):
    stock = ev.evaluate_stage1(str(EXAMPLE / "initial_program.c"))
    assert stock["wrong_answers"] == 0 and stock["solved"] == ev.STAGE1_INSTANCES
    assert stock["tick_score"] == 1.0              # deterministic: stock equals its baseline
    assert stock["conflicts_ratio"] == 1.0
    broken = tmp_path / "broken.c"
    broken.write_text(_with_block("\nbool kissat_restarting (kissat *solver) { return }\n"))
    assert ev.evaluate_stage1(str(broken))["error"].startswith("build failed")
