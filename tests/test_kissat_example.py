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


def test_tick_par2_penalty_is_fixed_by_stock(monkeypatch):
    monkeypatch.setenv("KISSAT_EVAL_BUDGET", "3")
    stock = [{"status": "sat", "ticks": 100}, {"status": "unsat", "ticks": 300},
             {"status": "sat", "ticks": 50}]
    cand = [{"status": "sat", "ticks": 80}, {"status": "unsat", "ticks": 330},
            {"status": "unknown", "ticks": 12345}]          # its own ticks don't matter
    assert ev.tick_par2(cand, stock) == (80 + 330 + 2 * 3 * 50) / 3
    assert ev.tick_par2(stock, stock) == (100 + 300 + 50) / 3
    frontier = [{"status": "unknown", "ticks": 999}]        # stock unsolved: no tick reference
    assert ev.tick_par2(frontier, frontier) is None


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
def test_stage1_scores_stock_and_rejects_a_wrong_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("KISSAT_EVAL_CACHE", "off")      # really build and run
    stock = ev.evaluate_stage1(str(EXAMPLE / "initial_program.c"))
    assert stock["wrong_answers"] == 0 and stock["solved"] == ev.STAGE1_INSTANCES
    assert stock["tick_score"] == 1.0              # deterministic: stock equals its baseline
    assert stock["conflicts_ratio"] == 1.0
    broken = tmp_path / "broken.c"
    broken.write_text(_with_block("\nbool kissat_restarting (kissat *solver) { return }\n"))
    assert ev.evaluate_stage1(str(broken))["error"].startswith("build failed")


def test_tokens_ignore_comments_and_spacing_but_not_code():
    a = 'if (x>1){ return a+=2; } /* c */ // d\nchar*s="// kept";'
    b = 'if ( x > 1 ) {\n  return a += 2;  }\nchar *s = "// kept";'
    assert ev.c_tokens(a) == ev.c_tokens(b)
    assert ev.c_tokens(a) != ev.c_tokens(a.replace("a+=2", "a+=3"))
    assert ev.c_tokens(a) != ev.c_tokens(a.replace("a+=2", "a=+2"))       # '+=' vs '=' '+'
    assert '"// kept"' in ev.c_tokens(a)                                   # not a comment


def _cache_env(monkeypatch, tmp_path):
    monkeypatch.setenv("KISSAT_EVAL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(ev, "baseline", lambda: {"sets": {"train": {"h": {"ticks": 1}}}})


def test_cache_returns_stored_metrics_for_a_repeat(monkeypatch, tmp_path):
    _cache_env(monkeypatch, tmp_path)
    p1, p2 = tmp_path / "a.c", tmp_path / "b.c"
    p1.write_text(INITIAL)
    p2.write_text(INITIAL.replace("bool kissat_restarting (kissat *solver) {",
                                  "/* same code */ bool kissat_restarting(kissat*solver){"))
    calls = []
    compute = lambda: calls.append(1) or {"combined_score": 1.25}
    assert ev.cached(str(p1), "train", 300, compute) == {"combined_score": 1.25, "cache_hit": 0.0}
    assert ev.cached(str(p2), "train", 300, compute) == {"combined_score": 1.25, "cache_hit": 1.0}
    assert calls == [1]
    ev.cached(str(p1), "stage1", 300, compute)                   # another set: another key
    ev.cached(str(p1), "train", 600, compute)                    # another cap: another key
    assert len(calls) == 3


def test_cache_skips_pool_failures_and_can_be_disabled(monkeypatch, tmp_path):
    _cache_env(monkeypatch, tmp_path)
    p = tmp_path / "a.c"
    p.write_text(INITIAL)
    calls = []
    fail = lambda: calls.append(1) or {"combined_score": 0.0, "error": "pool task x failed (spot)"}
    ev.cached(str(p), "train", 300, fail)
    ev.cached(str(p), "train", 300, fail)
    assert len(calls) == 2                                        # not a property of the program
    monkeypatch.setenv("KISSAT_EVAL_CACHE", "off")
    ok = lambda: calls.append(1) or {"combined_score": 1.0}
    assert ev.cached(str(p), "train", 300, ok) == {"combined_score": 1.0}
    assert len(calls) == 3


def test_repeat_penalty_scores_copies_zero(monkeypatch, tmp_path):
    _cache_env(monkeypatch, tmp_path)
    monkeypatch.setenv("KISSAT_EVAL_REPEATS", "penalize")
    p = tmp_path / "a.c"
    p.write_text(INITIAL)
    first = ev.cached(str(p), "train", 300, lambda: {"combined_score": 1.25})
    again = ev.cached(str(p), "train", 300, lambda: pytest.fail("re-evaluated"))
    assert first["combined_score"] == 1.25
    assert again["combined_score"] == 0.0 and again["repeat_of_score"] == 1.25
    assert "repeat" in again["error"]


def test_repeat_penalty_requires_a_run_scoped_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "baseline", lambda: {"sets": {"train": {}}})
    monkeypatch.setenv("KISSAT_EVAL_REPEATS", "penalize")
    monkeypatch.delenv("KISSAT_EVAL_CACHE", raising=False)
    p = tmp_path / "a.c"
    p.write_text(INITIAL)
    with pytest.raises(RuntimeError, match="KISSAT_EVAL_CACHE"):
        ev.cached(str(p), "train", 300, lambda: {"combined_score": 1.0})


def test_detail_lists_per_instance_results(monkeypatch, tmp_path):
    monkeypatch.setenv("KISSAT_EVAL_DETAIL", "1")
    monkeypatch.setattr(ev, "baseline", lambda: {"cap": 300, "budget_factor": 3.0,
                                                 "sets": {"stage1": {r["hash"]: {**r, "ticks": 1}
                                                                     for r in ev.instance_sets()["stage1"]}}})
    monkeypatch.setattr(ev, "build", lambda prog, tmp: Path("/bin/true"))
    monkeypatch.setattr(ev, "run_batch", lambda k, rows, cap: [
        {"hash": r["hash"], "seed": 0, "status": "sat", "ticks": 1, "conflicts": 1, "wall_s": 1.0,
         "wrong": False} for r in rows])
    m = ev.score(str(EXAMPLE / "initial_program.c"), "stage1", 300)
    assert [d["hash"] for d in m["detail"]] == [r["hash"] for r in ev.instance_sets()["stage1"]]
    assert set(m["detail"][0]) == {"hash", "seed", "status", "ticks", "conflicts", "wall_s"}


def test_mixed_split_is_stratified_by_family_and_result():
    rows = [json.loads(l) for l in (EXAMPLE / "mixed_split.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["hash"] for r in rows}) == 148
    assert {r["split"] for r in rows} == {"train", "heldout"}
    groups = {}
    for r in rows:
        groups.setdefault((r["family"], r["expected"]), []).append(r["split"])
    for splits in groups.values():                     # each family x result is halved
        assert abs(splits.count("train") - splits.count("heldout")) <= 1
    assert all(10 <= r["wall_s"] <= 200 and r["status"] == r["expected"] for r in rows)


def test_split_file_selects_its_own_stock_numbers(monkeypatch, tmp_path):
    monkeypatch.delenv("KISSAT_EVAL_SPLIT_FILE", raising=False)
    monkeypatch.delenv("KISSAT_EVAL_CAP", raising=False)
    monkeypatch.setenv("KISSAT_EVAL_HOME", str(tmp_path))
    assert ev.stock_ticks_file().name == "stock_ticks.json" and ev.baseline_file().name == "baseline.json"
    assert ev.cap() == 300 and "frontier" in ev.instance_sets()
    monkeypatch.setenv("KISSAT_EVAL_SPLIT_FILE", "mixed_split.jsonl")
    assert ev.stock_ticks_file().name == "mixed_split.stock_ticks.json"
    assert ev.baseline_file() == tmp_path / "baseline-mixed_split.json"
    assert set(ev.instance_sets()) == {"stage1", "train", "heldout"}
    assert len(ev.instance_sets()["train"]) == 74
    monkeypatch.setenv("KISSAT_EVAL_CAP", "77")
    assert ev.cap() == 77


def test_frontier_rows_use_their_own_budget_and_stock_penalty(monkeypatch):
    monkeypatch.setenv("KISSAT_EVAL_BUDGET", "3")
    assert ev.conflict_budget({"conflicts": 5_000_000, "budget": 5_000_000}) == 5_000_000
    stock = [{"status": "sat", "ticks": 100}, {"status": "unknown", "ticks": 1000, "budget_reached": True}]
    # stock: 100 + 2 x its own 1000 at the budget
    assert ev.tick_par2(stock, stock) == (100 + 2000) / 2
    gained = [{"status": "sat", "ticks": 100}, {"status": "unsat", "ticks": 700}]
    assert ev.tick_par2(gained, stock) == (100 + 700) / 2
    lost = [{"status": "unknown", "ticks": 300}, {"status": "unknown", "ticks": 1000}]
    assert ev.tick_par2(lost, stock) == (2 * 3 * 100 + 2000) / 2
    wall_capped = [stock[0], {"status": "unknown", "ticks": 400, "budget_reached": False}]
    assert ev.tick_par2(wall_capped, wall_capped) is None      # stock score undefined: frontier fallback


def test_mixed_frontier_split_adds_reachable_stock_failures():
    rows = [json.loads(l) for l in (EXAMPLE / "mixed_frontier_split.jsonl").read_text().splitlines()]
    mixed = {json.loads(l)["hash"]: json.loads(l) for l in (EXAMPLE / "mixed_split.jsonl").read_text().splitlines()}
    extra = [r for r in rows if r["hash"] not in mixed]
    assert len(rows) == len({r["hash"] for r in rows}) == 193 and len(extra) == 45
    assert all(r == mixed[r["hash"]] for r in rows if r["hash"] in mixed)
    assert all(r["status"] == "unknown" and r["budget"] == r["conflicts"] and r["sc2024_best_s"] <= 300
               for r in extra)
    assert sorted(sum(r["split"] == s for r in extra) for s in ("train", "heldout")) == [22, 23]


def test_seeds_pool_runs_against_stock_at_the_same_seed(monkeypatch):
    monkeypatch.setenv("KISSAT_EVAL_SEEDS", "0,1")
    monkeypatch.delenv("KISSAT_EVAL_SPLIT_FILE", raising=False)
    rows = ev.instance_sets()["stage1"]
    assert ev.runs(rows[:1]) == [(rows[0], 0), (rows[0], 1)]
    assert ev.run_key("abc", 0) == "abc" and ev.run_key("abc", 2) == "abc@2"
    stock = {ev.run_key(r["hash"], s): {"status": "sat", "conflicts": 1, "ticks": 100 * (s + 1)}
             for r in rows for s in (0, 1)}
    monkeypatch.setattr(ev, "baseline", lambda: {"cap": 300, "budget_factor": 3.0, "sets": {"stage1": stock}})
    monkeypatch.setattr(ev, "build", lambda prog, tmp: Path("/bin/true"))
    # candidate: as stock at seed 0, unsolved at seed 1 (penalty 2 x 3 x 200)
    monkeypatch.setattr(ev, "run_batch", lambda k, rows_, cap: [
        {"hash": r["hash"], "seed": s, "status": "sat" if s == 0 else "unknown", "ticks": 100,
         "conflicts": 1, "wall_s": 1.0, "wrong": False} for r, s in ev.runs(rows_)])
    m = ev.score(str(EXAMPLE / "initial_program.c"), "stage1", 300)
    assert m["runs"] == 2 * len(rows) and m["solved"] == len(rows)
    assert m["combined_score"] == pytest.approx((100 + 200) / (100 + 1200))
    monkeypatch.setenv("KISSAT_EVAL_SEEDS", "0,1,2")
    with pytest.raises(RuntimeError, match="seeds"):
        ev.score(str(EXAMPLE / "initial_program.c"), "stage1", 300)
    a = ev.cache_key(INITIAL, "stage1", 300)
    monkeypatch.setenv("KISSAT_EVAL_SEEDS", "0")
    monkeypatch.setattr(ev, "baseline", lambda: {"cap": 300, "budget_factor": 3.0, "sets": {"stage1": stock}})
    assert ev.cache_key(INITIAL, "stage1", 300) != a
