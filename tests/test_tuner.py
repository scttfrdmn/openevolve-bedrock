# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""The inner tuner (#64): equal-evaluation budgets, no repeats, constraints, determinism, and the GA's edge."""
import pytest

from openevolve_bedrock import tuner
from openevolve_bedrock.standin import problem as P

SPACE = {"block": {"values": list(P.BLOCKS)}, "unroll": {"values": list(P.UNROLLS)},
         "fuse_bias": {"values": [False, True]}}


def true(c):
    return P.total_time(P.INITIAL_CONFIG, P.SEARCH_SHAPES) / P.total_time(c, P.SEARCH_SHAPES)


@pytest.mark.parametrize("name", ["random", "genetic"])
def test_budget_is_distinct_evaluations_and_seeded(name):
    calls = []

    def f(c):
        calls.append(c)
        return true(c)

    h = tuner.TUNERS[name](SPACE, f, 25, seed=3)
    assert len(h.scores) == 25 == len(calls)
    assert len({tuple(sorted(c.items())) for c in calls}) == 25            # never the same config twice
    assert tuner.TUNERS[name](SPACE, true, 25, seed=3).configs == h.configs
    assert h.best_so_far() == sorted(h.best_so_far())


@pytest.mark.parametrize("name", ["random", "genetic"])
def test_a_finite_space_is_exhausted_not_overrun(name):
    h = tuner.TUNERS[name](SPACE, true, 500, seed=0)
    assert len(h.scores) <= 72 and h.best()[1] == pytest.approx(P.OPTIMUM_SCORE)


@pytest.mark.parametrize("name", ["random", "genetic"])
def test_constraints_hold(name):
    no_small_blocks_with_fuse = lambda c: not (c["fuse_bias"] and c["block"] < 16)  # noqa: E731
    h = tuner.TUNERS[name](SPACE, true, 30, seed=1, constraint=no_small_blocks_with_fuse)
    assert all(no_small_blocks_with_fuse(c) for c in h.configs)


def test_ranges_and_log_scale():
    space = {"a": {"low": 1, "high": 1000, "type": "int", "log": True}, "b": {"low": 0.0, "high": 1.0}}
    h = tuner.genetic(space, lambda c: c["b"], 40, seed=2)
    assert all(1 <= c["a"] <= 1000 and isinstance(c["a"], int) and 0 <= c["b"] <= 1 for c in h.configs)


def test_ga_beats_random_search_at_equal_evaluations_on_the_standin():
    """The #64 result the default rests on: at 40 evaluations, over 50 seeds."""
    def hit_rate(name):
        return sum(true(tuner.TUNERS[name](SPACE, true, 40, seed=s).best()[0]) >= P.OPTIMUM_SCORE * 0.995
                   for s in range(50)) / 50
    assert hit_rate("genetic") > hit_rate("random") + 0.25
    assert tuner.DEFAULT == "genetic"
