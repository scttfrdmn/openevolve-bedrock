# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
#64: genetic algorithm against random search at equal evaluations, on three spaces.

    python experiments/two_level/tuner_compare.py [--qiskit-table space.jsonl] [--runs 200]

- **standin**: the #46 stand-in's CONFIG space, 72 configurations, true optimum 1.821, 1% noise.
- **qiskit**: the #58 table of 320 real Qiskit configurations (train score at 2 seeds; over the
  CPU cap = 0), if ``--qiskit-table`` is given. The table is the truth here.
- **synthetic8**: 8 parameters of 8 levels each (16.7M configurations), coupled pairwise
  (a quadratic form with off-diagonal terms), optimum 1.8, 1% noise. A probe of the
  "6 or more interacting parameters" case, not a real problem.

Each tuner picks its best by the noisy score it saw. What is reported is the **true** score of
that pick, so selection on noise counts against it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as S
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from openevolve_bedrock import tuner  # noqa: E402
from openevolve_bedrock.standin import problem as P  # noqa: E402


def standin():
    space = {"block": {"values": list(P.BLOCKS)}, "unroll": {"values": list(P.UNROLLS)},
             "fuse_bias": {"values": [False, True]}}

    def true(c):
        return P.total_time(P.INITIAL_CONFIG, P.SEARCH_SHAPES) / P.total_time(c, P.SEARCH_SHAPES)
    return space, true, P.OPTIMUM_SCORE


def qiskit(table: Path):
    rows = [json.loads(l) for l in table.open()]
    keys = ("max_iterations", "layout_trials", "swap_trials", "vf2post", "opt_level3")
    space = {k: {"values": sorted({r[k] for r in rows}, key=lambda v: (str(type(v)), v))} for k in keys}
    lookup = {tuple(r[k] for k in keys): (0.0 if r.get("error") else r["score"]) for r in rows}

    def true(c):
        return lookup[tuple(c[k] for k in keys)]
    return space, true, max(lookup.values())


def synthetic8(seed=7):
    rng = random.Random(seed)
    n, levels = 8, 8
    target = [rng.randrange(levels) for _ in range(n)]
    a = [[(1.0 if i == j else (0.45 if abs(i - j) == 1 else 0.0)) for j in range(n)] for i in range(n)]
    space = {f"x{i}": {"values": list(range(levels))} for i in range(n)}

    def true(c):
        d = [c[f"x{i}"] - target[i] for i in range(n)]
        q = sum(d[i] * a[i][j] * d[j] for i in range(n) for j in range(n))
        return 1.0 + 0.8 * math.exp(-q / 12.0)
    return space, true, 1.8


def noisy(true, sd, seed):
    def f(c):
        rng = random.Random(f"{json.dumps(c, sort_keys=True, default=str)}:{seed}")
        return true(c) * (1 + rng.gauss(0, sd)) if true(c) > 0 else 0.0
    return f


def compare(name, space, true, optimum, sd, runs, budgets):
    print(f"\n{name}: optimum {optimum:.4f}, noise sd {sd:.0%}")
    print(f"{'budget':>6} {'tuner':>8} {'true score of pick, mean':>25} {'p10':>8} {'within 0.5% of optimum':>23}")
    out = {}
    for b in budgets:
        for t in ("random", "genetic"):
            picks = []
            for s in range(runs):
                h = tuner.TUNERS[t](space, noisy(true, sd, s) if sd else true, b, seed=s)
                c, _ = h.best()
                picks.append(true(c))
            picks.sort()
            within = sum(p >= optimum * 0.995 for p in picks) / runs
            out[(b, t)] = (S.mean(picks), picks[runs // 10], within)
            print(f"{b:6d} {t:>8} {S.mean(picks):25.4f} {picks[runs // 10]:8.4f} {within:23.0%}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qiskit-table")
    ap.add_argument("--runs", type=int, default=200)
    a = ap.parse_args()
    budgets = (10, 20, 40, 80)
    results = {"standin": compare("standin", *standin(), 0.01, a.runs, budgets)}
    if a.qiskit_table:
        results["qiskit"] = compare("qiskit (#58 table)", *qiskit(Path(a.qiskit_table)), 0.0, a.runs, budgets)
    results["synthetic8"] = compare("synthetic8", *synthetic8(), 0.01, a.runs, (20, 40, 80, 160, 320))
    return results


if __name__ == "__main__":
    main()
