# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
#68 part 8 on the NumPy stand-in: how much of the performance spread is parameters (within a
structure) and how much is structure (between structures), with every structure tuned first.

    python experiments/two_level/decompose_numpy.py measure table.jsonl     # every parameter point
    python experiments/two_level/decompose_numpy.py analyse table.jsonl     # tune, decompose, heldout

``measure`` times every point of every structure's declared space (search shapes, threads
pinned). ``analyse`` then:

1. runs the inner tuner (``openevolve_bedrock.tuner.DEFAULT``) on each structure against the
   measured table (free, since the table is the objective), to show it reaches each structure's
   best, and takes each structure's **tuned best**;
2. decomposes the variance of log(score) into a within-structure component (parameters) and a
   between-structure component. Structures are weighted equally, since their spaces differ in
   size (6, 50, 250). Reported twice: over all points, and over the tuned bests (the
   comparison the two-level design makes);
3. re-times each structure's default and tuned best on the held-out shapes, with intervals.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "two_level" / "numpy_standin"))
import evaluate as ev  # noqa: E402
import spec  # noqa: E402

from openevolve_bedrock import tuner  # noqa: E402


def points(mod):
    keys = list(mod.SPACE)
    return [dict(zip(keys, v)) for v in itertools.product(*[mod.SPACE[k]["values"] for k in keys])]


def measure(out: Path, repeats: int = 5) -> None:
    done = {(json.loads(l)["structure"], json.dumps(json.loads(l)["params"], sort_keys=True))
            for l in out.open()} if out.exists() else set()
    for name, mod in ev.structures().items():
        for p in points(mod):
            if (name, json.dumps(p, sort_keys=True)) in done:
                continue
            m = ev.evaluate(name, p, repeats=repeats)
            with out.open("a") as f:
                f.write(json.dumps({"structure": name, "params": p, "score": m["combined_score"],
                                    "lo": m.get("interval_lo"), "hi": m.get("interval_hi"),
                                    "error": m.get("error")}) + "\n")
            print(name, p, round(m["combined_score"], 4), flush=True)


def analyse(table: Path) -> None:
    rows = [json.loads(l) for l in table.open()]
    mods = ev.structures()
    by = {n: [r for r in rows if r["structure"] == n and not r.get("error")] for n in mods}
    print(f"{len(rows)} points; failures: {sum(1 for r in rows if r.get('error'))}")

    # 1. tune each structure against its measured table
    print("\n1. inner tuner per structure (budget 30, 20 seeds), against the measured table")
    tuned = {}
    for n, mod in mods.items():
        look = {json.dumps(r["params"], sort_keys=True): r["score"] for r in by[n]}
        best = max(by[n], key=lambda r: r["score"])
        default = look[json.dumps(mod.DEFAULTS, sort_keys=True)]
        found = []
        for s in range(20):
            h = tuner.TUNERS[tuner.DEFAULT](mod.SPACE, lambda c: look[json.dumps(c, sort_keys=True)],
                                            min(30, len(by[n])), seed=s)
            found.append(h.best()[1])
        tuned[n] = best
        print(f"  {n:12} {len(by[n]):4d} points  default {default:.4f}  best {best['score']:.4f} "
              f"{best['params']}  tuner (30 evals) mean {S.mean(found):.4f}, "
              f"{sum(f >= best['score'] * 0.99 for f in found)}/20 within 1% of best")

    # 2. variance decomposition of log(score), structures weighted equally
    def decompose(groups):
        means = [S.mean(g) for g in groups]
        within = S.mean([S.pvariance(g) if len(g) > 1 else 0.0 for g in groups])
        between = S.pvariance(means)
        return within, between

    logs = [[math.log(r["score"]) for r in by[n]] for n in mods]
    w, b = decompose(logs)
    print(f"\n2. variance of log(score), all points: within {w:.5f} ({w / (w + b):.0%}), "
          f"between {b:.5f} ({b / (w + b):.0%})")
    best_logs = [math.log(tuned[n]["score"]) for n in mods]
    bests = ", ".join(f"{n} {tuned[n]['score']:.4f}" for n in mods)
    print(f"   tuned bests: {bests}; spread between tuned structures "
          f"{max(best_logs) - min(best_logs):.4f} in log units")
    gains = {n: math.log(tuned[n]["score"]) - math.log(next(r["score"] for r in by[n] if r["params"] ==
                                                            {**mods[n].DEFAULTS})) for n in mods}
    print("   default -> tuned gain within each structure (log units): "
          + ", ".join(f"{n} {g:+.4f}" for n, g in gains.items()))
    worst_to_best = {n: math.log(max(r["score"] for r in by[n])) - math.log(min(r["score"] for r in by[n]))
                     for n in mods}
    print("   worst -> best within each structure (log units): "
          + ", ".join(f"{n} {g:.4f}" for n, g in worst_to_best.items()))

    # 3. heldout re-timing of default and tuned best per structure
    print("\n3. heldout shapes, repeats 9 (score [p10, p90])")
    for n, mod in mods.items():
        for label, p in (("default", mod.DEFAULTS), ("tuned", tuned[n]["params"])):
            m = ev.evaluate(n, p, shapes=spec.HELDOUT_SHAPES, repeats=9)
            print(f"  {n:12} {label:8} {m['combined_score']:.4f} [{m['interval_lo']:.4f}, {m['interval_hi']:.4f}] {p}")


if __name__ == "__main__":
    {"measure": lambda: measure(Path(sys.argv[2])), "analyse": lambda: analyse(Path(sys.argv[2]))}[sys.argv[1]]()
