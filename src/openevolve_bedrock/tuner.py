# Copyright 2026 Scott Friedman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The inner tuner of the two-level search (issue #64): search a structure's
declared parameter space with no model.

A **space** maps each parameter name to one of:

- ``{"values": [...]}``: categorical, bool or a discrete numeric grid;
- ``{"low": a, "high": b, "type": "int"}`` or ``"float"``: a bounded range,
  with ``"log": True`` for log scale.

An optional ``constraint(config) -> bool`` rejects invalid combinations.

A tuner is any callable ``tune(space, objective, budget, seed, constraint=None)
-> History``, where ``objective(config) -> float`` scores a configuration
(higher is better, 0 for an invalid one). The budget counts distinct
evaluations: both tuners here cache, so a repeated configuration costs nothing
and is never evaluated twice. That keeps them comparable at equal evaluations.
An external tuner can replace these by implementing the same signature.

Two tuners are provided. Per #64, random search was the default unless the GA beat it at
equal evaluations. It did, on all three test spaces, from about 20 evaluations on, so the
default is the GA (``experiments/two_level/tuner_compare.py``):

- ``random_search``: uniform sampling without replacement where the space is finite;
- ``genetic``: tournament selection, uniform crossover, bounded per-parameter
  mutation, constraint repair by resampling, and one elite.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

Config = Dict[str, object]


@dataclass
class History:
    configs: List[Config] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    def best(self) -> Tuple[Optional[Config], float]:
        if not self.scores:
            return None, 0.0
        i = max(range(len(self.scores)), key=self.scores.__getitem__)
        return self.configs[i], self.scores[i]

    def best_so_far(self) -> List[float]:
        out, b = [], -math.inf
        for s in self.scores:
            b = max(b, s)
            out.append(b)
        return out


def _key(c: Config) -> str:
    return json.dumps(c, sort_keys=True, default=str)


def _sample_param(spec: dict, rng: random.Random):
    if "values" in spec:
        return rng.choice(spec["values"])
    lo, hi = spec["low"], spec["high"]
    if spec.get("log"):
        v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    else:
        v = rng.uniform(lo, hi)
    return int(round(v)) if spec.get("type") == "int" else v


def _size(space: Dict[str, dict]) -> Optional[int]:
    n = 1
    for spec in space.values():
        if "values" not in spec:
            if spec.get("type") != "int":
                return None
            n *= spec["high"] - spec["low"] + 1
        else:
            n *= len(spec["values"])
    return n


def _sample(space, rng, constraint, tries=200) -> Config:
    for _ in range(tries):
        c = {k: _sample_param(v, rng) for k, v in space.items()}
        if constraint is None or constraint(c):
            return c
    raise ValueError("could not sample a configuration that satisfies the constraint")


class _Counted:
    """The objective, evaluated at most once per configuration and at most ``budget`` times."""

    def __init__(self, objective, budget):
        self.objective, self.budget, self.cache, self.h = objective, budget, {}, History()

    @property
    def spent(self) -> bool:
        return len(self.h.scores) >= self.budget

    def __call__(self, c: Config) -> float:
        k = _key(c)
        if k not in self.cache:
            s = float(self.objective(c))
            self.cache[k] = s
            self.h.configs.append(dict(c))
            self.h.scores.append(s)
        return self.cache[k]


def random_search(space, objective, budget, seed=0, constraint=None) -> History:
    rng = random.Random(seed)
    f = _Counted(objective, budget)
    limit = _size(space)
    misses = 0
    while not f.spent and (limit is None or len(f.cache) < limit) and misses < 10 * budget:
        c = _sample(space, rng, constraint)
        if _key(c) in f.cache:
            misses += 1
            continue
        f(c)
    return f.h


def genetic(space, objective, budget, seed=0, constraint=None, population=8, tournament=3,
            mutation=None, crossover=0.9) -> History:
    rng = random.Random(seed)
    f = _Counted(objective, budget)
    names = list(space)
    pm = mutation if mutation is not None else 1.0 / len(names)
    limit = _size(space)

    def mutate(c):
        out = dict(c)
        for k in names:
            if rng.random() < pm:
                spec = space[k]
                if "values" in spec:
                    vals = spec["values"]
                    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals) and len(vals) > 2:
                        i = vals.index(out[k]) if out[k] in vals else rng.randrange(len(vals))
                        out[k] = vals[max(0, min(len(vals) - 1, i + rng.choice([-1, 1])))]   # a neighbour on the grid
                    else:
                        out[k] = rng.choice(vals)
                else:
                    span = (spec["high"] - spec["low"]) * 0.2
                    v = min(spec["high"], max(spec["low"], out[k] + rng.gauss(0, span)))
                    out[k] = int(round(v)) if spec.get("type") == "int" else v
        return out

    def repair(c):
        for _ in range(50):
            if constraint is None or constraint(c):
                return c
            k = rng.choice(names)
            c = {**c, k: _sample_param(space[k], rng)}
        return _sample(space, rng, constraint)

    pop = []
    while len(pop) < population and not f.spent:
        c = _sample(space, rng, constraint)
        pop.append((c, f(c)))
    stall = 0
    while not f.spent and (limit is None or len(f.cache) < limit) and stall < 20 * budget:
        def pick():
            return max(rng.sample(pop, min(tournament, len(pop))), key=lambda p: p[1])[0]

        a, b = pick(), pick()
        child = {k: (a[k] if rng.random() < 0.5 else b[k]) for k in names} if rng.random() < crossover else dict(a)
        child = repair(mutate(child))
        if _key(child) in f.cache:
            stall += 1
            child = repair(mutate(child))       # nudge off already-evaluated points
            if _key(child) in f.cache:
                continue
        s = f(child)
        stall = 0
        elite = max(pop, key=lambda p: p[1])
        worst = min(range(len(pop)), key=lambda i: pop[i][1])
        if s >= pop[worst][1]:
            pop[worst] = (child, s)
        if elite not in pop:
            pop[worst] = elite
    return f.h


TUNERS: Dict[str, Callable] = {"random": random_search, "genetic": genetic}
DEFAULT = "genetic"  # #64: beat random search at equal evaluations on the stand-in, the Qiskit table and synthetic8
