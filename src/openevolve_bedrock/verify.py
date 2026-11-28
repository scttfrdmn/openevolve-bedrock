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
Re-score a run's best programs on held-out cases at several seeds (issue #41).

Every result in this repo's issues needed this step, done by hand each time: a
run's best training score is the maximum of many noisy scores, so it shrinks
on data the search never saw. ``oe-bedrock verify`` does it for any evaluator
that defines the optional hook::

    def evaluate_heldout(program_path: str, seed: int) -> dict   # metrics, like evaluate()

From the run's last checkpoint it takes the ``--top`` distinct programs by
training score, plus the initial program as a reference line, and scores each at
every ``--seeds`` value. Per program it reports:

- the training score recorded in the checkpoint;
- the held-out score at each seed;
- their mean, with a 95% t-interval;
- the shrinkage, held-out mean minus training score.

A program whose held-out evaluation fails correctness at any seed is **rejected**,
and its training score is shown beside it. The result goes to the ledger as
``kind: "verify"``.
"""

from __future__ import annotations

import json
import math
import statistics as S
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from openevolve_bedrock import integrity
from openevolve_bedrock import ledger as _ledger
from openevolve_bedrock.selftest import _module

# two-sided 95% t quantiles by degrees of freedom (1..30); normal beyond
_T95 = [12.71, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


@dataclass
class Verified:
    name: str
    train: Optional[float]
    heldout: List[float] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: List[dict] = field(default_factory=list)     # numeric held-out metrics, one dict per seed

    def metric_mean(self, key: str) -> Optional[float]:
        """Mean of one held-out metric (e.g. ``cpu_ratio``) over the scored seeds, or None if absent."""
        xs = [m[key] for m, o in zip(self.metrics, self.outcomes) if o == integrity.SCORED and key in m]
        return S.mean(xs) if xs else None

    @property
    def rejected(self) -> bool:
        """A held-out correctness failure. An evaluator failure is not one (#47)."""
        return any(o == integrity.CANDIDATE_FAILED for o in self.outcomes)

    @property
    def scored(self) -> List[float]:
        return [x for x, o in zip(self.heldout, self.outcomes) if o == integrity.SCORED]

    @property
    def evaluator_failures(self) -> int:
        return sum(1 for o in self.outcomes if o == integrity.EVALUATOR_FAILED)

    @property
    def mean(self) -> Optional[float]:
        return S.mean(self.scored) if self.scored and not self.rejected else None

    @property
    def interval(self):
        xs = self.scored
        if self.rejected or len(xs) < 2:
            return None
        t = _T95[len(xs) - 2] if len(xs) - 2 < len(_T95) else 1.96
        h = t * S.stdev(xs) / math.sqrt(len(xs))
        return self.mean - h, self.mean + h

    @property
    def shrinkage(self) -> Optional[float]:
        return None if self.mean is None or self.train is None else self.mean - self.train


def checkpoint_programs(output_dir: Path) -> List[dict]:
    cks = sorted((output_dir / "checkpoints").glob("checkpoint_*"), key=lambda d: int(d.name.split("_")[1]))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {output_dir}")
    return [json.loads(f.read_text()) for f in (cks[-1] / "programs").glob("*.json")]


def top_distinct(programs: List[dict], k: int) -> List[dict]:
    seen, out = set(), []
    for p in sorted(programs, key=lambda p: -float((p.get("metrics") or {}).get("combined_score") or 0)):
        code = "".join(p["code"].split())
        if code in seen:
            continue
        seen.add(code)
        out.append(p)
        if len(out) == k:
            break
    return out


def run(output_dir: str, evaluator: str, top: int = 4, seeds=(0, 1, 2, 3),
        initial: Optional[str] = None) -> List[Verified]:
    mod = _module(Path(evaluator).resolve())
    if not hasattr(mod, "evaluate_heldout"):
        raise AttributeError(f"{evaluator} has no evaluate_heldout(program_path, seed); see openevolve_bedrock.verify")
    rows: List[Verified] = []
    with tempfile.TemporaryDirectory(prefix="oeb-verify-") as tmp:
        targets = [(f"#{i + 1} {p['id'][:8]} (it {p.get('iteration_found')})", p["code"],
                    float(p["metrics"].get("combined_score") or 0))
                   for i, p in enumerate(top_distinct(checkpoint_programs(Path(output_dir)), top))]
        if initial:
            targets.append(("initial", Path(initial).read_text(), None))
        for name, code, train in targets:
            f = Path(tmp) / f"p{len(rows)}.py"
            f.write_text(code)
            v = Verified(name, train)
            for s in seeds:
                for _ in range(integrity.retries() + 1):     # evaluator failures are retried, as in a run (#47)
                    m = mod.evaluate_heldout(str(f), s)
                    if integrity.classify(m) != integrity.EVALUATOR_FAILED:
                        break
                v.outcomes.append(integrity.classify(m))
                v.heldout.append(float(m.get("combined_score") or 0.0))
                v.errors.append(str(m.get("error", "")))
                v.metrics.append({k: float(x) for k, x in m.items()
                                  if isinstance(x, (int, float)) and not isinstance(x, bool)})
            rows.append(v)
    _ledger.append({"ts": time.time(), "kind": "verify", "output_dir": str(output_dir), "seeds": list(seeds),
                    "programs": [{"name": v.name, "train": v.train, "heldout": v.heldout, "rejected": v.rejected,
                                  "mean": v.mean, "shrinkage": v.shrinkage} for v in rows]})
    return rows


def table(rows: List[Verified]) -> str:
    out = [f"{'program':28} {'train':>8} {'heldout mean':>12} {'95% interval':>19} {'shrinkage':>9}  per seed"]
    for v in rows:
        tr = "" if v.train is None else f"{v.train:.4f}"
        if v.rejected:
            bad = next(e for o, e in zip(v.outcomes, v.errors) if o == integrity.CANDIDATE_FAILED)
            out.append(f"{v.name:28} {tr:>8} {'REJECTED':>12} {'':>19} {'':>9}  {bad[:70]}")
            continue
        ci = v.interval
        out.append(f"{v.name:28} {tr:>8} {v.mean:12.4f} {('%.4f–%.4f' % ci) if ci else '':>19} "
                   f"{'' if v.shrinkage is None else f'{v.shrinkage:+.4f}':>9}  "
                   + " ".join(f"{x:.4f}" if o == integrity.SCORED else "evalfail" for x, o in zip(v.heldout, v.outcomes)))
    return "\n".join(out)
