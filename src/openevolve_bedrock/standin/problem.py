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
A cheap CPU stand-in for an expensive kernel evaluator (issue #46).

The "kernel" is a row-wise masked softmax with an additive bias, written in
pure Python, with a ``CONFIG`` dict (block size, unroll, bias fusion) whose
effect on run time comes from a known cost model. So a run has:

- **a correctness stage** that checks the kernel against a reference on a set
  of shapes, failing on a wrong value, a NaN or a masked-position leak, the
  failure classes of the trn2 NKI evaluator (#34);
- **a costly stage** whose run time is simulated, with a known optimum and
  seeded multiplicative noise (``STANDIN_NOISE``, default 1%), and which is
  charged simulated minutes (``sim_minutes``);
- **search and held-out shapes**;
- **injectable infrastructure faults** (``STANDIN_FAULT_RATE``). A fault
  returns what real evaluators in this repo return today, a score of 0 with an
  error, and ``outcome: evaluator_failed``.

Every result has an ``outcome``: ``scored``, ``candidate_failed`` or
``evaluator_failed``. Fitness is relative to the initial program's CONFIG:

    combined_score = simulated_time(initial CONFIG) / simulated_time(candidate CONFIG)

The optimum is block = 32, unroll = 4, fuse_bias = True, at a score of
``OPTIMUM_SCORE`` (1.821 on the search shapes).
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
SEARCH_SHAPES = (5, 16, 37, 64, 120)
HELDOUT_SHAPES = (9, 50, 200)
SMALL_SHAPES = (5, 16)           # stage 1
ROWS = 4
TOL = 1e-9
MINUTES_PER_SHAPE = 0.5          # simulated cost of the costly stage
INITIAL_CONFIG = {"block": 8, "unroll": 1, "fuse_bias": False}
OPTIMAL_CONFIG = {"block": 32, "unroll": 4, "fuse_bias": True}
BLOCKS = tuple(2 ** k for k in range(0, 9))
UNROLLS = (1, 2, 4, 8)
_attempts: Dict[str, int] = {}   # per program, for fault injection that varies across retries


def reference(x: List[float], mask: List[bool], bias: List[float]) -> List[float]:
    kept = [x[i] + bias[i] for i in range(len(x)) if mask[i]]
    m = max(kept)
    s = sum(math.exp(v - m) for v in kept)
    return [math.exp(x[i] + bias[i] - m) / s if mask[i] else 0.0 for i in range(len(x))]


def inputs(length: int, row: int, seed: int = 0):
    rng = random.Random(f"{seed}:{length}:{row}")
    x = [rng.uniform(-4, 4) for _ in range(length)]
    bias = [rng.uniform(-1, 1) for _ in range(length)]
    mask = [rng.random() > 0.3 for _ in range(length)]
    mask[rng.randrange(length)] = True
    return x, mask, bias


def sim_time(cfg: dict, length: int) -> float:
    eff = (1 + 0.15 * abs(math.log2(cfg["block"] / 32))) * (1 + 0.1 * abs(math.log2(cfg["unroll"] / 4))) \
        * (1.0 if cfg["fuse_bias"] else 1.2)
    return length * eff + 3.0


def total_time(cfg: dict, shapes) -> float:
    return sum(sim_time(cfg, n) for n in shapes)


OPTIMUM_SCORE = total_time(INITIAL_CONFIG, SEARCH_SHAPES) / total_time(OPTIMAL_CONFIG, SEARCH_SHAPES)


def _load(path: str):
    spec = importlib.util.spec_from_file_location(f"standin_cand_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config_error(cfg) -> Optional[str]:
    if not isinstance(cfg, dict) or set(cfg) != set(INITIAL_CONFIG):
        return f"CONFIG must have exactly the keys {sorted(INITIAL_CONFIG)}"
    if cfg["block"] not in BLOCKS:
        return f"CONFIG block must be one of {BLOCKS}"
    if cfg["unroll"] not in UNROLLS:
        return f"CONFIG unroll must be one of {UNROLLS}"
    if not isinstance(cfg["fuse_bias"], bool):
        return "CONFIG fuse_bias must be a bool"
    return None


def check(mod, shapes) -> Optional[str]:
    """None if the kernel is correct on every shape, else the first failure."""
    for n in shapes:
        for r in range(ROWS):
            x, mask, bias = inputs(n, r)
            try:
                got = mod.kernel(list(x), list(mask), list(bias))
            except Exception as e:  # noqa: BLE001
                return f"kernel raised on length {n}: {type(e).__name__}: {e}"
            if not isinstance(got, list) or len(got) != n:
                return f"wrong output length on length {n}"
            if any(v != v or v in (math.inf, -math.inf) for v in got):
                return f"NaN or Inf on length {n}"
            if any(not mask[i] and got[i] != 0.0 for i in range(n)):
                return f"masked-position leak on length {n}"
            want = reference(x, mask, bias)
            err = max(abs(a - b) for a, b in zip(got, want))
            if err > TOL:
                return f"wrong output on length {n} (max abs error {err:.3g})"
    return None


def _digest(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def _noise(code: str, shapes) -> float:
    sd = float(os.environ.get("STANDIN_NOISE", "0.01"))
    rng = random.Random(f"{_digest(code)}:{os.environ.get('STANDIN_SEED', '0')}:{','.join(map(str, shapes))}")
    return max(0.5, 1 + rng.gauss(0, sd))


def _fault(code: str) -> bool:
    rate = float(os.environ.get("STANDIN_FAULT_RATE", "0"))
    if rate <= 0:
        return False
    key = _digest(code)
    _attempts[key] = _attempts.get(key, 0) + 1
    rng = random.Random(f"{key}:{_attempts[key]}:{os.environ.get('STANDIN_SEED', '0')}")
    return rng.random() < rate


def score(program_path: str, shapes=SEARCH_SHAPES, costly: bool = True) -> dict:
    code = Path(program_path).read_text()
    try:
        mod = _load(program_path)
    except Exception as e:  # noqa: BLE001
        return {"combined_score": 0.0, "outcome": "candidate_failed", "error": f"load failed: {type(e).__name__}: {e}"}
    cfg = getattr(mod, "CONFIG", None)
    reason = _config_error(cfg)
    if reason:
        return {"combined_score": 0.0, "outcome": "candidate_failed", "error": reason}
    if not hasattr(mod, "kernel"):
        return {"combined_score": 0.0, "outcome": "candidate_failed", "error": "no kernel()"}
    reason = check(mod, shapes)
    if reason:
        return {"combined_score": 0.0, "outcome": "candidate_failed", "error": reason}
    if not costly:
        return {"combined_score": 1.0, "outcome": "scored", "sim_minutes": 0.0}
    if _fault(code):
        return {"combined_score": 0.0, "outcome": "evaluator_failed",
                "error": "simulated infrastructure fault: transport error", "sim_minutes": MINUTES_PER_SHAPE}
    t = total_time(cfg, shapes) * _noise(code, shapes)
    return {"combined_score": total_time(INITIAL_CONFIG, shapes) / t, "outcome": "scored",
            "sim_minutes": MINUTES_PER_SHAPE * len(shapes),
            "true_score": total_time(INITIAL_CONFIG, shapes) / total_time(cfg, shapes)}


def evaluate_stage1(program_path: str) -> dict:
    """Correctness on the small shapes only, no costly stage."""
    return score(program_path, SMALL_SHAPES, costly=False)


def evaluate_stage2(program_path: str) -> dict:
    return score(program_path, SEARCH_SHAPES)


def evaluate(program_path: str) -> dict:
    return evaluate_stage2(program_path)


def evaluate_heldout(program_path: str) -> dict:
    return score(program_path, HELDOUT_SHAPES)


def defects_dir() -> Path:
    return HERE / "defects"


def initial_program() -> Path:
    return HERE / "initial_program.py"
