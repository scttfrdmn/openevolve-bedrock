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
Evaluator self-test before any search (issue #48).

Three checks, using the evaluator a run will use:

1. **reference:** the seed or hand-written program scores: ``reference.py`` in the
   planted directory if there is one (for problems whose initial program is not yet a valid
   candidate, such as KernelBench's PyTorch reference), else the initial program;
2. **null:** every ``null*.py`` in the planted directory fails;
3. **planted:** every other ``*.py`` in the planted directory fails.

"Fails" means the evaluation's outcome (``integrity.classify``) is not ``scored``.
An evaluator failure during the self-test is retried like any evaluation (#47);
if it persists, the self-test fails, since an evaluator that can't evaluate
can't be trusted to search. The result includes the **false-accept rate**: the
fraction of null and planted candidates that scored.

The planted directory comes from ``--planted`` / ``OE_BEDROCK_SELFTEST_DIR``, or
``selftest/`` next to the evaluator. It is extensible: each evaluator brings its
own known-bad programs, one file each, with a docstring saying what is wrong.

With ``OE_BEDROCK_SELFTEST=1``, ``oe-bedrock run`` runs this first and aborts
if any check fails. The result goes to the ledger as ``kind: "selftest"``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from openevolve_bedrock import integrity
from openevolve_bedrock import ledger as _ledger

FLAG_ENV = "OE_BEDROCK_SELFTEST"
DIR_ENV = "OE_BEDROCK_SELFTEST_DIR"


@dataclass
class SelfTest:
    reference: str                     # outcome of the initial program
    reference_score: float
    null: Dict[str, str] = field(default_factory=dict)       # name -> outcome
    planted: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)     # name -> evaluator's reason
    seconds: float = 0.0
    planted_dir: Optional[str] = None

    @property
    def bad(self) -> Dict[str, str]:
        return {**self.null, **self.planted}

    @property
    def false_accepts(self) -> List[str]:
        return sorted(n for n, o in self.bad.items() if o == integrity.SCORED)

    @property
    def false_accept_rate(self) -> Optional[float]:
        return len(self.false_accepts) / len(self.bad) if self.bad else None

    @property
    def passed(self) -> bool:
        return self.reference == integrity.SCORED and bool(self.null) and not self.false_accepts \
            and integrity.EVALUATOR_FAILED not in self.bad.values()

    def summary(self) -> str:
        lines = [f"self-test {'passed' if self.passed else 'FAILED'} ({self.seconds:.1f} s, {self.planted_dir})",
                 f"  reference: {self.reference} (score {self.reference_score:.4g})"]
        for kind, group in (("null", self.null), ("planted", self.planted)):
            for name, outcome in sorted(group.items()):
                mark = "ok" if outcome == integrity.CANDIDATE_FAILED else "ACCEPTED" if outcome == integrity.SCORED \
                    else "EVALUATOR FAILED"
                lines.append(f"  {kind} {name}: {mark} ({self.errors.get(name, '')[:90]})")
        if not self.null:
            lines.append("  no null candidate (null*.py) in the planted directory")
        rate = self.false_accept_rate
        lines.append(f"  false-accept rate: {'n/a' if rate is None else f'{rate:.0%}'} "
                     f"({len(self.false_accepts)} of {len(self.bad)})")
        return "\n".join(lines)


def _module(evaluator: Path):
    spec = importlib.util.spec_from_file_location("oeb_selftest_evaluator", evaluator)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(evaluator.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(evaluator.parent))
    return mod


def _evaluate(mod, program: Path) -> dict:
    """The evaluation OpenEvolve would do: stage 1 gates stage 2 when both exist."""
    if hasattr(mod, "evaluate_stage1") and hasattr(mod, "evaluate_stage2"):
        m1 = mod.evaluate_stage1(str(program))
        if integrity.classify(m1) != integrity.SCORED:
            return m1
        return mod.evaluate_stage2(str(program))
    return mod.evaluate(str(program))


def _with_retries(mod, program: Path) -> dict:
    for _ in range(integrity.retries() + 1):
        m = _evaluate(mod, program)
        if integrity.classify(m) != integrity.EVALUATOR_FAILED:
            return m
    return m


def planted_dir(evaluator: Path, override: Optional[str] = None) -> Path:
    d = override or os.environ.get(DIR_ENV)
    return Path(d) if d else evaluator.parent / "selftest"


def run(initial_program: str, evaluator: str, planted: Optional[str] = None) -> SelfTest:
    t0 = time.time()
    ev = Path(evaluator).resolve()
    mod = _module(ev)
    d = planted_dir(ev, planted)
    ref = d / "reference.py" if (d / "reference.py").is_file() else Path(initial_program)
    m = _with_retries(mod, ref)
    st = SelfTest(reference=integrity.classify(m), reference_score=float(m.get("combined_score") or 0.0),
                  planted_dir=str(d))
    if st.reference != integrity.SCORED:
        st.errors["reference"] = str(m.get("error", ""))
    for f in sorted(f for f in d.glob("*.py") if f.name != "reference.py") if d.is_dir() else []:
        r = _with_retries(mod, f)
        group = st.null if f.stem.startswith("null") else st.planted
        group[f.stem] = integrity.classify(r)
        st.errors[f.stem] = str(r.get("error", ""))
    st.seconds = time.time() - t0
    _ledger.append({"ts": time.time(), "kind": "selftest", "passed": st.passed,
                    "false_accept_rate": st.false_accept_rate, **{k: v for k, v in asdict(st).items() if k != "errors"}})
    return st


def enabled() -> bool:
    return os.environ.get(FLAG_ENV, "").strip().lower() in ("1", "true", "yes", "on")
