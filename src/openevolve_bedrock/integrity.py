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
Fitness integrity: an evaluator failure is not a candidate failure (issue #47).

Every evaluation is classified as one of:

- ``scored``: the evaluator produced a score above 0;
- ``candidate_failed``: a score of 0 for a reason in the candidate (it did not
  build, gave a wrong answer, hacked, or timed out in OpenEvolve's evaluator);
- ``evaluator_failed``: the evaluator says the attempt failed for a reason
  outside the candidate (a lost worker, a transport error, no result from the
  GPU host). An evaluator signals this with ``evaluator_failed: 1.0`` in its
  metrics. It has to be a number, because OpenEvolve's cascade keeps only
  numeric metrics.

With ``OE_BEDROCK_FITNESS_INTEGRITY=1``, an ``evaluator_failed`` attempt is
retried up to ``OE_BEDROCK_EVAL_RETRIES`` times (default 2). If every attempt
fails, the evaluation raises ``EvaluationSkipped``. OpenEvolve's worker turns
that into an iteration error, so the child is not added to the archive, rather
than being archived with a score of 0. Each attempt adds a ``kind: "outcome"``
line to the ledger.

Without the variable, nothing changes: OpenEvolve's ``evaluate_program`` runs as is.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Dict, Iterable

from openevolve_bedrock import ledger as _ledger

FLAG_ENV = "OE_BEDROCK_FITNESS_INTEGRITY"
RETRIES_ENV = "OE_BEDROCK_EVAL_RETRIES"
OBSERVE_ENV = "OE_BEDROCK_OUTCOMES"   # log each evaluation's outcome without retrying or skipping (#53)
SCORED, CANDIDATE_FAILED, EVALUATOR_FAILED, SKIPPED = "scored", "candidate_failed", "evaluator_failed", "skipped"
_installed = False


class EvaluationSkipped(RuntimeError):
    """Every attempt failed in the evaluator; the candidate has no fitness."""


def enabled() -> bool:
    return os.environ.get(FLAG_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def observing() -> bool:
    return os.environ.get(OBSERVE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def retries() -> int:
    return max(0, int(os.environ.get(RETRIES_ENV, "2")))


def classify(metrics) -> str:
    m = metrics if isinstance(metrics, dict) else {}
    outcome = m.get("outcome")
    if outcome in (SCORED, CANDIDATE_FAILED, EVALUATOR_FAILED):
        return outcome
    if float(m.get("evaluator_failed") or 0.0) >= 1.0:
        return EVALUATOR_FAILED
    try:
        return SCORED if float(m.get("combined_score") or 0.0) > 0 else CANDIDATE_FAILED
    except (TypeError, ValueError):
        return CANDIDATE_FAILED


def code_hash(code: str) -> str:
    """Whitespace-insensitive hash of a candidate, for duplicate counting (#53)."""
    return hashlib.sha256("".join((code or "").split()).encode()).hexdigest()[:16]


def _record(program_id: str, cls: str, attempt: int, metrics, code: str = "") -> None:
    _ledger.append({"ts": time.time(), "kind": "outcome", "program_id": program_id, "class": cls,
                    "attempt": attempt, "code_hash": code_hash(code),
                    "error": str((metrics or {}).get("error", ""))[:300] if isinstance(metrics, dict) else ""})


def counts(records: Iterable[dict]) -> Dict[str, int]:
    """Final outcome per program (its last attempt; ``evaluator_failed`` there means
    skipped), plus the number of attempts and of evaluator-failed attempts."""
    recs = [r for r in records if r.get("kind") == "outcome"]
    last: Dict[str, str] = {}
    for i, rec in enumerate(recs):
        last[rec.get("program_id") or f"anon-{i}"] = rec["class"]
    out = {SCORED: 0, CANDIDATE_FAILED: 0, SKIPPED: 0}
    for cls in last.values():
        out[SKIPPED if cls == EVALUATOR_FAILED else cls] += 1
    out["attempts"] = len(recs)
    out["evaluator_failed_attempts"] = sum(1 for r in recs if r["class"] == EVALUATOR_FAILED)
    return out


def outcome_counts(path=None) -> Dict[str, int]:
    return counts(_ledger.read(path or _ledger.ledger_path()))


def wrap(original):
    """The retrying ``evaluate_program`` around OpenEvolve's (a pass-through unless enabled)."""

    async def evaluate_program(self, program_code, program_id=""):
        if not enabled():
            metrics = await original(self, program_code, program_id)
            if observing():      # record only: behaviour is exactly OpenEvolve's
                _record(program_id, classify(metrics), 0, metrics, program_code)
            return metrics
        n = retries()
        for attempt in range(n + 1):
            metrics = await original(self, program_code, program_id)
            cls = classify(metrics)
            _record(program_id, cls, attempt, metrics, program_code)
            if cls != EVALUATOR_FAILED:
                return metrics
        raise EvaluationSkipped(f"evaluator failed on all {n + 1} attempts for program {program_id or '?'}: "
                                f"{(metrics or {}).get('error', '') if isinstance(metrics, dict) else ''}")

    evaluate_program.__wrapped__ = original
    return evaluate_program


def install() -> None:
    """Wrap ``openevolve.evaluator.Evaluator.evaluate_program`` (idempotent)."""
    global _installed
    if _installed:
        return
    try:
        from openevolve.evaluator import Evaluator
    except ImportError:
        return
    Evaluator.evaluate_program = wrap(Evaluator.evaluate_program)
    _installed = True
