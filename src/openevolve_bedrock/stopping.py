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
Stopping a run early: on a dollar budget, and with a correct last checkpoint.

OpenEvolve stops on ``--target-score`` itself, but then saves its final
checkpoint under ``start + max_iterations - 1`` whatever iteration it stopped
at. ``register()`` wraps two ``ProcessParallelController`` methods (parent
process only; workers never call them):

* ``_submit_iteration``: with ``OE_BEDROCK_BUDGET_USD`` and
  ``OE_BEDROCK_PRICING`` set, the ledger is priced before each new iteration is
  submitted, and nothing more is submitted once the total reaches the budget.
  Iterations already in flight finish and are recorded, so the overshoot is
  those calls and the ledger stays complete. OpenEvolve's loop ends when
  nothing is pending.
* ``run_evolution``: after any early stop (budget or ``--target-score``), a
  checkpoint is saved at the last completed iteration, and the controller's
  shutdown flag is set so OpenEvolve skips its mislabeled final checkpoint.
  OpenEvolve then logs "stopped due to shutdown request"; the best program is
  saved as usual.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from openevolve_bedrock import ledger as _ledger

logger = logging.getLogger(__name__)

BUDGET_ENV = "OE_BEDROCK_BUDGET_USD"
PRICING_ENV = "OE_BEDROCK_PRICING"

_original_run_evolution = None
_original_submit_iteration = None


def budget_from_env() -> Optional[Tuple[float, dict]]:
    """(budget USD, pricing) from the environment, or None if no budget is set."""
    raw = os.environ.get(BUDGET_ENV)
    if not raw:
        return None
    try:
        budget = float(raw)
    except ValueError:
        raise ValueError(f"openevolve-bedrock: {BUDGET_ENV}={raw!r} is not a number")
    path = os.environ.get(PRICING_ENV)
    if not path:
        raise ValueError(f"openevolve-bedrock: {BUDGET_ENV} needs {PRICING_ENV} (pricing.yaml)")
    return budget, _ledger.load_pricing(Path(path))


def spent_usd(pricing: dict) -> float:
    """Priced total of this run's ledger so far."""
    path = _ledger.ledger_path()
    if path is None or not path.is_file():
        return 0.0
    return _ledger.summarize(_ledger.read(path), pricing).total_usd


def _check_priced(models, pricing: dict) -> None:
    missing = [m.name for m in models if _ledger._match_price(m.name, pricing) is None]
    if missing:
        raise ValueError(
            "openevolve-bedrock: a budget needs a price for every model; "
            f"{os.environ.get(PRICING_ENV)} has none for {', '.join(missing)}"
        )


def _submit_iteration(self, iteration, *args, **kwargs):
    budget = getattr(self, "_oe_bedrock_budget", None)
    if budget is not None:
        if self._oe_bedrock_stop == "budget":
            return None
        total = spent_usd(budget[1])
        if total >= budget[0]:
            logger.warning(
                "Budget $%.4f reached ($%.4f spent); submitting no more iterations, "
                "finishing those in flight", budget[0], total,
            )
            self._oe_bedrock_stop = "budget"
            return None
    return _original_submit_iteration(self, iteration, *args, **kwargs)


async def _run_evolution(self, start_iteration, max_iterations, target_score=None,
                         checkpoint_callback=None, *args, **kwargs):
    self._oe_bedrock_stop = None
    self._oe_bedrock_budget = budget_from_env()
    if self._oe_bedrock_budget is not None:
        _check_priced(self.config.llm.models, self._oe_bedrock_budget[1])
        logger.info("Budget: stop at $%.4f (%s)", self._oe_bedrock_budget[0],
                    os.environ.get(PRICING_ENV))
    result = await _original_run_evolution(
        self, start_iteration, max_iterations, target_score, checkpoint_callback, *args, **kwargs
    )
    last = self.database.last_iteration
    final = start_iteration + max_iterations - 1
    if 0 < last < final and not self.shutdown_event.is_set():
        reason = self._oe_bedrock_stop or "target score"
        logger.info("Stopped early (%s) at iteration %d", reason, last)
        if checkpoint_callback and last % self.config.checkpoint_interval != 0:
            checkpoint_callback(last)
        self.shutdown_event.set()  # OpenEvolve then skips checkpoint_{final}
    return result


def install() -> None:
    """Wrap the two controller methods (idempotent)."""
    global _original_run_evolution, _original_submit_iteration
    try:
        from openevolve.process_parallel import ProcessParallelController as C
    except ImportError:
        return
    if not (hasattr(C, "run_evolution") and hasattr(C, "_submit_iteration")):
        logger.debug("OpenEvolve layout not recognised; budget stop not installed")
        return
    if C.run_evolution is not _run_evolution:
        _original_run_evolution = C.run_evolution
        C.run_evolution = _run_evolution
    if C._submit_iteration is not _submit_iteration:
        _original_submit_iteration = C._submit_iteration
        C._submit_iteration = _submit_iteration
