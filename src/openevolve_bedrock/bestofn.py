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
Best-of-N control for the evolution loop (issues #30, #43, #53).

    oe-bedrock bestofn initial_program.py evaluator.py --config config.yaml --samples 200 --output out/

Every sample is one LLM call on the prompt OpenEvolve would build for the initial program as
the parent, with no population and no feedback: the same config, models, prompt templates,
diff handling, EVOLVE-BLOCK enforcement, code-length cap and evaluator (cascade, timeouts) as
an ``oe-bedrock run`` of the same config. ``--samples`` equals the run's ``--iterations``, so
both make the same number of LLM calls.

The model for sample ``i`` is drawn from the ensemble weights with the sampler seeded from
``(random_seed, i)``, the derivation ``register()`` uses for evolution iterations on
OpenEvolve 0.4.0, so the model mix matches and does not depend on concurrency.

LLM calls run ``OE_BEDROCK_BESTOFN_CONCURRENCY`` at a time (default 4); evaluations run
``evaluator.parallel_evaluations`` at a time, as in the run, so timing-based scores see the
same load. The output directory has the layout ``verify`` and ``compare`` read: one
checkpoint (``checkpoints/checkpoint_<N>/programs/*.json``) and ``best/``. Each sample's outcome
is a line in ``samples.jsonl``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

CONCURRENCY_ENV = "OE_BEDROCK_BESTOFN_CONCURRENCY"


def _child(cfg, parent: str, response: str):
    """The child program from one response, or (None, reason); mirrors OpenEvolve's iteration worker."""
    from openevolve.utils.code_utils import apply_diff_blocks, extract_diffs, parse_full_rewrite

    if response is None:
        return None, "LLM returned None response"
    if cfg.diff_based_evolution:
        try:
            blocks = extract_diffs(response, cfg.diff_pattern)
        except ValueError as e:
            return None, str(e)
        if not blocks:
            return None, "No valid diffs found in response"
        code, applied = apply_diff_blocks(parent, blocks)
        if applied == 0:
            return None, f"None of the {len(blocks)} SEARCH block(s) matched the parent program"
        if code == parent:
            return None, "Diff did not change the parent program"
    else:
        code = parse_full_rewrite(response, cfg.language)
        if not code:
            return None, "No valid code found in response"
        if code == parent:
            return None, "Rewrite is identical to the parent program"
    if getattr(cfg, "enforce_evolve_blocks", False):
        from openevolve.utils.code_utils import enforce_evolve_blocks
        try:
            code = enforce_evolve_blocks(parent, code)
        except ValueError as e:
            return None, str(e)
        if code == parent:
            return None, "All edits were outside the EVOLVE-BLOCK regions"
    if len(code) > cfg.max_code_length:
        return None, f"Generated code exceeds maximum length ({len(code)} > {cfg.max_code_length})"
    return code, None


async def _run(initial_program: str, evaluation_file: str, config: Optional[str], samples: int, out: Path) -> dict:
    from openevolve.config import load_config
    from openevolve.database import Program
    from openevolve.evaluator import Evaluator
    from openevolve.llm.ensemble import LLMEnsemble
    from openevolve.prompt.sampler import PromptSampler

    from openevolve_bedrock.provider import _iteration_seed

    cfg = load_config(config)
    seed = cfg.random_seed
    for m in cfg.llm.models:
        m.random_seed = seed
    ens = LLMEnsemble(cfg.llm.models)
    sampler = PromptSampler(cfg.prompt)
    ev_prompt = PromptSampler(cfg.prompt)
    ev_prompt.set_templates("evaluator_system_message")
    evaluator = Evaluator(cfg.evaluator, evaluation_file, LLMEnsemble(cfg.llm.evaluator_models), ev_prompt,
                          database=None, suffix=getattr(cfg, "file_suffix", ".py"))
    llm_gate = asyncio.Semaphore(max(1, int(os.environ.get(CONCURRENCY_ENV, "4"))))
    eval_gate = asyncio.Semaphore(max(1, int(cfg.evaluator.parallel_evaluations or 1)))

    parent_code = Path(initial_program).read_text()
    async with eval_gate:
        parent_metrics = await evaluator.evaluate_program(parent_code, "initial")
    parent = Program(id=str(uuid.uuid4()), code=parent_code, language=cfg.language, metrics=parent_metrics,
                     iteration_found=0)
    programs = [parent]
    log = (out / "samples.jsonl").open("a")

    async def one(i: int):
        prompt = sampler.build_prompt(current_program=parent_code, parent_program=parent_code,
                                      program_metrics=parent_metrics, previous_programs=[parent.to_dict()],
                                      top_programs=[parent.to_dict()], inspirations=[], language=cfg.language,
                                      evolution_round=i, diff_based_evolution=cfg.diff_based_evolution)
        rec = {"i": i, "ts": time.time()}
        try:
            async with llm_gate:
                if seed is not None:
                    ens.random_state.seed(_iteration_seed(seed, i))   # no await between this and the draw
                resp = await ens.generate_with_context(system_message=prompt["system"],
                                                       messages=[{"role": "user", "content": prompt["user"]}])
            code, why = _child(cfg, parent_code, resp)
            if code is None:
                rec["error"] = why
            else:
                pid = str(uuid.uuid4())
                async with eval_gate:
                    metrics = await evaluator.evaluate_program(code, pid)
                programs.append(Program(id=pid, code=code, language=cfg.language, parent_id=parent.id, generation=1,
                                        metrics=metrics, iteration_found=i))
                rec.update(id=pid, metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        except Exception as e:  # noqa: BLE001 - one bad sample is a lost sample, as in a run
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        log.write(json.dumps(rec) + "\n")
        log.flush()

    await asyncio.gather(*(one(i) for i in range(1, samples + 1)))
    log.close()

    def score(p):
        return float((p.metrics or {}).get("combined_score") or 0.0)

    ck = out / "checkpoints" / f"checkpoint_{samples}" / "programs"
    ck.mkdir(parents=True, exist_ok=True)
    for p in programs:
        (ck / f"{p.id}.json").write_text(json.dumps(p.to_dict()))
    best = max(programs, key=score)
    (out / "best").mkdir(exist_ok=True)
    (out / "best" / f"best_program{getattr(cfg, 'file_suffix', '.py')}").write_text(best.code)
    info = {"id": best.id, "iteration": best.iteration_found, "metrics": best.metrics, "samples": samples,
            "evaluated": len(programs) - 1}
    (out / "best" / "best_program_info.json").write_text(json.dumps(info, indent=1))
    return info


def run(initial_program: str, evaluation_file: str, config: Optional[str], samples: int, output: str) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_run(initial_program, evaluation_file, config, samples, out))
