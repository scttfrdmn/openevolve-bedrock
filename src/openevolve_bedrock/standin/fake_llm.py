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
``provider: oeb-fake``: a deterministic stand-in for a model, for tests and A/B
runs against the stand-in evaluator with no AWS account (issue #46).

From the parent program in the prompt (its ``CONFIG``), each reply is one of:

- **mutate**: one CONFIG setting moved one step (block x2 or /2, unroll to a
  neighbour, fuse_bias toggled);
- **duplicate**: the parent again with only a comment changed (a near-duplicate);
- **defect**: the parent with one of the planted defects applied;
- **junk**: a reply with no code block.

Rates come from ``OE_BEDROCK_FAKE_RATES`` (default
``mutate=0.6,duplicate=0.15,defect=0.15,junk=0.1``). The choice is seeded by
the prompt and the model's ``random_seed``, so a run reproduces exactly. Each
call is recorded in the ledger as a call to model ``oeb-fake`` with
token counts of len/4, so ``oe-bedrock cost`` prices it from pricing.yaml like
any model.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from typing import Dict, List

from openevolve_bedrock import ledger as _ledger

PROVIDER_NAME = "oeb-fake"
DEFAULT_RATES = {"mutate": 0.6, "duplicate": 0.15, "defect": 0.15, "junk": 0.1}

# The planted defects as edits to a kernel written like initial_program.py.
DEFECT_EDITS = {
    "mask_leak": ("    e = [math.exp(z[i] - m) if mask[i] else 0.0 for i in range(n)]\n    s = sum(e)",
                  "    e = [math.exp(z[i] - m) for i in range(n)]\n    s = sum(e[i] for i in range(n) if mask[i])"),
    "off_by_one": ("range(start, min(start + block, n))", "range(start, min(start + block, n) - 1)"),
    "dropped_bias": ("            z[i] = x[i] + bias[i]", "            z[i] = x[i]"),
    "wrong_at_largest": ("    n = len(x)\n", "    n = min(len(x), 100)\n"),
}
_CODE = re.compile(r"```python\n(.*?)```", re.S)
_CONFIG = re.compile(r'CONFIG = \{"block": (\d+), "unroll": (\d+), "fuse_bias": (True|False)\}')


def rates() -> Dict[str, float]:
    raw = os.environ.get("OE_BEDROCK_FAKE_RATES")
    if not raw:
        return dict(DEFAULT_RATES)
    out = {k: 0.0 for k in DEFAULT_RATES}
    for part in raw.split(","):
        k, _, v = part.partition("=")
        if k.strip() not in out:
            raise ValueError(f"OE_BEDROCK_FAKE_RATES: unknown kind {k!r}")
        out[k.strip()] = float(v)
    return out


def parent_program(text: str) -> str | None:
    blocks = [b for b in _CODE.findall(text) if _CONFIG.search(b)]
    return blocks[-1] if blocks else None


def mutate(code: str, rng: random.Random) -> str:
    m = _CONFIG.search(code)
    block, unroll, fuse = int(m.group(1)), int(m.group(2)), m.group(3) == "True"
    knob = rng.choice(["block", "unroll", "fuse"])
    if knob == "block":
        block = max(1, min(256, block * 2 if rng.random() < 0.5 else block // 2))
    elif knob == "unroll":
        opts = [1, 2, 4, 8]
        i = opts.index(unroll) if unroll in opts else 0
        unroll = opts[max(0, min(3, i + rng.choice([-1, 1])))]
    else:
        fuse = not fuse
    return code[:m.start()] + f'CONFIG = {{"block": {block}, "unroll": {unroll}, "fuse_bias": {fuse}}}' + code[m.end():]


def reply(prompt_text: str, seed: int = 0) -> tuple[str, str]:
    """(kind, response text) for a prompt; deterministic given (prompt, seed)."""
    rng = random.Random(hashlib.sha256(f"{seed}:{prompt_text}".encode()).hexdigest())
    r = rates()
    kind = rng.choices(list(r), weights=list(r.values()))[0]
    parent = parent_program(prompt_text)
    if parent is None or kind == "junk":
        return "junk", "I would try a larger block size, but I can't show the code right now."
    if kind == "duplicate":
        code = parent.rstrip() + f"\n# revisit {rng.randrange(10 ** 6)}\n"
    elif kind == "defect":
        name, (a, b) = rng.choice(sorted(DEFECT_EDITS.items()))
        code = parent.replace(a, b, 1) if a in parent else mutate(parent, rng)
        kind = f"defect:{name}" if a in parent else "mutate"
    else:
        code = mutate(parent, rng)
    return kind, f"Here is the revised program.\n\n```python\n{code}```\n"


class FakeLLM:
    """OpenEvolve LLMInterface backed by ``reply``."""

    def __init__(self, model_cfg) -> None:
        self.model = getattr(model_cfg, "name", None) or PROVIDER_NAME
        self.seed = getattr(model_cfg, "random_seed", None) or 0
        self.last_kind = None

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context("", [{"role": "user", "content": prompt}], **kwargs)

    async def generate_with_context(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> str:
        text = "\n".join(m.get("content", "") for m in messages)
        t0 = time.monotonic()
        kind, out = reply(text, self.seed)
        self.last_kind = kind
        _ledger.append(_ledger.Usage(model=self.model, transport="fake",
                                     input_tokens=len(system_message + text) // 4, output_tokens=len(out) // 4,
                                     latency_ms=(time.monotonic() - t0) * 1000, stop_reason=kind))
        return out


def register() -> None:
    from openevolve.llm import ensemble

    registry = getattr(ensemble, "_PROVIDER_REGISTRY", None)
    if isinstance(registry, dict):
        registry[PROVIDER_NAME] = lambda cfg: FakeLLM(cfg)
