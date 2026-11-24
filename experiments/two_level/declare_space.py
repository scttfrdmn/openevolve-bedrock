# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
#60: ask a model to declare a parameter space for a program, without showing it
any results or known winners, then check whether the known winners fall inside.

    python experiments/two_level/declare_space.py                 # print prompts and the cost estimate
    python experiments/two_level/declare_space.py --go --pricing pricing.yaml --out declared.jsonl

The prompt contains only the starting program and the problem's description
(what is measured, what may not change). Nothing from #13, #28, #30 or #43.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]

INSTRUCTIONS = """You are designing the search space for a tuner. Read the program below and
declare the parameters a tuner should search, as JSON only:

{"parameters": [{"name": ..., "type": "int" | "float" | "bool" | "categorical",
                 "low": ..., "high": ..., "values": [...], "log_scale": true|false,
                 "default": ..., "why": "..."}],
 "constraints": ["..."]}

Rules: a parameter must be something the program could expose without changing what is computed
(the operation is fixed; only how it is scheduled or configured may vary). Use ranges you would
expect to contain the best setting. Defaults are the program's current values. Do not add
commentary outside the JSON."""

TARGETS = {
    "qiskit_level2": {
        "describe": ("Qiskit 2.6 transpiles circuits for IBM Torino (133 qubits, heavy-hex, CZ basis). "
                     "Fitness is fewer CZ gates than preset optimization level 2, within 2x its CPU time; "
                     "outputs must stay equivalent to the input circuit. The program returns preset level 2, "
                     "whose layout stage is SabreLayout(max_iterations=2, swap_trials=20, layout_trials=20)."),
        "program": ROOT / "examples/qiskit_routing/initial_program.py",
    },
    "kissat_restart": {
        "describe": ("Kissat 4.0.4's restart policy (src/restart.c). Fitness is PAR-2 in search ticks on SAT "
                     "Competition instances at a fixed conflict budget; answers must stay correct. Options "
                     "visible to the policy include restartint, restartmargin, and the fast/slow glue EMAs."),
        "program": ROOT / "examples/kissat/initial_program.c",
    },
    "kissat_mode": {
        "describe": ("Kissat 4.0.4's focused/stable mode switching (src/mode.c). Fitness as above. The policy "
                     "sets how long each mode phase lasts; per-mode EMAs of glue and decision rate are visible."),
        "program": ROOT / "examples/kissat/initial_program_mode.c",
    },
}
MODELS = ("us.anthropic.claude-sonnet-5-5", "us.anthropic.claude-haiku-5-5")
SAMPLES = 3
OUT_TOKENS = 2000


def prompt(target: str) -> str:
    t = TARGETS[target]
    code = t["program"].read_text()
    m = re.search(r"EVOLVE-BLOCK-START.*?\n(.*?)(?://|#|/\*)\s*EVOLVE-BLOCK-END", code, re.S)
    return f"{INSTRUCTIONS}\n\nProblem: {t['describe']}\n\nProgram (the part that may change):\n```\n{m.group(1) if m else code}```\n"


def estimate(pricing: dict) -> float:
    usd = 0.0
    for target in TARGETS:
        tin = len(prompt(target)) / 4
        for model in MODELS:
            p = pricing.get(model, {})
            usd += SAMPLES * (tin / 1000 * p.get("input", 0) + OUT_TOKENS / 1000 * p.get("output", 0))
    return usd


async def ask(model: str, text: str) -> str:
    from openevolve_bedrock import BedrockLLM

    cfg = SimpleNamespace(name=model, api_base="aws://us-west-2", api_key=None, temperature=0.7, top_p=None,
                          max_tokens=OUT_TOKENS, timeout=300, retries=None, retry_delay=None, random_seed=None,
                          reasoning_effort=None)
    return await BedrockLLM(cfg).generate(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="make the model calls (costs money)")
    ap.add_argument("--pricing", default=str(ROOT / "pricing.yaml"))
    ap.add_argument("--out", default="declared.jsonl")
    a = ap.parse_args()
    import yaml

    pricing = (yaml.safe_load(Path(a.pricing).read_text()) or {}).get("models", {}) if Path(a.pricing).is_file() else {}
    n = len(TARGETS) * len(MODELS) * SAMPLES
    print(f"{n} calls ({len(TARGETS)} targets x {len(MODELS)} models x {SAMPLES} samples); "
          f"estimated ${estimate(pricing):.2f}")
    if not a.go:
        for t in TARGETS:
            print(f"\n===== {t} ({len(prompt(t))} chars) =====\n{prompt(t)[:1500]}")
        return 0
    with open(a.out, "a") as f:
        for t in TARGETS:
            for model in MODELS:
                for s in range(SAMPLES):
                    text = asyncio.run(ask(model, prompt(t)))
                    f.write(json.dumps({"target": t, "model": model, "sample": s, "response": text}) + "\n")
                    print(t, model, s, len(text), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
