# openevolve-bedrock — working notes for Claude Code

Apache License 2.0, Copyright 2026 Scott Friedman. Keep the license header on
new source files.

## What this is

A provider plugin for OpenEvolve (≥ 0.4.0) that routes its LLM calls to Amazon
Bedrock and keeps a per-call usage ledger. The evolutionary loop itself lives in
OpenEvolve; this repo does not reimplement it. Three parts:

- `src/openevolve_bedrock/provider.py` — `BedrockLLM` (OpenEvolve
  `LLMInterface`), two transports (Converse via boto3; OpenAI-compatible via the
  `openai` SDK), retry, `register()`.
- `src/openevolve_bedrock/ledger.py` — JSONL append, pricing, cost summary,
  first-checkpoint-at-threshold.
- `src/openevolve_bedrock/cli.py` — `oe-bedrock run|models|check|cost|install-hook`.

## Invariants (tests enforce these)

- Importing the package registers `provider: bedrock`. Spawned OpenEvolve
  workers must also import it; `oe-bedrock run` and `python -m
  openevolve_bedrock` satisfy this because the `__main__` module imports the
  package. `tests/test_e2e_spawn.py` is the proof — keep it passing.
- Never send `seed` or `top_p: null` to Bedrock.
- Extra config for this provider rides on existing OpenEvolve fields
  (`provider`, `api_base`, `api_key`, `name`, `reasoning_effort`) or
  `OE_BEDROCK_*` env vars. OpenEvolve loads config with dacite in non-strict
  mode, so unknown YAML keys are silently dropped — a new key would never reach
  the provider and nothing would say so. Do not add new ones to the YAML schema.
- The ledger is append-only, one JSON object per line, written with
  `O_APPEND`. Don't buffer or rewrite it.
- No prices are hardcoded. `pricing.yaml` is user-supplied; `cost` reports
  unpriced models rather than guessing.

## Commands

```bash
pip install -e ".[dev]"
pytest                        # offline, ~3 s
oe-bedrock models --profiles --vendor anthropic
oe-bedrock check --config examples/function_minimization/config.yaml
oe-bedrock run examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py \
    --config examples/circle_packing/config.yaml --iterations 100
oe-bedrock cost examples/circle_packing/openevolve_output/ --pricing pricing.yaml --threshold 0.99
```

## Where the real-AWS unknowns are

- Model/profile ids in `examples/*/config.yaml` and `bench/matrix.py` aliases:
  confirm with `oe-bedrock models`.
- Converse request shape for Claude extended thinking
  (`additionalModelRequestFields.thinking`) — written from the API as
  documented, not exercised live.
- Whether the OpenAI-compatible endpoint reports `prompt_tokens_details.cached_tokens`.

## Out of scope for 0.1.0

Evaluator fan-out to EC2 — tracked in a GitHub issue, not here.

## Style

- Prose in docs: plain statements, numbers, mechanisms. No marketing framing.
- Propose changes as measurable experiments: what number moves, what run
  produces it.
