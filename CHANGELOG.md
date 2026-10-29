# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-28

### Added
- `provider: bedrock` for OpenEvolve ≥ 0.4.0, registered on import through
  OpenEvolve's `_PROVIDER_REGISTRY`.
- Converse transport (`api_base: aws://<region>`): boto3/SigV4 or Bedrock API
  key; cross-region inference-profile ids; system-message cache point.
- OpenAI-compatible transport (`api_base: https://…`) for `bedrock-mantle` and
  `bedrock-runtime/openai/v1`. Never sends `seed` or `top_p: null`. Without
  `api_key` or `AWS_BEARER_TOKEN_BEDROCK` it mints short-term Bedrock API keys
  from the AWS credentials, re-signed every 15 minutes. Records
  `cached_tokens` and `cache_write_tokens` separately from input. Strips
  gpt-oss's leading inline `<reasoning>` span. Reports a wrong base path
  ("isn't supported on this route") as such.
- The transport is chosen per model, so one ensemble can mix Converse, Mantle
  and `bedrock-runtime` Chat Completions models (`llm.models[].api_base`).
- Any Bedrock text model from any vendor (#6). Models disagree on what a
  request may contain, so the provider sends the current-generation shape and,
  when Bedrock rejects a field, adapts for that model and remembers: drops the
  cache point or `temperature`/`top_p`; switches Claude thinking between
  adaptive (`output_config.effort`) and `budget_tokens`; moves reasoning effort
  between `reasoning_effort`, `reasoning.effort` and none; switches
  `max_tokens` to `max_completion_tokens`.
- `reasoning_effort` maps to Claude adaptive thinking and to other vendors'
  reasoning effort.
- Per-iteration reseeding of the ensemble's model sampler from
  `(llm_seed, iteration)`, because upstream seeds every worker identically
  (algorithmicsuperintelligence/openevolve#554). Draws stay reproducible and
  become independent across workers. Left alone with `random_seed: null`. (#3)
- Throttle-aware retry with jittered exponential backoff; throttles counted.
- JSONL usage ledger (tokens incl. cache read/write, latency, attempts,
  throttles, stop reason), `O_APPEND`-safe across OpenEvolve worker processes.
- `oe-bedrock` CLI: `run`; `models` (profile-only models, lifecycle,
  vendor-prefix match, text-only profiles unless `--all`); `check`
  (`--cache-probe` sends a ~5K-token system prompt twice); `cost`
  (`--threshold` for $-to-result); `install-hook`.
- `bench/matrix.py` ensemble-mix × seed experiment runner → CSV, with a
  `realized_mix` column from the ledger.
- Examples: circle packing (n=26) and function minimization, on Claude Haiku
  and Sonnet 5.5; `pricing.example.yaml` for current models.
- Tests: 50 unit tests with fake clients; one offline end-to-end run with
  process-parallel workers under the spawn start method against a fake Mantle
  server.

### Verified live (us-west-2, us-east-1; #2)
- Converse: 25 models from 13 vendors, with and without `reasoning_effort`;
  cache reads and writes on Claude; a 10-iteration `function_minimization`
  run priced by `oe-bedrock cost` ($0.016).
- OpenAI-compatible, with minted keys: gpt-oss, Qwen, GLM, Devstral and
  GPT-6.1 Sol on Mantle and `bedrock-runtime`; GPT-6.1 Sol cache writes and
  reads; one ensemble mixing Converse, Mantle and `bedrock-runtime` models.

[Unreleased]: https://github.com/scttfrdmn/openevolve-bedrock/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/scttfrdmn/openevolve-bedrock/releases/tag/v0.1.0
