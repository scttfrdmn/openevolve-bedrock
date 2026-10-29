# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Ensemble model draws were identical across OpenEvolve workers, because
  upstream seeds every worker's sampler with the same `llm_seed`
  (algorithmicsuperintelligence/openevolve#554). `register()` now wraps
  `_run_iteration_worker` to reseed the sampler per iteration from
  `(llm_seed, iteration)`: still reproducible, with independent draws. With
  `random_seed: null` the sampler is left alone. (#3)

### Added
- `realized_mix` column in `bench/matrix.py` results, computed from the ledger's
  calls per model. The script refuses to append to a `results.csv` with
  different columns.

## [0.1.0] - 2026-10-28

### Added
- `provider: bedrock` for OpenEvolve ≥ 0.4.0, registered on import through
  OpenEvolve's `_PROVIDER_REGISTRY`.
- Converse transport (`api_base: aws://<region>`): boto3/SigV4 or Bedrock API
  key; system-message cache point with per-model auto-disable on rejection;
  `reasoning_effort` → Claude extended thinking or `openai.*` reasoning effort;
  cross-region inference-profile ids.
- OpenAI-compatible transport (`api_base: https://…`) for `bedrock-mantle` and
  `bedrock-runtime/openai/v1`; never sends `seed` or `top_p: null`.
- Throttle-aware retry with jittered exponential backoff; throttles counted.
- JSONL usage ledger (tokens incl. cache read/write, latency, attempts,
  throttles, stop reason), `O_APPEND`-safe across OpenEvolve worker processes.
- `oe-bedrock` CLI: `run`, `models`, `check` (`--cache-probe`), `cost`
  (`--threshold` for $-to-result), `install-hook`.
- `bench/matrix.py` ensemble-mix × seed experiment runner → CSV.
- Examples: circle packing (n=26) and function minimization, with Bedrock configs.
- Tests: 24 unit tests with fake clients; one offline end-to-end run with
  process-parallel workers under the spawn start method against a fake Mantle
  server.

### Not verified
- No run against live Bedrock in this release; model ids in example configs
  and the extended-thinking request shape are to be confirmed with
  `oe-bedrock models` and `oe-bedrock check`.

[Unreleased]: https://github.com/scttfrdmn/openevolve-bedrock/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/scttfrdmn/openevolve-bedrock/releases/tag/v0.1.0
