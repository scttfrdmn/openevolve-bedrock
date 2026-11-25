# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- `examples/kissat`: a Kissat option the solver rejects no longer scores as a
  wrong answer. `KISSAT_EVAL_OPTIONS` is checked against the pinned Kissat's
  `options.h` (name and numeric range) before any run. If Kissat still refuses
  its command line, the evaluation is `evaluator_failed`. In #59,
  `--restartmargin=30` (the range is 0–25) had been reported as wrong answers
  on every instance (#73).
- Provider: when several calls to a new model were in flight and Bedrock
  rejected a field (e.g. `temperature` on Claude 5.5), only the first call
  adapted. The others raised, because the quirk was already learned when they
  got the same rejection for their older request. They now retry, at most 8
  adaptations per call; the quirk is still learned and logged once. This hit
  any OpenEvolve run with concurrent LLM calls on a model's first use.

### Added
- `experiments/two_level/numpy_standin`: the two-level stand-in, with real timing (#61).
  - **Operation:** a fixed specification, `C = relu(A @ B + b)` in float64. Its tolerance class (1e-9 × output scale) is part of the specification and is not a setting (#71).
  - **Structures:** `unfused`, `tiled_fused` and `tiled_k`. Each has a declared parameter space and an `emit(shape, params)`.
  - **Timing:** in a fresh process with BLAS/OpenMP/vecLib threads pinned to one; candidate and reference are timed alternately, and each configuration gets a timing interval.
  - **Planted defects:** a skipped tile, a dropped bias, wrong only at the largest shape, a float32 cast, and zeros.
  - **Noise floor:** the reference against itself across 10 processes has sd 0.20% (range 0.6%) pinned, against 1.51% (range 5.5%) unpinned.
- `openevolve_bedrock.tuner`: the two-level search's inner tuner (#64).
  - Searches a declared parameter space with no model, behind an interface an external tuner can replace.
  - Two tuners: random search (without replacement), and a genetic algorithm (tournament selection, uniform crossover, grid-neighbour mutation, constraint repair, one elite).
  - The budget counts distinct evaluations, so the two compare at equal cost.
  - The GA beat random search at equal evaluations on all three test spaces, so it is the default:
    - stand-in, 40 evaluations: 100% vs 50% of runs within 0.5% of the optimum;
    - the #58 Qiskit table, 40 evaluations: 94% vs 70%;
    - an 8-parameter coupled probe, 320 evaluations: 72% vs 0%.
- `oe-bedrock verify`: re-scores a run's best distinct programs on held-out cases at several seeds through an optional evaluator hook, `evaluate_heldout(program_path, seed)`.
  - Reports per-seed scores, the mean with a 95% t-interval, and the shrinkage from training. A program failing held-out correctness is reported as rejected.
  - Hooks for the stand-in and `qiskit_routing`.
  - On #30 run 2 it reproduces the hand-computed result: winner 1.0203 on training, 1.0050 on held-out.
  - The stand-in's held-out shapes now have the same count and total length as its search shapes, so both share the optimum (1.821) and shrinkage measures selection only (#41).
- **Fitness integrity** (`OE_BEDROCK_FITNESS_INTEGRITY=1`, off by default).
  - Every evaluation is classified as `scored`, `candidate_failed` or `evaluator_failed`, and each attempt gets a `kind: "outcome"` ledger line.
  - Evaluator failures are retried (`OE_BEDROCK_EVAL_RETRIES`, default 2), then the candidate is skipped instead of archived at 0.
  - `SpawnPool`, KernelBench's no-result path and the stand-in flag `evaluator_failed: 1.0`.
  - On the stand-in at 10% injected faults, the final archive had 1, 2 and 0 programs scored differently from their fault-free score over 3 seeds without it, and 0, 0, 0 with it (#47).
- **Evaluator self-test** (`oe-bedrock selftest`, or `OE_BEDROCK_SELFTEST=1` on `run`).
  - The reference must score, and null and planted candidates must fail; the run aborts otherwise.
  - Reports the false-accept rate, and records it in the ledger.
  - Planted sets for the stand-in, `qiskit_routing` and KernelBench `level1/19_ReLU`. All three evaluators measured 0% false accepts (0 of 5, 0 of 4, 0 of 5). Deliberately weakened stand-in evaluators measured 80% and 100% (#48).
- **`openevolve_bedrock.standin`.** A CPU stand-in evaluator (masked softmax + bias with a CONFIG whose cost has a known optimum, seeded noise, held-out shapes, injectable faults, planted defects) and `provider: oeb-fake`, a deterministic fake model logged in the ledger. With these, the search-quality capabilities (#57) are tested and A/B'd with no AWS account (#46).
- `examples/kernelbench/`: evolve a CUDA or Triton kernel for one KernelBench
  problem. Its score is relative to the fastest of PyTorch eager, `torch.compile`
  and `torch.compile(mode="max-autotune")` in the same GPU process.
  - Checks: KernelBench's static checker with its warnings as errors, plus bans
    on `torch.compile`, dynamo, inductor, private `torch._` APIs and TF32.
    Correctness on 3 fresh input sets, and runtime checks for input mutation,
    aliasing and stale results.
  - Evaluation runs on a GPU host over SSH (`KB_HOST`), in a container with no
    network.
  - Calibration: timing nulls of 0.01–0.02% median on an RTX PRO 6000 Blackwell;
    `torch.compile` alone is 1.13× (L1) and 1.45× (L2) over eager on an RTX
    5090 (#29).
- `examples/qiskit_routing/`: evolve a Qiskit 2.6 transpiler pipeline for IBM
  Torino. The score is the geomean CZ ratio against preset level 2 on 71
  Benchpress training circuits at 2 seeds. Every output is checked for basis
  and connectivity, and by random-input simulation for equivalence (up to 20
  active qubits). A 2× CPU cap applies. Candidates run in a separate Qiskit
  interpreter (`QISKIT_EVAL_PYTHON`, `-I`). A screen at 4 seeds gave a null
  spread of ±0.006, and best-of-2 seeds as the compute-matched bar at 1.016
  heldout (#30). A 50-iteration run ($0.52 of LLM, local) rebuilt level 3:
  heldout 1.009 at 1.67× CPU, below best-of-2's 1.019.
- `examples/qiskit_routing`: `QISKIT_EVAL_MAX_PRESET` and `config_no_l3.yaml`.
  A 200-iteration run without level 3 ($2.11) found a new composition
  (resynthesis loop, deep VF2 layout, Sabre fallback). It scored 1.020 on
  training at 2 seeds but 1.005 on heldout at 4 seeds, below level 3 and the
  best-of-2 bar (#30).

### Changed
- `examples/kissat/config_mode.yaml`: the prompt said the per-mode averages
  had an `.initialized` field next to the glue fields. In fact `.initialized`
  belongs to `solver->averages[i]` and each glue field is a `smooth` with
  `.value`. In the #28 run, 1 of 50 candidates failed to build on this.
- `examples/kissat/README.md`: the mode-switching evolution result (#28); none
  of the top four beats the seed nulls on heldout. Also notes on pool
  idle timeout, TTL and cost limit for long evaluations.

## [0.2.0] - 2026-11-04

### Added
- `examples/kissat`: `KISSAT_EVAL_SPLIT_FILE` selects the instance list, and
  `mixed_split.jsonl` is a second one: 148 main-track instances from 32
  families (83 SAT / 65 UNSAT) that stock solves in 10–200 s, halved into
  train and heldout by family and answer. Its stock numbers are in
  `mixed_split.stock_ticks.json` (cap 1200 s). `prepare.py --stock-ticks`
  writes that file, and `remote.py baseline:SET` measures it on a pool worker.
  On it, none of six restart policies that did well on Hamiltonian is
  distinguishable from stock (#13).
- `examples/kissat/mixed_frontier_split.jsonl`: the mixed split plus 45
  instances stock fails in 300 s but a 2024 competitor solves in 300 s, so a
  candidate can gain instances as well as lose them. A split row may set its own
  conflict `budget`. A stock run that ends at it is recorded as
  `budget_reached`, and an unsolved instance then costs twice stock's ticks
  there. Six policies score 1.01–1.26 on it, and stock with only its seed
  changed scores 0.93–1.17, so none of them is distinguishable from chance (#13).
- `examples/kissat`: `KISSAT_EVAL_SEEDS` runs each instance at several Kissat
  seeds and pools the runs, each against stock at the same seed. Stock at seeds
  0–3 is in `mixed_frontier_split.stock_ticks.json` (`hash@seed`). Four seeds
  cut the null spread from 0.07–0.17 to 0.03–0.05, and with that no restart
  policy tested differs from stock (#13).
- `examples/kissat`: a second target, the mode-switching policy:
  `initial_program_mode.c` (Kissat's `src/mode.c` with an EVOLVE-BLOCK) and
  `config_mode.yaml`, chosen by `KISSAT_EVAL_SOURCE=mode.c`.
  `KISSAT_EVAL_OPTIONS` passes extra Kissat flags to candidate runs for
  screening. In a nine-variant screen at 4 seeds, mode and rephasing changes
  moved the score by 3–9 null sd, with family-dependent sign, and a per-family
  mode choice made on train scored 1.046 on heldout against 0.988 for nulls (#13).
- `examples/kissat`: the system message tells the model that the programs shown
  are already evaluated and asks for a different modification. Copies of the
  best program in a 60-iteration run's final population fell from 19 to 2.
  `KISSAT_EVAL_REPEATS=penalize` scores exact repeats 0; distinct programs rose
  from 18 to 34 of ~45 (#13).
- `examples/kissat`: results are cached by the candidate's C token sequence,
  ignoring comments and spacing (`KISSAT_EVAL_CACHE`). In a 60-iteration
  pool run, 30 of 59 evaluations were repeats answered from the cache, and
  wall time per iteration fell from 39.2 s to 14.8 s.
- Pool evaluation ledger lines record when each task ran: `started_at` /
  `ended_at` from spawn's completion record, `run_s` (time on the worker),
  `wait_s` (time queued), `submitted_at` and `instance_type`. `eval_s`
  (submit to result) includes queue time, so worker utilization comes from
  `run_s`. `oe-bedrock cost` reports both, and `--json` gains
  `evals.worker_seconds`.
- `openevolve_bedrock.pool.SpawnPool`: run an evaluation on a spore.host
  `spawn pool` worker. The problem directory is staged as a content-addressed
  bundle, each candidate submitted as a TaskSpec, and `metrics.json` read back
  from S3. Configured by `OE_BEDROCK_POOL_*`. Each evaluation adds a
  `kind: "eval"` ledger line with spawn's `cost_estimate`, and `oe-bedrock cost`
  reports evaluation $ apart from LLM $ (#1).
- `examples/kissat/evaluator_pool.py` and `remote.py`: the Kissat evaluation on
  a Graviton pool. Workers build Kissat once and score against the committed
  `stock_ticks.json`.
- `examples/kissat/`: evolve Kissat 4.0.4's restart policy on 24
  Hamiltonian-cycle instances from the SAT Competition 2024 main track. Fitness
  is PAR-2 in Kissat's deterministic `search_ticks` relative to stock, with a
  per-instance conflict budget, so a score is the same on every run and machine.
  Wall-clock PAR-2 is reported too, and is the fitness with
  `KISSAT_EVAL_METRIC=time`. Every SAT witness is verified, a wrong answer,
  crash, build failure or forbidden code in the EVOLVE-BLOCK scores 0, and a
  two-stage cascade is included. `prepare.py`
  builds the pinned Kissat, downloads the instances from GBD and measures the
  stock baseline for each instance set. The split was chosen from a Graviton4
  calibration of all 400 main-track instances (#13).
- `oe-bedrock run --budget-usd X --pricing FILE` (or `OE_BEDROCK_BUDGET_USD` /
  `OE_BEDROCK_PRICING`): no new iterations once the priced ledger reaches $X;
  in-flight ones finish and are recorded. (#11)
- `bench/matrix.py --target-score` / `--budget-usd`, passed to every run.

### Changed
- `OE_BEDROCK_POOL_POLL` defaults to 3 s (was 10): the median time from a
  task ending to its result being seen fell from 6.8 s to 3.1 s.
- The per-iteration reseed of the model sampler is skipped on OpenEvolve
  0.4.1 and later, which reseeds per iteration itself (`LLMEnsemble.reseed`,
  upstream #559). On 0.4.1 ours was overridden anyway, and the spawn test,
  which predicted counts from our hash, failed there. The test now predicts
  from whichever derivation is active (#5).
- `bench/matrix.py`: `--threshold` takes several values, with `iter@T` and
  `usd@T` columns per value, blank if not reached; adds a `best@N` column; and
  writes `curves.csv` (best-so-far against cumulative $ per checkpoint) next to
  `results.csv`. Both are rebuilt from the run dirs on every invocation;
  `--rescore` re-scores without running. Finished runs are reused, so an
  interrupted matrix resumes. Generated configs checkpoint every 5 iterations.

### Fixed
- Pool evaluation: `OE_BEDROCK_POOL_TIMEOUT` counts only time running on a
  worker. Before, it counted from submit, so with more tasks queued than
  workers, the later ones returned "no completion record" and scored 0 while
  still running and billed. Each task now writes `<staging>/started/<task>` as
  its first command. Time waiting in the queue is bounded separately by
  `OE_BEDROCK_POOL_QUEUE_TIMEOUT` (default 24 h), and a task that never starts
  is recorded as `queue_timeout` (#24).
- After an early stop (`--target-score` or budget), the last checkpoint is
  saved at the iteration where the run stopped. OpenEvolve labelled it with
  the maximum iteration, so `iter@T` and the curves read a run that stopped at
  57 as reaching its best at 100. (#11)
- `bench/matrix.py` treats a run with a clean `run_meta.json` as finished (a
  run stopped early has no final checkpoint), and moves an interrupted run's
  dir aside instead of appending to its partial ledger.
- `examples/circle_packing/requirements.txt` (scipy, matplotlib), copied from
  upstream. Without it, candidates using `scipy.optimize` scored 0, which
  capped circle packing runs below upstream's result.
- `bench/matrix.py --variant` accepts model ids containing `:`
  (`openai.gpt-oss-120b-1:0:1.0`); the weight now follows the last `:`.

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

[Unreleased]: https://github.com/scttfrdmn/openevolve-bedrock/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/scttfrdmn/openevolve-bedrock/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/scttfrdmn/openevolve-bedrock/releases/tag/v0.1.0
