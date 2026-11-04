# Kissat restart policy

Evolve the restart policy of [Kissat](https://github.com/arminbiere/kissat) 4.0.4, a
CDCL SAT solver, scored by PAR-2 on SAT Competition instances. This is the first
evaluator-heavy problem for openevolve-bedrock (issue #13): one evaluation runs 24
solver instances of up to 300 s each, against about 20 s for the LLM call that
wrote the candidate.

## Files

| file | what it is |
|---|---|
| `initial_program.c` | Kissat's `src/restart.c` with an EVOLVE-BLOCK around `kissat_restarting` and `kissat_update_focused_restart_limit` |
| `evaluator.py` | builds Kissat with the candidate, runs an instance set, scores it against stock |
| `prepare.py` | fetches and builds the pinned Kissat, downloads the instances, measures the stock baseline |
| `hamiltonian_split.jsonl` | the 40 instances, their split, known result and stock numbers from the Graviton calibration |
| `config.yaml` | OpenEvolve config: Bedrock models, prompt, cascade |
| `initial_program_mode.c`, `config_mode.yaml` | the second target: Kissat's `src/mode.c` with an EVOLVE-BLOCK around the mode-switching policy (`KISSAT_EVAL_SOURCE=mode.c`), and its config |
| `evaluator_pool.py` | the same evaluation, run on a spore.host `spawn pool` worker (issue #1) |
| `remote.py` | worker entry point for `evaluator_pool.py` |
| `stock_ticks.json` | stock Kissat's conflicts and ticks per instance set (machine-independent), used when no local `baseline.json` exists |
| `mixed_split.jsonl`, `mixed_split.stock_ticks.json` | a second instance list, 148 instances from 32 families, and its stock numbers (`KISSAT_EVAL_SPLIT_FILE=mixed_split.jsonl`) |
| `mixed_frontier_split.jsonl`, `mixed_frontier_split.stock_ticks.json` | the mixed split plus 45 instances stock fails that a 2024 competitor solves in 300 s |
| `LICENSE.kissat` | Kissat's MIT licence, which covers `initial_program.c` |

## Instances and split

The SAT Competition 2024 main track's `hamiltonian` family, from the
[Global Benchmark Database](https://benchmark-database.de/) (GBD; 40 instances,
0.4 MB compressed). It was chosen from a calibration run of stock Kissat on all 400
main-track instances. On Graviton4 (c8g.48xlarge), at a 300 s cap, Kissat solved
34 of these 40, with runtimes spread from 6 to 258 s. Most other families were
either mostly unsolved or solved in under a second (issue #13).

| split | instances | SAT / UNSAT | purpose |
|---|---|---|---|
| `train` | 24 solved by stock | 10 / 14 | fitness |
| `heldout` | 10 solved by stock | 4 / 6 | generalisation check |
| `frontier` | 6 unsolved by stock in 300 s | 6 / 0 | can a candidate solve what stock can't |

Within each answer class the solved instances are sorted by stock runtime and
evenly spaced ones are held out, so `train` and `heldout` have the same SAT share
and runtime range. The unsolved instances are all SAT, so they form their own set.

### Mixed split

`mixed_split.jsonl` is every main-track instance stock Kissat solved in 10–200 s
in the same calibration: 148 instances from 32 families, 83 SAT and 65 UNSAT
(3.3 CPU-hours for stock). Within each family and answer class, instances sorted
by stock runtime alternate between `train` (74) and `heldout` (74). The 200 s
limit and the 1200 s cap keep a candidate at the 3x conflict budget clear of the
wall cap. 12 `miter` instances are 170–350 MB compressed and need up to 15 GB
of memory, so on a pool use a memory-optimised type (r8g.8xlarge, 256 GB). Stock
measured on one worker in 13–16 min per half; its conflicts matched the
calibration on all 148.

Six policies that did well on the Hamiltonian split, on both halves (issue #13).
"geomean" is the geometric mean of stock ticks / candidate ticks per instance,
with an unsolved instance at the fixed penalty; the interval is a 95% bootstrap
over instances.

| policy | tick-PAR-2 score train / heldout | geomean train | geomean heldout | solved train / heldout |
|---|---|---|---|---|
| `restartmargin` 15 | 0.84 / 0.82 | 1.04 (0.90–1.20) | 1.04 (0.86–1.23) | 71 / 68 |
| `restartmargin` 20 | 0.78 / 0.76 | 0.99 (0.83–1.18) | 1.08 (0.92–1.25) | 67 / 72 |
| `restartmargin` 30 | 0.71 / 0.63 | 1.00 (0.80–1.27) | 0.99 (0.79–1.29) | 66 / 67 |
| focused restart interval log term × 1.5 | 0.67 / 0.96 | 0.87 (0.73–1.03) | 1.16 (0.97–1.39) | 66 / 71 |
| evolved, 1.2425 on Hamiltonian | 0.74 / 0.66 | 0.97 (0.82–1.14) | 0.91 (0.76–1.08) | 70 / 66 |
| evolved, 0.917 on Hamiltonian | 0.68 / 0.73 | 0.90 (0.74–1.08) | 1.02 (0.86–1.20) | 66 / 69 |

None is distinguishable from stock: every interval contains 1. The tick-PAR-2
score is below 1 for all six because the set is defined by stock solving it, so a
change can only lose instances. Each one lost (2–8 of 74 here) costs 6× stock's
ticks on it, which outweighs gains elsewhere. Effects differ by family. On its 29
Hamiltonian instances `restartmargin` 30 has geomean 1.56 (1.01–2.75), but on
`software-verification` it has 0.71, and on `heule-nol` 0.50.

### Mixed split with stock failures

On a set stock fully solves, a change can only lose instances.
`mixed_frontier_split.jsonl` adds 45 instances that stock didn't solve in the
300 s calibration, but that at least one SAT Competition 2024 main-track solver
solved in 300 s or less (the competition's `detailed_main.csv`). They are 30 SAT
and 15 UNSAT from 21 families, split 22 to `train` and 23 to `heldout` by family
and answer. Each has its own conflict budget (`budget` in the row): stock's
conflicts at 300 s. Stock ends every one at exactly that budget. An unsolved
instance costs twice the ticks of a run to the budget, which on these is stock's
own ticks (`budget_reached` in the stock file). One evaluation takes as long as
on the mixed split, about 16 min per half on an r8g.8xlarge, because the extra
instances run alongside.

The six policies, and as a null control stock with only its random seed
changed. The control is `solver->options.seed = N` in
`kissat_update_focused_restart_limit`, which runs before the seed is read, so it
equals `--seed=N`. Score is the combined score of this split (tick-PAR-2 ratio):

| program | train | heldout | solved lost (of 74 / 74) | stock failures solved (of 22 / 23) |
|---|---|---|---|---|
| `restartmargin` 15 | 1.07 | 1.15 | 3 / 5 | 5 / 7 |
| `restartmargin` 20 | 1.18 | 1.04 | 7 / 2 | 9 / 4 |
| `restartmargin` 30 | 1.05 | 1.01 | 8 / 7 | 6 / 6 |
| focused restart interval log term × 1.5 | 1.06 | 1.26 | 8 / 3 | 7 / 7 |
| evolved, 1.2425 on Hamiltonian | 1.08 | 1.02 | 4 / 7 | 7 / 6 |
| evolved, 0.917 on Hamiltonian | 1.04 | 1.05 | 8 / 4 | 7 / 6 |
| stock, seed 1 | 1.02 | 0.93 | 6 / 7 | 5 / 2 |
| stock, seed 2 | 0.99 | 1.06 | 5 / 5 | 5 / 6 |
| stock, seed 3 | 0.96 | 1.16 | 6 / 4 | 4 / 6 |
| stock, seed 4 | 1.04 | 1.17 | 7 / 3 | 6 / 7 |

A seed change alone scores 0.93–1.17 (mean 1.04, sd 0.09) and moves as many
instances as a policy change. The policies average 1.09. The difference is
within noise (Welch t = 1.2). Across the ten programs, train score doesn't
predict heldout (Spearman −0.12). At this size, one evaluation of one seed can't
separate a restart-policy effect of about 10% from the change of trajectory it
causes. A credible improvement has to clear the seed spread on both halves, or be
measured over several seeds.

### Several seeds

`KISSAT_EVAL_SEEDS=0,1,2,3` runs every instance at each Kissat seed
(`--seed=s`) and pools all runs into one tick-PAR-2, each run against stock at
the same seed. The stock file keys a run at seed s > 0 as `hash@s`. The mixed
split with stock failures has stock at seeds 0–3. Stock solves 72–74 of the 74
train instances depending on the seed, and every run that it doesn't solve ends
on its budget. The same ten programs, pooled over seeds 0–3 (the null programs
use seed N + 4s):

| program | train, 4 seeds | heldout, 4 seeds |
|---|---|---|
| `restartmargin` 15 | 0.98 | 1.00 |
| `restartmargin` 20 | 1.01 | 1.03 |
| `restartmargin` 30 | 1.01 | 0.99 |
| focused restart interval log term × 1.5 | 1.07 | 1.09 |
| evolved, 1.2425 on Hamiltonian | 1.03 | 0.86 |
| evolved, 0.917 on Hamiltonian | 1.02 | 1.03 |
| stock, seed offsets 1–4 | 1.01–1.11 | 0.92–1.00 |

Pooling four seeds cuts the null spread from sd 0.07 (train) and 0.17 (heldout)
for one seed to 0.05 and 0.03. With it, the policies average 1.01 and the nulls
1.00 (Welch t = 0.4). No policy is above every null on both halves, and train
doesn't predict heldout (Spearman −0.22). The Hamiltonian effects don't survive
either. On the 34 Hamiltonian instances, `restartmargin` 30 has a 4-seed geomean
of 1.11 (0.84–1.50), and one null has 1.15. Changes to the restart constants of
Kissat 4.0.4 have no effect measurable at about ±5% on these 193 instances. An
evaluation at 4 seeds costs 4×: about 65 min per half on one r8g.8xlarge, or 16
min spread over 4 workers.

## Which heuristic to evolve

A heuristic is worth evolving if a single change moves the 4-seed pooled score
beyond the null spread, in either direction. Nine one-option variants of stock
(`KISSAT_EVAL_OPTIONS`) on the mixed split with stock failures, at seeds 0–3,
with z in null standard deviations (nulls: train 1.04 ± 0.05, heldout 0.96 ±
0.03):

| variant | flags | train (z) | heldout (z) |
|---|---|---|---|
| focused mode only | `--stable=0` | 0.64 (−8.6) | 0.72 (−8.2) |
| no rephasing | `--rephase=0` | 0.79 (−5.3) | 0.73 (−7.6) |
| stable mode only | `--stable=2` | 0.88 (−3.3) | 0.83 (−4.4) |
| reduce less | `--reducehigh=700 --reducelow=300` | 1.00 (−0.7) | 0.87 (−3.2) |
| no chronological backtracking | `--chrono=0` | 0.93 (−2.2) | 1.02 (+2.0) |
| target phases in focused mode too | `--target=2` | 0.98 (−1.1) | 0.91 (−1.9) |
| later first mode switch | `--modeinit=1e4` | 1.00 (−0.7) | 0.92 (−1.5) |
| faster score decay | `--decay=100` | 0.96 (−1.6) | 0.94 (−0.8) |
| larger tier 2 | `--tier2=10` | 1.00 (−0.9) | 0.94 (−0.7) |

The reduce variant hit the wall cap on 97 runs, so its numbers are partly
wall-clock. Mode and rephasing carry weight, and the mode effect depends on the
family. Focused only needs 2.2× less search than stock on `random-circuits` and
2.9× more on `software-verification`. Stable only needs 1.3× less on
`argumentation` and 2.1× more on `random-circuits`. Choosing the best of the three
mode variants per family on train, then scoring that choice on heldout, gives
1.046. The same procedure over the four seed nulls gives 0.988.

So the second target is the mode-switching policy (`initial_program_mode.c`,
`config_mode.yaml`). It is Kissat's `src/mode.c` with `update_mode_limit` and
`kissat_switching_search_mode` moved together into the EVOLVE-BLOCK. The block
drops only verbose-only logging. Built from it, Kissat gives the same status,
conflicts and ticks as stock on 8 Hamiltonian instances × 2 seeds. To count as a
result, an evolved policy needs a pooled heldout score above the null maximum
(1.00) by more than 2 null sd, so at least 1.06.

## Fitness

Each instance runs with a conflict budget of 3× stock's conflicts on it (from the
Graviton calibration), plus a 300 s wall-time safety cap.

    ticks (default):  combined_score = tick-PAR-2(stock) / tick-PAR-2(candidate)
    time:             combined_score = PAR-2(stock) / PAR-2(candidate)

A score above 1 is better than stock. Tick-PAR-2 is the mean of Kissat's
`search_ticks` (its deterministic effort counter) per instance; an instance not
solved within its budget counts a fixed 2 × budget × stock's ticks on it (6× at
the default budget). For a given binary it is
the same on every run and every machine. Three runs of one instance gave
163,375 conflicts and 388,314,498 ticks each time, and Graviton4 recorded the same
conflict count.

Wall time isn't deterministic, and on a shared machine it isn't usable. On a
12-core Mac with load average 20–30, stock Kissat scored itself at 0.90 on one run
and 0.59 on the next: identical search, and 3 instances timed out on the second
run. Wall-clock PAR-2 is still reported (`par2_s`, `time_score`). Set
`KISSAT_EVAL_METRIC=time` to score on it on a dedicated instance type. That checks
the one assumption the tick metric makes: that a restart policy changes the
amount of search, not the cost of a tick.

A candidate scores 0 when:
- it gives a wrong answer: a status that contradicts the known result, or a SAT
  witness that doesn't satisfy the formula (every witness is checked);
- it crashes;
- it fails to build;
- its EVOLVE-BLOCK contains a preprocessor line, I/O, process control or inline
  assembly. Restart logic has no use for these, and they'd let a candidate fake
  output.

`conflicts_ratio` (candidate conflicts / stock conflicts on the same instances) is
also reported. All 29 instances solved both on a Mac and on Graviton4 gave identical
conflict counts.

Cascade:
- **stage 1** builds the candidate and runs the 3 fastest training instances. A
  candidate under half of stock's speed stops here.
- **stage 2** runs all 24.

## Run

```bash
pip install -e ".[dev]"
python examples/kissat/prepare.py          # ~20 min on a laptop; once per machine
oe-bedrock run examples/kissat/initial_program.c examples/kissat/evaluator.py \
    --config examples/kissat/config.yaml --iterations 50 \
    --budget-usd 20 --pricing pricing.yaml
KISSAT_EVAL_SPLIT=heldout python -c "import sys; sys.path.insert(0, 'examples/kissat'); \
    import evaluator; print(evaluator.evaluate('openevolve_output/best/best_program.c'))"
```

| environment | default | meaning |
|---|---|---|
| `KISSAT_EVAL_HOME` | `examples/kissat/.cache` | where `prepare.py` puts Kissat, instances and `baseline.json` |
| `KISSAT_EVAL_JOBS` | CPU count | instances run in parallel; must match between `prepare.py` and the run |
| `KISSAT_EVAL_SPLIT_FILE` | `hamiltonian_split.jsonl` | instance list; its stock numbers are `stock_ticks.json` for the default, else `<stem>.stock_ticks.json`, and `prepare.py` writes `baseline-<stem>.json` |
| `KISSAT_EVAL_CAP` | the stock file's cap (300; 1200 for `mixed_split`) | per-instance cap in seconds; must match `prepare.py --cap` |
| `KISSAT_EVAL_SPLIT` | `train` | set scored by `evaluate` / stage 2 |
| `KISSAT_EVAL_METRIC` | `ticks` | `ticks` (deterministic) or `time` (wall-clock PAR-2) |
| `KISSAT_EVAL_SOURCE` | `restart.c` | Kissat source file a candidate replaces: `restart.c` or `mode.c` |
| `KISSAT_EVAL_OPTIONS` | none | extra `--name=value` Kissat flags for the candidate's runs only, for screening heuristics; `prepare.py` refuses to measure stock with it set |
| `KISSAT_EVAL_SEEDS` | `0` | Kissat seeds each instance runs with; the score pools all runs, each against stock at the same seed |
| `KISSAT_EVAL_BUDGET` | 3 | conflict budget as a multiple of stock's conflicts; must match `prepare.py` |
| `KISSAT_EVAL_CACHE` | `<home>/eval-cache` | result cache by C token sequence; `off` disables |
| `KISSAT_EVAL_REPEATS` | `reuse` | `penalize`: an exact repeat of an evaluated program scores 0, so copies don't fill the population. Needs `KISSAT_EVAL_CACHE` set to a directory for that run only |

With the tick metric, a laptop gives the same scores as Graviton, only slower. Wall
time (`time` mode) belongs on a fixed, dedicated Graviton instance type.

## Search diversity

Without guidance the model copies the winning change from the top programs in
its prompt, whatever parent it was given. In a 60-iteration run, 19 of 42
programs in the final population were copies of the best, none of them bred from
it. The config's system message says the programs shown are already evaluated
and asks for a different modification. With the same settings and seed (issue #13):

| run | distinct programs in final population | copies of the best |
|---|---|---|
| no instruction | 18 of 42 | 19 |
| instruction | 27 of 44 | 2 |
| instruction + `KISSAT_EVAL_REPEATS=penalize` | 34 of 46 | 2 |

None of the three found a better policy than the
`2 * (kissat_logn (restarts) - 1)` change within 60 iterations.

## Run on a spawn pool (Graviton)

`evaluator_pool.py` sends each evaluation to a [spore.host](https://spore.host)
worker pool. Each worker builds Kissat once and fetches the instances, then scores
candidates exactly as `evaluator.py` would, against `stock_ticks.json`.

```bash
B=oeb-kissat-<account>-us-west-2                 # private bucket, instances staged under main_2024/
spawn pool create --run-id kissat-1 --workers 4 --instance-type c8g.8xlarge \
    --ttl 4h --idle-timeout 15m --cost-limit 5 --visibility-timeout 2400 \
    --s3-read $B --s3-write $B --region us-west-2
export OE_BEDROCK_POOL_RUN_ID=kissat-1 OE_BEDROCK_POOL_STAGING=s3://$B/pool \
       OE_BEDROCK_POOL_REGION=us-west-2 KISSAT_EVAL_INSTANCES_S3=s3://$B/main_2024
oe-bedrock run examples/kissat/initial_program.c examples/kissat/evaluator_pool.py \
    --config config.yaml --iterations 50 --budget-usd 20 --pricing pricing.yaml   # parallel_evaluations: 4
spawn pool drain --run-id kissat-1
oe-bedrock cost openevolve_output/ --pricing pricing.yaml   # LLM $ and evaluation $ separately
```

- Set `parallel_evaluations` to about 2× the number of workers. The extra
  iterations make their model calls while others are being evaluated, and the
  pool's queue holds them until a worker is free.
- Results are cached by the candidate's C token sequence (`KISSAT_EVAL_CACHE`;
  comments and spacing ignored). Evolution repeats programs often and tick
  fitness is deterministic, so a repeat is answered locally without a pool task.
- `--visibility-timeout` must exceed one evaluation, or SQS hands it to a second
  worker.
- One stage-2 evaluation runs 24 instances in parallel, so use a worker type with
  at least 24 vCPUs.
- Each evaluation's cost is spawn's `cost_estimate` for the task, recorded in the
  ledger as a `kind: "eval"` line. Worker idle and boot time isn't attributed to
  any task.

Measured 2026-11-02, 4× c8g.8xlarge, Kissat restart policy, seed 42 (issue #1):

| run | wall per iteration | pool $ per iteration | pool tasks / iteration |
|---|---|---|---|
| this Mac, one evaluation at a time (20 iterations) | 117 s | – | – |
| pool, 4 in flight (20 iterations) | 45.5 s | 0.070 | 1.45 |
| pool, 8 in flight (20 iterations) | 39.2 s | 0.062 | 1.40 |
| pool, 8 in flight, cache, 3 s poll (60 iterations) | **14.8 s** | **0.022** | **0.62** |

In the last run, 30 of 59 evaluations came from the cache. The run still ends
on a ~4 min tail, about one stage-2 evaluation running alone after the last
model call, so short runs pay it in full.

Measured 2026-10-29 on c8g.8xlarge ($1.276/h):
- Stock Kissat scored `tick_score` exactly 1.0 on both stages against the Mac-measured
  baseline (gcc 11 on Amazon Linux against clang on macOS: same ticks).
- Stage 1 took 52 s end to end and cost $0.009.
- Stage 2 (24 instances, 218 s on the worker) took 254 s end to end and cost $0.083.
- Worker setup (build and instance fetch) took about 13 s.

## Source

Kissat 4.0.4, commit `8af8e56f174b778aef3aa45af9f739b2a5f492c2`, © 2019–2025 Armin
Biere, Mathias Fleury, Florian Pollitt and contributors, MIT licence. Instances: SAT
Competition 2024, via GBD.
