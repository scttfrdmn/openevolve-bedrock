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

## Fitness

Each instance runs with a conflict budget of 3× stock's conflicts on it (from the
Graviton calibration), plus a 300 s wall-time safety cap.

    ticks (default):  combined_score = tick-PAR-2(stock) / tick-PAR-2(candidate)
    time:             combined_score = PAR-2(stock) / PAR-2(candidate)

A score above 1 is better than stock. Tick-PAR-2 is the mean of Kissat's
`search_ticks` (its deterministic effort counter) per instance; an instance not
solved within its budget counts twice the ticks it used. For a given binary it is
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
| `KISSAT_EVAL_CAP` | 300 | per-instance cap in seconds; must match `prepare.py --cap` |
| `KISSAT_EVAL_SPLIT` | `train` | set scored by `evaluate` / stage 2 |
| `KISSAT_EVAL_METRIC` | `ticks` | `ticks` (deterministic) or `time` (wall-clock PAR-2) |
| `KISSAT_EVAL_BUDGET` | 3 | conflict budget as a multiple of stock's conflicts; must match `prepare.py` |

With the tick metric, a laptop gives the same scores as Graviton, only slower. Wall
time (`time` mode) belongs on a fixed, dedicated Graviton instance type, through the
spawn pool evaluator of issue #1.

## Source

Kissat 4.0.4, commit `8af8e56f174b778aef3aa45af9f739b2a5f492c2`, © 2019–2025 Armin
Biere, Mathias Fleury, Florian Pollitt and contributors, MIT licence. Instances: SAT
Competition 2024, via GBD.
