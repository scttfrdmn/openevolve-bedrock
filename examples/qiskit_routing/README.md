# Qiskit transpiler pipeline

Evolve the pipeline Qiskit 2.6 uses to map circuits onto IBM Torino (133
qubits, heavy-hex, CZ), scored by CZ count against the preset optimization
level 2 on [Benchpress](https://github.com/Qiskit/benchpress) circuits (issue
#30). This is the second CPU problem after `kissat/`. It tests whether the
measurement method from #13 carries over: deterministic fitness, seed nulls, a
held-out split, and a compute-matched bar.

## Files

| file | what it is |
|---|---|
| `initial_program.py` | `build_pass_manager(backend, seed)` in an EVOLVE-BLOCK, returning preset level 2 |
| `evaluator.py` | source check, runs the candidate in worker processes, scores against the reference |
| `worker.py` | runs in the Qiskit interpreter: transpiles, checks basis/connectivity and equivalence |
| `prepare.py` | downloads the circuits (Benchpress at a pinned commit), writes the split, measures the reference |
| `split.jsonl` | 138 circuits and their train/heldout assignment |
| `config.yaml` | OpenEvolve config: Bedrock models, prompt, cascade |
| `config_no_l3.yaml` | the second run: preset level 3 unavailable (`QISKIT_EVAL_MAX_PRESET=2`), 200 iterations |
| `requirements.txt` | `qiskit==2.6.2`, `qiskit-ibm-runtime==0.51.0` (for `FakeTorino`) |

## Circuits and split

The circuits come from Benchpress's `feynman` and `qasmbench-{small,medium,large}`
directories (original files, not the `_transpiled` copies), at commit
`4c04b38`. A circuit is kept if it fits in 133 qubits and is unitary once final
measurements are removed: no mid-circuit measurement, reset or control flow.
That leaves 138 circuits. They alternate between `train` (71) and `heldout` (67)
within each directory.

## Fitness

    combined_score = geomean over (circuit, seed) of (CZ_level2 + 1) / (CZ_candidate + 1)

The score uses the training circuits at seeds 0 and 1, each against level 2 at
the same `seed_transpiler`. Above 1 means fewer CZ gates. For a given pipeline,
seed and Qiskit version, CZ counts are deterministic.

The score is a geometric mean because CZ counts span five orders of magnitude.
On `feynman`, five `hwb` circuits account for 98% of the total.

A candidate scores 0 when:
- an output uses a gate outside the target, or a 2-qubit gate on a pair that
  isn't a coupling edge;
- an output isn't equivalent to its input. The check simulates random product
  inputs through the transpiled circuit and the inverse of the original, and
  requires |0…0⟩. It runs on every output with at most 20 active qubits and
  2×10⁸ gates × 2^qubits; a single dropped CZ fails it;
- its total CPU time (all threads) is more than 2× level 2's on the same runs;
- its EVOLVE-BLOCK imports, defines classes, does I/O or introspection, or uses
  `approximation_degree`. A candidate can only compose Qiskit's own passes,
  which preserve semantics.

Stage 1 runs the 10 cheapest training circuits at seed 0, with equivalence
checks, as a correctness gate. Stage 2 runs all 71 at seeds 0 and 1, about 10 s
with 10 worker processes.

## Baseline and screen

With the same seeds against level 2, at 4 seeds, on the earlier 157-circuit set:

| pipeline | train | heldout | CPU × level 2 |
|---|---|---|---|
| level 2, other seeds (null) | 1.000 | 0.997 | 0.92 |
| Sabre `max_iterations` 4 | 1.010 | 1.007 | 1.02 |
| level 3 | 1.012 | 1.007 | 1.58 |
| Sabre 40 trials | 1.005 | 1.004 | 2.38 |
| best of 2 seeds | 1.015 | 1.016 | 1.82 |
| `dense` layout | 0.62 | 0.64 | 0.37 |

`lookahead` routing ran past 120 s on an 8-qubit adder.

Single-seed nulls range from 0.983 to 1.009, about 10× tighter than Kissat's.
Best-of-2 is the compute-matched bar: under the 2× CPU cap, an evolved pipeline
has to beat about 1.016 on heldout. Otherwise running level 2 twice does as
well.

## Result (issue #30)

50 iterations: Haiku 5.5 / Sonnet 5.5, local, 21 min, $0.52.
- 40 programs; 13 raised on Qiskit API mistakes, which stage 1 caught.
- Training scores: median 1.001, best 1.0153 at 1.6× CPU.
- The best programs are level 2 with level 3's layout and optimization stages
  swapped in. In effect evolution rebuilt level 3.

At 4 seeds:

| pipeline | train | heldout | CPU × level 2 (heldout) |
|---|---|---|---|
| null (level 2, seed + 100) | 1.001 | 0.993 | 0.94 |
| level 3 | 1.013 | 1.009 | 1.62 |
| evolved: level 2 + level 3 layout and optimization | 1.013 | 1.009 | 1.67 |
| evolved, layout from seed + 1 | 1.011 | 1.008 | 1.67 |
| best of 2 seeds (bar) | 1.017 | 1.019 | ~1.8–2 |

All 124 checked heldout outputs were equivalent. The evolved pipeline scores
the same as level 3 on heldout. It beats the null, but stays under the
compute-matched bar, so running level 2 twice is better. The seed-shift
variant's training edge (+0.002) was noise.

The programs, ledger and per-run results are in
`analysis/qiskit-30-2026-11-05/` in the example's S3 bucket.

### Second run: preset level 3 unavailable

`config_no_l3.yaml` with `QISKIT_EVAL_MAX_PRESET=2`, under which the worker
makes `generate_preset_pass_manager` raise above level 2:
- 200 iterations, local, about 1.5 h, $2.11.
- Training scores: median 1.001, best 1.0203 at 1.85× CPU.

The best program composes a new pipeline:
1. a CZ-guarded resynthesis loop on the logical circuit before layout;
2. VF2Layout with 5,000 trials, falling back to Sabre with 3 iterations and 24 trials;
3. two extra resynthesis rounds after routing.

At 4 seeds:

| pipeline | train | heldout (95% CI over circuits) | CPU × level 2 |
|---|---|---|---|
| best (the composition above) | 1.015 | 1.005 (0.998–1.013) | 1.65 |
| the same without the layout change | 1.006 | 0.999 (0.994–1.005) | 1.48 |
| level 3 (first run) | 1.013 | 1.009 | 1.62 |
| best of 2 seeds (bar) | 1.017 | 1.019 | ~1.8–2 |

The training score fell from 1.020 (2 seeds) to 1.015 (4 seeds) to 1.005 on
heldout. That is selection on the training seeds, which the 4-seed and heldout
checks are there to catch. All 246 checked outputs were equivalent. Neither
run beat level 3, let alone the compute-matched bar.

## Run

```bash
python -m venv ~/qiskit-venv && ~/qiskit-venv/bin/pip install -r examples/qiskit_routing/requirements.txt
export QISKIT_EVAL_PYTHON=~/qiskit-venv/bin/python
python examples/qiskit_routing/prepare.py      # circuits + level-2 reference on this machine, ~1 min
oe-bedrock run examples/qiskit_routing/initial_program.py examples/qiskit_routing/evaluator.py \
    --config examples/qiskit_routing/config.yaml --iterations 50 --budget-usd 5 --pricing pricing.yaml
QISKIT_EVAL_SPLIT=heldout python -c "import sys; sys.path.insert(0, 'examples/qiskit_routing'); \
    import evaluator; print(evaluator.evaluate('openevolve_output/best/best_program.py'))"
```

| environment | default | meaning |
|---|---|---|
| `QISKIT_EVAL_PYTHON` | (required) | interpreter with `requirements.txt` installed. Candidates run there with `-I`, not in the evaluator's process |
| `QISKIT_EVAL_HOME` | `examples/qiskit_routing/.cache` | circuits and `reference.json` |
| `QISKIT_EVAL_JOBS` | CPU count − 2 | worker processes |
| `QISKIT_EVAL_SEEDS` | `0,1` | seeds |
| `QISKIT_EVAL_SPLIT` | `train` | set scored by stage 2 |
| `QISKIT_EVAL_CPU_CAP` | 2 | CPU limit as a multiple of level 2's |
| `QISKIT_EVAL_DETAIL` | unset | add per-run results to the metrics |
| `QISKIT_EVAL_MAX_PRESET` | unset | highest level `generate_preset_pass_manager` may build; above it, it raises |

`reference.json` records level 2's CZ counts, which are the same on any machine
for Qiskit 2.6.2, and its CPU time, which belongs to the machine it was
measured on.
