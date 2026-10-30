# Examples

In `function_minimization/` and `circle_packing/`, `initial_program.py` and
`evaluator.py` are copied verbatim from [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)
(Apache License 2.0, Copyright Asankhaya Sharma), as is
`circle_packing/requirements.txt`. Only their `config.yaml` files are new: they
point the same problems at Amazon Bedrock. `kissat/` is this project's own; its
`initial_program.c` is Kissat's `src/restart.c` (MIT, see `kissat/LICENSE.kissat`).

| directory | what it is | why it is here |
|---|---|---|
| `function_minimization/` | 10-iteration toy problem | cheapest end-to-end check of credentials, model access and the ledger |
| `circle_packing/` | n=26 circle packing, the AlphaEvolve paper's Figure 14 problem | the standard reference point: AlphaEvolve 2.635, OpenEvolve 2.634 |
| `kissat/` | Kissat SAT solver restart policy, PAR-2 on 24 SAT Competition instances | evaluator-heavy: one evaluation is up to 24 × 300 s of solver time ([README](kissat/README.md)) |

Candidate programs run in the same interpreter as the evaluator. Upstream's
best circle packing (2.63598) uses `scipy.optimize.minimize`; without scipy
installed every such candidate scores 0, which caps the run.

Before the first run:

```bash
pip install -r circle_packing/requirements.txt                        # scipy, matplotlib
cp ../src/openevolve_bedrock/pricing.example.yaml ../pricing.yaml   # fill in prices
oe-bedrock models --profiles --vendor anthropic                       # confirm model ids
oe-bedrock check --config circle_packing/config.yaml                  # one call per model
```
