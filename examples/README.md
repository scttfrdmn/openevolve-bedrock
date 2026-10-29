# Examples

`initial_program.py` and `evaluator.py` in each directory are copied verbatim
from [OpenEvolve](https://github.com/codelion/openevolve) (Apache License 2.0,
Copyright Asankhaya Sharma). Only the `config.yaml` files are new: they point
the same problems at Amazon Bedrock.

| directory | what it is | why it is here |
|---|---|---|
| `function_minimization/` | 10-iteration toy problem | cheapest end-to-end check of credentials, model access and the ledger |
| `circle_packing/` | n=26 circle packing, the AlphaEvolve paper's Figure 14 problem | the standard reference point: AlphaEvolve 2.635, OpenEvolve 2.634 |

Before the first run:

```bash
cp ../src/openevolve_bedrock/pricing.example.yaml ../pricing.yaml   # fill in prices
oe-bedrock models --profiles --vendor anthropic                       # confirm model ids
oe-bedrock check --config circle_packing/config.yaml                  # one call per model
```
