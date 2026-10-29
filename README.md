# openevolve-bedrock

Amazon Bedrock as the LLM backend for [OpenEvolve](https://github.com/codelion/openevolve),
the open-source AlphaEvolve-class evolutionary coding agent — plus a per-call
usage ledger so a run reports what it cost, not just what it found.

```yaml
llm:
  provider: bedrock
  api_base: aws://us-west-2
  models:
    - name: us.anthropic.claude-haiku-4-5-20251001-v1:0    # breadth
      weight: 0.8
    - name: us.anthropic.claude-sonnet-4-5-20250929-v1:0   # depth
      weight: 0.2
```

```bash
pip install openevolve-bedrock
oe-bedrock run initial_program.py evaluator.py --config config.yaml --iterations 100
oe-bedrock cost openevolve_output/ --pricing pricing.yaml --threshold 0.99
```

## What it does

OpenEvolve's loop is unchanged: prompt sampler → weighted LLM ensemble →
your `evaluate()` → MAP-Elites archive with islands. This package replaces the
model calls and adds accounting:

| | |
|---|---|
| **Two transports** | `api_base: aws://<region>` uses the **Converse API** through boto3 (SigV4, IAM roles, instance profiles, or a Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`). An `https://` `api_base` uses Bedrock's **OpenAI-compatible Chat Completions** (`bedrock-mantle.<region>.api.aws/v1` or `bedrock-runtime.<region>.amazonaws.com/openai/v1`) with a Bedrock API key. Same config, same ledger; swap the URL to compare endpoints. |
| **Prompt caching** | Converse transport adds a cache point after the system message. Models that reject it are detected on the first call and the cache is turned off for that model. Cache hits require the system message to clear the model's minimum (on the order of 1–2K tokens): put the stable problem description and instructions in `prompt.system_message`, not the user turn. `OE_BEDROCK_CACHE=off` disables it. |
| **Throttle-aware retry** | `ThrottlingException`, 429, `ServiceUnavailable`, `ModelNotReady` and connection errors are retried with exponential backoff and jitter (`OE_BEDROCK_MAX_ATTEMPTS`, default 8). Throttles are counted in the ledger so you can see what your island parallelism is costing in wall-clock. |
| **Parameter hygiene** | `seed` is never sent (Bedrock rejects it); `top_p: null` is never sent (openevolve #414); `reasoning_effort` maps to Claude extended thinking (`low/medium/high` → 1K/4K/16K budget, or `OE_BEDROCK_THINKING_BUDGET=N`) or to `reasoning_effort` for `openai.*` models. |
| **Ledger** | One JSON line per call: model, transport, input/output/cache-read/cache-write tokens, latency, attempts, throttles, stop reason. Written with `O_APPEND` so OpenEvolve's worker processes share one file. Default location `<output_dir>/bedrock_ledger.jsonl`. |
| **Cost and $/result** | `oe-bedrock cost` prices the ledger from a `pricing.yaml` you fill from the Bedrock pricing page. `--threshold X --metric combined_score` finds the first checkpoint that reached X and reports spend up to that point. |

## Install

```bash
pip install openevolve-bedrock           # or: pip install -e . from a checkout
cp $(python -c 'import openevolve_bedrock,os;print(os.path.dirname(openevolve_bedrock.__file__))')/pricing.example.yaml pricing.yaml
```

Fill `pricing.yaml` (USD per 1K tokens; cross-region prefixes `us.`/`eu.`/… are
matched automatically). Prices are not shipped because they change; the tool
reports any model it can't price.

Credentials: whatever boto3 finds (`AWS_PROFILE`, instance role, SSO), or a
Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`. Region from `api_base`,
`OE_BEDROCK_REGION`, or `AWS_REGION`.

## Run

```bash
oe-bedrock models --profiles --vendor anthropic      # ids enabled in your region
oe-bedrock check --config config.yaml                # one 8-token call per model
oe-bedrock check --config config.yaml --cache-probe  # long system prompt: do cache tokens show?
oe-bedrock run initial_program.py evaluator.py --config config.yaml --iterations 100
oe-bedrock cost openevolve_output/ --pricing pricing.yaml
```

`oe-bedrock run` is OpenEvolve's own CLI with the provider registered and the
ledger pointed at the output directory; extra flags pass through. The Python API works
too — importing the package is the registration:

```python
import openevolve_bedrock  # registers provider: bedrock
from openevolve import OpenEvolve
```

If you launch OpenEvolve some other way (`openevolve-run.py`, your own script)
and use process-parallel workers under the spawn start method (macOS default),
the workers must import this package too. Either put `import openevolve_bedrock`
at the top of the script that is `__main__`, or run `oe-bedrock install-hook`
once to add a `.pth` file that imports it in every interpreter of that environment.

### Mantle vs. Converse

Both endpoints bill the same per token. They differ in how throughput is
managed: `bedrock-runtime` has per-account RPM/TPM quotas; Mantle schedules and
queues for fair share with higher initial throughput, which suits the bursty
fan-out of island-parallel evolution. To compare on your account:

```yaml
llm:
  provider: bedrock
  api_base: https://bedrock-mantle.us-west-2.api.aws/v1
  api_key: "${AWS_BEARER_TOKEN_BEDROCK}"
```

Everything else stays the same. `cost` reports throttles per run, so the
comparison is wall-clock and throttle count at fixed `parallel_evaluations`.

## Experiments the ledger is built for

`bench/matrix.py` runs one problem under several ensemble mixes and seeds and
writes a CSV of configured and realized model mix, best metric,
iterations-to-threshold, $ total, $ to threshold, calls, throttles and wall time:

```bash
python bench/matrix.py examples/circle_packing --pricing pricing.yaml \
    --iterations 100 --threshold 0.99 --seeds 1 2 3 \
    --variant fast-only "haiku:1.0" \
    --variant 80-20     "haiku:0.8,sonnet:0.2" \
    --variant 50-50     "haiku:0.5,sonnet:0.5" \
    --variant open      "gpt-oss:0.8,sonnet:0.2"
```

Three measurements this makes cheap:

1. **Ensemble mix** — the paper's fast-heavy/strong-light split against all-fast
   and 50/50, on iterations-to-threshold and $. OpenEvolve seeds every
   worker's ensemble RNG identically, so all workers draw the same model
   sequence ([openevolve#554](https://github.com/algorithmicsuperintelligence/openevolve/issues/554));
   this package reseeds the sampler per iteration from `(llm_seed, iteration)`,
   which keeps runs reproducible with independent draws. The `realized_mix`
   column still reports what each run actually got: with 100 iterations a
   0.2 weight can land anywhere from roughly 0.12 to 0.28.
2. **Endpoint** — same config with `--api-base` set to the Mantle URL; compare
   throttles and wall-clock at the same island parallelism.
3. **Open-weight on Bedrock vs. local** — `openai.gpt-oss-120b` (or any
   open-weight model Bedrock serves) in the same loop, same evaluator, same
   seeds; the local side of that comparison is any OpenAI-compatible server
   (vLLM, llama.cpp) via a plain `api_base`.

`combined_score` for circle packing is `sum_radii / 2.635`, so `--threshold 0.99`
is 2.609 and `1.0` is the AlphaEvolve figure.

## Configuration reference

| key | meaning |
|---|---|
| `llm.provider: bedrock` | required; may be set per model instead |
| `llm.api_base` | `aws://<region>` (Converse) or an `https://` OpenAI-compatible Bedrock endpoint |
| `llm.api_key` | Bedrock API key for the `https://` transport; `"${VAR}"` is resolved from the environment |
| `llm.models[].name` | model id or inference-profile id (`us.`, `eu.`, `global.` …) |
| `temperature`, `top_p`, `max_tokens`, `timeout`, `reasoning_effort` | as in OpenEvolve; `retries`/`retry_delay` are superseded by `OE_BEDROCK_MAX_ATTEMPTS` |

| environment | meaning |
|---|---|
| `OE_BEDROCK_REGION`, `OE_BEDROCK_PROFILE` | default region; boto3 profile |
| `OE_BEDROCK_CACHE` | `auto` (default) or `off` |
| `OE_BEDROCK_LEDGER` | ledger path, or `off` |
| `OE_BEDROCK_MAX_ATTEMPTS` | retry budget for throttles/transient errors (default 8) |
| `OE_BEDROCK_THINKING_BUDGET` | enable Claude extended thinking with this token budget |
| `AWS_BEARER_TOKEN_BEDROCK` | Bedrock API key (both transports) |

## Tests

```bash
pip install -e ".[dev]"
pytest            # 29 tests, ~15 s, no AWS credentials needed
```

`tests/test_e2e_spawn.py` runs a real 40-iteration OpenEvolve job twice with
process-parallel workers under the spawn start method against a fake Mantle
server that returns programs, reports usage and throws 429s at random — the
case that matters for macOS users and for the retry path. It asserts that
per-model call counts match the per-iteration seeding exactly and repeat
across runs.

## Not in scope (yet)

- Evaluator fan-out to EC2. OpenEvolve's `evaluate()` runs in local worker
  processes; for expensive evaluators the next step is a shim that ships
  candidates to a pool and returns metrics. The ledger already separates token
  cost from everything else so the two tiers can be reported side by side.
- Bedrock batch inference. The loop is latency-bound per iteration, so batch
  jobs don't fit without restructuring generation into offline waves.

## License

Apache License 2.0 — Copyright 2026 Scott Friedman. OpenEvolve and the example
problems are Apache-2.0, Copyright Asankhaya Sharma; see `NOTICE`.
