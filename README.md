# openevolve-bedrock

Amazon Bedrock as the LLM backend for [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve),
the open-source AlphaEvolve-class evolutionary coding agent — plus a per-call
usage ledger so a run reports what it cost, not just what it found.

```yaml
llm:
  provider: bedrock
  api_base: aws://us-west-2
  models:
    - name: us.anthropic.claude-haiku-5-5    # breadth
      weight: 0.8
    - name: us.anthropic.claude-sonnet-5-5   # depth
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
| **Two transports** | `api_base: aws://<region>` uses the **Converse API** through boto3 (SigV4, IAM roles, instance profiles, or a Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`). An `https://` `api_base` uses Bedrock's **OpenAI-compatible Chat Completions** (`bedrock-mantle.<region>.api.aws/v1` or `bedrock-runtime.<region>.amazonaws.com/openai/v1`) with a Bedrock API key, or short-term keys minted from your AWS credentials when none is set. Same config, same ledger; swap the URL to compare endpoints. |
| **Any Bedrock model** | Any text model Bedrock serves, from any vendor: Claude, Nova, Llama, Mistral, DeepSeek, Qwen, OpenAI, Moonshot, Z.AI, MiniMax, xAI, Gemma, Nemotron… Models disagree on what a request may contain (cache point, `temperature`, thinking and reasoning fields), so the provider sends the current-generation shape and, when Bedrock rejects a field, drops or swaps it for that model, retries without spending a retry attempt, and remembers for the rest of the process. One warning per model and adaptation is logged. No model table to go stale. |
| **Prompt caching** | Converse transport adds a cache point after the system message; models that don't take one are detected on the first call. Cache hits require the system message to clear the model's minimum — 512 tokens on current Claude models, up to 4096 on some (Claude Haiku 4.5): put the stable problem description and instructions in `prompt.system_message`, not the user turn. Some models cache on their own (xAI Grok). `OE_BEDROCK_CACHE=off` disables the cache point. |
| **Throttle-aware retry** | `ThrottlingException`, 429, `ServiceUnavailable`, `ModelNotReady` and connection errors are retried with exponential backoff and jitter (`OE_BEDROCK_MAX_ATTEMPTS`, default 8). Throttles are counted in the ledger so you can see what your island parallelism is costing in wall-clock. |
| **Parameter hygiene** | `seed` is never sent (Bedrock rejects it); `top_p: null` is never sent (openevolve #414); `temperature`/`top_p` are dropped for models that reject them (current Claude, GPT-5.5/6.x, Grok, Kimi K3). `reasoning_effort` maps to Claude adaptive thinking with `output_config.effort` (falling back to a `budget_tokens` budget on models without adaptive thinking, e.g. Haiku 4.5; `OE_BEDROCK_THINKING_BUDGET=N` asks for a budget), and to `reasoning_effort` or `reasoning.effort` for other vendors, dropped where neither is accepted. |
| **Ledger** | One JSON line per call: model, transport, input/output/cache-read/cache-write tokens, latency, attempts, throttles, stop reason. Written with `O_APPEND` so OpenEvolve's worker processes share one file. Default location `<output_dir>/bedrock_ledger.jsonl`. |
| **Cost and $/result** | `oe-bedrock cost` prices the ledger from a `pricing.yaml` you fill from the Bedrock pricing page. `--threshold X --metric combined_score` finds the first checkpoint that reached X and reports spend up to that point. |

## Install

```bash
pip install openevolve-bedrock           # or: pip install -e . from a checkout
cp $(python -c 'import openevolve_bedrock,os;print(os.path.dirname(openevolve_bedrock.__file__))')/pricing.example.yaml pricing.yaml
```

Fill `pricing.yaml` (USD per 1K tokens; cross-region prefixes `us.`/`eu.`/… are
matched automatically, but `us.` and `global.` profiles are priced differently,
so key the exact id to price it exactly). Prices are not shipped because they
change; the AWS Price List API (`AmazonBedrockFoundationModels`) has them per
region. The tool reports any model without `input`/`output` prices; empty
`cache_read`/`cache_write` count as 0.

Credentials: whatever boto3 finds (`AWS_PROFILE`, instance role, SSO), or a
Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`. The `https://` transport needs a
Bedrock API key; without one it signs short-term keys from the boto3
credentials (as `aws-bedrock-token-generator` does) and re-signs them every
15 minutes. Region from `api_base`,
`OE_BEDROCK_REGION`, or `AWS_REGION`.

## Run

```bash
oe-bedrock models --profiles                         # text models, profiles, lifecycle (LEGACY ...)
oe-bedrock models --profiles --vendor mistral        # one vendor
oe-bedrock check --config config.yaml                # one short call per model
oe-bedrock check --config config.yaml --cache-probe  # ~5K-token system prompt twice: does the 2nd read the cache?
oe-bedrock run initial_program.py evaluator.py --config config.yaml --iterations 100
oe-bedrock cost openevolve_output/ --pricing pricing.yaml
```

### Stopping early

- `--target-score T` is OpenEvolve's own: stop once a program's
  `combined_score` reaches T.
- `--budget-usd X --pricing pricing.yaml` stops once the run's ledger, priced
  with that file, reaches $X. It is checked before each new iteration is
  submitted. Iterations already in flight finish and are recorded, so the
  overshoot is at most `2 × parallel_evaluations` calls. Every model in the
  config must have a price, or the run refuses to start. Also settable as
  `OE_BEDROCK_BUDGET_USD` / `OE_BEDROCK_PRICING`.

After either stop, the last checkpoint is the iteration where the run
stopped. On its own, OpenEvolve labels its final checkpoint with the maximum
iteration (`checkpoint_100` for a run that stopped at 57). OpenEvolve then logs
"stopped due to shutdown request", which is expected.

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
  # api_key: "${AWS_BEARER_TOKEN_BEDROCK}"   # optional; else minted from AWS credentials
```

Everything else stays the same. `cost` reports throttles per run, so the
comparison is wall-clock and throttle count at fixed `parallel_evaluations`.

The transport is chosen per model, so one ensemble can mix them. Set
`api_base` on a model to override the `llm`-level one:

```yaml
llm:
  provider: bedrock
  api_base: aws://us-west-2                     # default: Converse
  models:
    - name: us.anthropic.claude-haiku-5-5       # Converse
      weight: 0.6
    - name: us.openai.gpt-6.1-sol               # bedrock-runtime Chat Completions
      weight: 0.2
      api_base: https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1
    - name: openai.gpt-oss-120b                 # Mantle Chat Completions
      weight: 0.2
      api_base: https://bedrock-mantle.us-west-2.api.aws/v1
```

All three share one set of AWS credentials and one ledger; `cost` breaks the
run down per model.

Notes on the OpenAI-compatible endpoints, from the
[Bedrock endpoint docs](https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html)
and live runs (2026-10-28):

- AWS recommends `https://bedrock-runtime.<region>.amazonaws.com/openai/v1`
  for Chat Completions. It takes cross-region ids (`us.openai.gpt-6.1-sol`) and
  versioned ids (`openai.gpt-oss-120b-1:0`), but has no `GET /models`.
- Mantle has no cross-region profiles; ids are unversioned
  (`openai.gpt-oss-120b`) and `GET /v1/models` lists them. Its base path varies
  by model: most are under `/v1`, some (GPT-6.1 Sol) under `/openai/v1`, and
  the wrong one answers "isn't supported on this route". Each model card's
  Programmatic Access table gives the URL. Claude isn't served on Mantle's Chat
  Completions; use Converse.
- Mantle needs IAM `bedrock-mantle:CreateInference`; `bedrock-runtime` needs
  `bedrock:InvokeModel`.
- Prompt caching depends on the model. GPT-5.6 and later cache automatically
  (1,024-token minimum) and report `prompt_tokens_details.cached_tokens` and
  `cache_write_tokens`; the ledger records both. gpt-oss, Qwen, GLM and
  Devstral report no cache usage.
- On `bedrock-runtime`, gpt-oss returns its reasoning inline in `content` as a
  leading `<reasoning>…</reasoning>` span; the provider strips it.

## Experiments the ledger is built for

`bench/matrix.py` runs one problem under several ensemble mixes and seeds,
checkpointing every 5 iterations, and writes two CSVs under `--out`:

- `results.csv`, a row per run: configured and realized mix, `best@N`, then
  for each `--threshold` the iteration and $ at which it was first reached
  (blank if never), then $ total, calls, throttles and wall time.
- `curves.csv`, a row per checkpoint per run: best-so-far, cumulative $ and
  calls, seconds since the first call. This is the $/result curve: what each
  extra dollar bought, which no single threshold shows.

```bash
python bench/matrix.py examples/circle_packing --pricing pricing.yaml \
    --iterations 100 --threshold 0.99 0.995 0.999 --seeds 1 2 3 \
    --variant fast-only "haiku:1.0" \
    --variant gpt-oss   "gpt-oss:1.0" \
    --variant 80-20     "haiku:0.8,sonnet:0.2" \
    --variant 50-50     "haiku:0.5,sonnet:0.5"
```

Thresholds only affect scoring. Both CSVs are rebuilt from every run dir
under `--out`, so `--rescore` re-scores finished runs at other thresholds
without calling Bedrock. `--target-score` and `--budget-usd` are passed to
every run. A finished run is reused, so an interrupted matrix resumes; an
interrupted run's dir is moved aside (`*.interrupted-<time>`) and rerun from
scratch. In `--variant`, a model id that contains `:`
needs an explicit weight (`openai.gpt-oss-120b-1:0:1.0`); aliases don't.

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
3. **Open-weight on Bedrock vs. local** — gpt-oss-120b ships its weights in
   MXFP4, so Bedrock and a local server run the same weights and the
   comparison is $/result and wall time alone, with no quantization
   difference. Run it fast-only (`gpt-oss:1.0`) against Haiku fast-only to
   isolate the model. The local side is any OpenAI-compatible server (vLLM,
   llama.cpp) via a plain `api_base`.

`combined_score` for circle packing is `sum_radii / 2.635`, so `--threshold 0.99`
is 2.609, `0.995` is 2.622 and `1.0` is the AlphaEvolve figure.

## Configuration reference

| key | meaning |
|---|---|
| `llm.provider: bedrock` | required; may be set per model instead |
| `llm.api_base` | `aws://<region>` (Converse) or an `https://` OpenAI-compatible Bedrock endpoint |
| `llm.api_key` | Bedrock API key for the `https://` transport; `"${VAR}"` is resolved from the environment |
| `llm.models[].name` | model id or inference-profile id (`us.`, `eu.`, `global.` …) |
| `llm.models[].api_base`, `.api_key` | per-model override, e.g. to mix Converse and OpenAI-compatible models in one ensemble |
| `temperature`, `top_p`, `max_tokens`, `timeout`, `reasoning_effort` | as in OpenEvolve; `retries`/`retry_delay` are superseded by `OE_BEDROCK_MAX_ATTEMPTS` |

| environment | meaning |
|---|---|
| `OE_BEDROCK_REGION`, `OE_BEDROCK_PROFILE` | default region; boto3 profile |
| `OE_BEDROCK_CACHE` | `auto` (default) or `off` |
| `OE_BEDROCK_LEDGER` | ledger path, or `off` |
| `OE_BEDROCK_MAX_ATTEMPTS` | retry budget for throttles/transient errors (default 8) |
| `OE_BEDROCK_THINKING_BUDGET` | enable Claude extended thinking with this token budget |
| `OE_BEDROCK_BUDGET_USD`, `OE_BEDROCK_PRICING` | stop the run at this spend, priced with this file (same as `run --budget-usd --pricing`) |
| `AWS_BEARER_TOKEN_BEDROCK` | Bedrock API key (both transports; optional for `https://`, which can mint one) |

## Tests

```bash
pip install -e ".[dev]"
pytest            # 50 tests, ~15 s, no AWS credentials needed
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
