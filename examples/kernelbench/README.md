# KernelBench GPU kernels

Evolve a CUDA or Triton kernel for one [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
problem, scored against the fastest thing PyTorch does on its own (issue #29). This is the first
GPU problem: one evaluation compiles and times a kernel on a GPU host, about 1 minute.

## Files

| file | what it is |
|---|---|
| `evaluator.py` | static check on this machine, then the GPU run on `KB_HOST` over SSH |
| `eval_kb.py` | runs on the GPU host, in the PyTorch container: correctness, hack checks, timing against the bar |
| `prepare.py` | downloads the problems and KernelBench's static checker at the pinned commit, writes the initial program, sets up the GPU host |
| `config.yaml` | OpenEvolve config: Bedrock models, prompt, cascade |

The problems and `kernel_static_checker.py` are KernelBench's (MIT, Copyright (c) 2023 Anne Ouyang,
Simon Guo, Azalia Mirhoseini, Stanford Scaling Intelligence Lab). `prepare.py` downloads them at
commit `423217d` into `.cache/`; they aren't committed here.

## Fitness

    combined_score = bar_ms / candidate_ms

The bar is the fastest of PyTorch eager, `torch.compile(mode="default")` and
`torch.compile(mode="max-autotune")`, timed in the same process as the candidate on the same inputs.
Timing uses CUDA events with the L2 cache flushed before each trial, about 1 s of trials per
measurement. Candidate and bar alternate (A B B A A B).

Eager alone is a weak reference. On the RTX 5090, `torch.compile` beats eager by 1.13× (level 1)
and 1.45× (level 2) geomean, and the best of the three by 1.23× and 1.48×. A kernel that only fuses
what `torch.compile` already fuses scores about 1.

A candidate scores 0 when:
- **it fails the static check.** KernelBench's checker runs with these checks as errors: code
  bypass (`try`/`except`, `pass`), timing patches, thread and stream injection, lazy evaluation,
  PyTorch compute ops or `torch.nn` modules doing the work, precision downgrade, and having no
  `__global__` or `@triton.jit` kernel. Also rejected: `torch.compile`, dynamo, inductor, private
  `torch._` APIs, TF32, `subprocess`, `importlib`. This is stage 1 and runs locally in milliseconds;
- **its output is wrong** (atol = rtol = 1e-2) on any of 3 fresh random input sets;
- **it modifies its inputs** (checked by a sum and ~4096 sampled elements, since inputs can be
  several GB), **returns an input**, or **returns a stale result** when called on new inputs;
- **it fails to compile or load, or the GPU run fails or times out.**

The GPU run uses `docker run --network none` with only the work directory mounted.

## Calibration (issue #29)

- **Noise.** On an RTX PRO 6000 Blackwell (g7e), timing each of 200 references against itself
  gives a median difference of 0.01–0.02%, and 0.15–0.9% at p90. Over 50 problems the geomean null
  has sd 0.05%. On the RTX 5090 (ceres) the p90 is 1.6–1.7%.
- **Memory.** About 23 level-1 problems need more than the 5090's 32 GB.
- **Check paths.** A hand-written CUDA ReLU (level 1, problem 19, 6.4 GB input) scored 0.983 on the
  5090, against `torch.compile` default as the bar. Test candidates were caught:
  - a wrong kernel ("wrong output … max abs diff 0.5");
  - a cached result;
  - an in-place kernel ("output aliases an input");
  - the reference itself and a `torch.compile` wrapper, both statically.

## Run

```bash
export KB_PROBLEM=level2/12_Gemm_Multiply_LeakyReLU KB_HOST=ceres.local
python examples/kernelbench/prepare.py
oe-bedrock run examples/kernelbench/.cache/programs/level2_12_Gemm_Multiply_LeakyReLU.py \
    examples/kernelbench/evaluator.py --config examples/kernelbench/config.yaml \
    --iterations 50 --budget-usd 5 --pricing pricing.yaml
```

The GPU host needs Docker with the NVIDIA runtime, usable without sudo, and SSH access with a key.

| environment | default | meaning |
|---|---|---|
| `KB_PROBLEM` | (required) | `level1/…` or `level2/…`, as in KernelBench |
| `KB_HOST` | (required) | SSH target of the GPU host |
| `KB_REMOTE_DIR` | `oeb-kernelbench` | work directory on the host, under `~` |
| `KB_IMAGE` | `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel` | container |
| `KB_TIMEOUT` | 900 | seconds per GPU evaluation |
| `KB_HOME` | `examples/kernelbench/.cache` | local cache |
