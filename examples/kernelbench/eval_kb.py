# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
GPU side of examples/kernelbench (issue #29): score one candidate on one problem.

    python eval_kb.py PROBLEM.py CANDIDATE.py OUT.json

Runs in a fresh process inside the PyTorch container. The reference ``Model``
comes from the KernelBench problem file; the candidate defines ``ModelNew``
with the same constructor and forward signature.

Correctness: on 3 fresh random input sets (seeds 1-3), the candidate's output
must match the reference's (atol = rtol = 1e-2, as KernelBench). Runtime
checks for the reward hacks KernelBench lists:
- input mutation: inputs must be unchanged after a candidate call;
- result reuse: called on input set B after set A, it must return B's answer;
- aliasing: the output must not be one of the inputs.

Timing: CUDA events, L2 flushed before each trial, ~1 s of trials per
measurement. Candidate and the bar are measured in alternation (A B B A A B).
The bar is the fastest of eager, torch.compile(mode="default") and
torch.compile(mode="max-autotune") on the same inputs in the same process.
"""

import importlib.util
import json
import statistics as S
import sys
import time
import traceback

import torch

ATOL = RTOL = 1e-2
SEEDS = (1, 2, 3)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inputs(mod, seed, dev):
    torch.manual_seed(seed)
    return [x.to(dev) if isinstance(x, torch.Tensor) else x for x in mod.get_inputs()]


def fingerprint(t):
    """Cheap stand-in for a clone (inputs can be several GB): sum plus ~4096 sampled elements."""
    flat = t.reshape(-1)
    step = max(1, flat.numel() // 4096)
    return flat.sum().item(), flat[::step].clone()


def changed(t, fp):
    s, sample = fingerprint(t)
    return not (s == fp[0] or (s != s and fp[0] != fp[0])) or not torch.equal(sample, fp[1])


def same(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return False
        fa, fb = a.reshape(-1), b.reshape(-1)
        chunk = 1 << 26     # compare in pieces: full-size temporaries would not fit next to several-GB tensors
        return all(torch.allclose(fa[i:i + chunk].float(), fb[i:i + chunk].float(), atol=ATOL, rtol=RTOL)
                   for i in range(0, fa.numel(), chunk))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main(problem, candidate, out_path):
    dev = torch.device("cuda:0")
    res = {"device": torch.cuda.get_device_name(0)}
    t0 = time.time()
    ref_mod = load(problem, "reference")
    try:
        cand_mod = load(candidate, "candidate")
        torch.manual_seed(0)
        init = ref_mod.get_init_inputs()
        ref = ref_mod.Model(*init).to(dev).eval()
        torch.manual_seed(0)
        cand = cand_mod.ModelNew(*init).to(dev).eval()
        res["load_s"] = round(time.time() - t0, 1)     # includes the candidate's kernel compile
    except Exception as e:  # noqa: BLE001
        return {**res, "correct": False, "error": f"load/compile failed: {type(e).__name__}: {e}"[:1500]}

    with torch.no_grad():
        try:
            for s in SEEDS:
                x = inputs(ref_mod, s, dev)
                want = ref(*x)
                before = [fingerprint(t) if isinstance(t, torch.Tensor) else None for t in x]
                got = cand(*x)
                torch.cuda.synchronize(dev)
                if any(isinstance(t, torch.Tensor) and changed(t, b) for t, b in zip(x, before)):
                    return {**res, "correct": False, "error": f"candidate modified its inputs (seed {s})"}
                if isinstance(got, torch.Tensor) and any(isinstance(t, torch.Tensor) and got.data_ptr() == t.data_ptr()
                                                         for t in x):
                    return {**res, "correct": False, "error": f"output aliases an input (seed {s})"}
                ok = same(got, want)
                if not ok:
                    diff = (got.float() - want.float()).abs().max().item() if isinstance(got, torch.Tensor) and \
                        isinstance(want, torch.Tensor) and got.shape == want.shape else None
                    return {**res, "correct": False,
                            "error": f"wrong output on inputs seed {s}" + (f" (max abs diff {diff:.3g})" if diff else
                                                                           " (shape or type mismatch)")}
                del x, want, got, before
                torch.cuda.empty_cache()
            xa = inputs(ref_mod, 11, dev)
            cand(*xa)
            del xa
            torch.cuda.empty_cache()
            xb = inputs(ref_mod, 12, dev)
            got_b = cand(*xb)
            want_b = ref(*xb)
            stale = not same(got_b, want_b)
            del xb, got_b, want_b
            torch.cuda.empty_cache()
            if stale:
                return {**res, "correct": False, "error": "returns a stale result when called on new inputs"}
        except Exception as e:  # noqa: BLE001
            return {**res, "correct": False, "error": f"candidate raised: {type(e).__name__}: {e}"[:1500]}
    res["correct"] = True

    x = inputs(ref_mod, 0, dev)
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)

    def measure(fn, trials):
        ts = []
        with torch.no_grad():
            for _ in range(trials):
                flush.zero_()
                torch.cuda.synchronize(dev)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); fn(*x); b.record()
                torch.cuda.synchronize(dev)
                ts.append(a.elapsed_time(b))
        return S.median(ts)

    with torch.no_grad():
        for _ in range(3):
            ref(*x); cand(*x)
        torch.cuda.synchronize(dev)
        t = time.time(); ref(*x); torch.cuda.synchronize(dev)
        trials = max(10, min(100, int(1000 / max((time.time() - t) * 1000, 0.01))))
    bars = {"eager": ref}
    for mode in ("default", "max-autotune"):
        try:
            torch._dynamo.reset()
            cm = torch.compile(ref, mode=mode)
            with torch.no_grad():
                cm(*x); torch.cuda.synchronize(dev)
            bars[mode] = cm
        except Exception as e:  # noqa: BLE001
            res[f"{mode}_error"] = repr(e)[:200]
    bar_ms = {k: measure(f, trials) for k, f in bars.items()}
    best = min(bar_ms, key=bar_ms.get)
    seq = []
    for who in "ABBAAB":
        seq.append((who, measure(cand if who == "A" else bars[best], trials)))
    cand_ms = S.median([v for w, v in seq if w == "A"])
    bar_best = S.median([v for w, v in seq if w == "B"])
    res.update({"trials": trials, "bar_ms": bar_ms, "bar": best, "bar_best_ms": bar_best, "candidate_ms": cand_ms,
                "speedup_vs_bar": bar_best / cand_ms, "speedup_vs_eager": bar_ms["eager"] / cand_ms,
                "eval_s": round(time.time() - t0, 1)})
    return res


if __name__ == "__main__":
    try:
        out = main(*sys.argv[1:3], None)
    except Exception:  # noqa: BLE001 - report, never crash without a result
        out = {"correct": False, "error": "evaluator error: " + traceback.format_exc()[-1500:]}
    with open(sys.argv[3], "w") as f:
        json.dump(out, f)
    print(json.dumps(out))
