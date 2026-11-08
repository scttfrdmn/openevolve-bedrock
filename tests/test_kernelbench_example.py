# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""examples/kernelbench: static rejection, remote-result handling and config (no GPU needed)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "kernelbench"


def _load():
    spec = importlib.util.spec_from_file_location("kb_evaluator", EXAMPLE / "evaluator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load()
HAS_CHECKER = (ev.home() / "kernel_static_checker.py").is_file()
KERNEL = '''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
src = r"""
__global__ void k(const float* x, float* y, long n){long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]>0.f?x[i]:0.f;}
torch::Tensor relu(torch::Tensor x){auto y=torch::empty_like(x); long n=x.numel(); k<<<(n+255)/256,256>>>(x.data_ptr<float>(),y.data_ptr<float>(),n); return y;}
"""
ext = load_inline("r", cpp_sources="torch::Tensor relu(torch::Tensor x);", cuda_sources=src, functions=["relu"])
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return ext.relu(x)
'''


@pytest.mark.skipif(not HAS_CHECKER, reason="run examples/kernelbench/prepare.py")
def test_a_cuda_kernel_passes_the_static_check():
    assert ev.check_source(KERNEL) is None


@pytest.mark.skipif(not HAS_CHECKER, reason="run examples/kernelbench/prepare.py")
@pytest.mark.parametrize("change", [
    ("return ext.relu(x)", "return torch.relu(x)"),                      # PyTorch does the work
    ("return ext.relu(x)", "return torch.compile(lambda t: t.clamp(min=0))(x)"),
    ("return ext.relu(x)", "try:\n            return ext.relu(x)\n        except Exception:\n            return x"),
    ("super().__init__()", "super().__init__()\n        torch.backends.cuda.matmul.allow_tf32 = True"),
    ("return ext.relu(x)", "return ext.relu(x.half()).float()"),
    ("class ModelNew", "class Model"),
])
def test_hacks_and_non_kernels_are_rejected(change):
    assert ev.check_source(KERNEL.replace(*change)) is not None


def test_comments_mentioning_torch_compile_are_fine():
    assert ev.EXTRA_FORBIDDEN.search("x = 1\n") is None
    if HAS_CHECKER:
        assert ev.check_source("# scored against torch.compile\n" + KERNEL) is None


@pytest.mark.parametrize("remote, want", [
    ({"correct": True, "speedup_vs_bar": 1.25, "speedup_vs_eager": 1.6, "candidate_ms": 2.0, "bar": "default"},
     {"combined_score": 1.25, "bar": 1}),
    ({"correct": False, "error": "wrong output on inputs seed 1"}, {"combined_score": 0.0}),
])
def test_remote_result_becomes_metrics(monkeypatch, tmp_path, remote, want):
    monkeypatch.setattr(ev, "check_source", lambda code: None)
    monkeypatch.setattr(ev, "run_remote", lambda code: remote)
    prog = tmp_path / "p.py"
    prog.write_text(KERNEL)
    m = ev.evaluate(str(prog))
    for k, v in want.items():
        assert m[k] == v
    assert ("error" in m) == (not remote["correct"])


def test_no_result_from_the_host_scores_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_PROBLEM", "level1/19_ReLU")
    monkeypatch.setenv("KB_HOST", "gpu.example")
    monkeypatch.setattr(ev, "ssh", lambda cmd, timeout, stdin=None: SimpleNamespace(stdout="", stderr="no route"))
    assert ev.run_remote(KERNEL) == {"correct": False, "error": "GPU run gave no result: no route"}
    monkeypatch.setattr(ev, "ssh", lambda cmd, timeout, stdin=None: SimpleNamespace(
        stdout="noise\n" + json.dumps({"correct": True, "speedup_vs_bar": 1.1}) + "\n", stderr=""))
    assert ev.run_remote(KERNEL)["speedup_vs_bar"] == 1.1


def test_config_loads():
    from openevolve.config import load_config
    cfg = load_config(str(EXAMPLE / "config.yaml"))
    assert cfg.evaluator.parallel_evaluations == 1 and cfg.evaluator.cascade_evaluation
    assert "Blackwell" in cfg.prompt.system_message
