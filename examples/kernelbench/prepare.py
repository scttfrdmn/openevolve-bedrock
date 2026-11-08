#!/usr/bin/env python3
# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Prepare examples/kernelbench (issue #29):

1. download KernelBench's level 1 and 2 problems and its static checker at
   the pinned commit into ``KB_HOME`` (MIT, Stanford Scaling Intelligence Lab;
   not committed here);
2. write the initial program for ``KB_PROBLEM``: the reference with ``Model``
   renamed ``ModelNew``, inside an EVOLVE-BLOCK (it scores 0 until a kernel
   replaces the PyTorch ops);
3. with ``KB_HOST`` set, copy ``eval_kb.py`` and the problems to the GPU host
   and pull the container there.

    KB_PROBLEM=level2/12_Gemm_Multiply_LeakyReLU KB_HOST=ceres.local python examples/kernelbench/prepare.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluator  # noqa: E402

URL = f"https://codeload.github.com/ScalingIntelligence/KernelBench/tar.gz/{evaluator.KERNELBENCH_COMMIT}"

HEADER = '''# KernelBench {level} problem {name} (ScalingIntelligence/KernelBench @ {commit}, MIT License,
# Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini), as the starting point for
# openevolve-bedrock's examples/kernelbench (issue #29).
#
# Replace the PyTorch operations in ModelNew with CUDA kernels (torch.utils.cpp_extension.load_inline)
# or Triton kernels. ModelNew must keep Model's constructor and forward signatures and return the
# same result. It is scored against the fastest of PyTorch eager, torch.compile and
# torch.compile(mode="max-autotune").
'''


def fetch(home: Path) -> None:
    if (home / "kernel_static_checker.py").is_file() and (home / "problems").is_dir():
        return
    print(f"fetching {URL}")
    with urllib.request.urlopen(URL, timeout=300) as r:
        data = r.read()
    root = f"KernelBench-{evaluator.KERNELBENCH_COMMIT}/"
    (home / "problems").mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            rel = m.name[len(root):]
            if rel == "src/kernelbench/kernel_static_checker.py":
                (home / "kernel_static_checker.py").write_bytes(t.extractfile(m).read())
            elif rel.startswith(("KernelBench/level1/", "KernelBench/level2/")) and rel.endswith(".py"):
                level, name = rel.split("/")[1:3]
                (home / "problems" / f"{level}_{name}").write_bytes(t.extractfile(m).read())
            elif rel == "LICENSE":
                (home / "LICENSE.kernelbench").write_bytes(t.extractfile(m).read())


def initial_program(home: Path, prob: str) -> Path:
    level, name = prob.split("/")
    src = (home / "problems" / f"{level}_{name}.py").read_text()
    body = src.replace("class Model(", "class ModelNew(").replace("super(Model, self)", "super(ModelNew, self)")
    out = home / "programs" / f"{level}_{name}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEADER.format(level=level, name=name, commit=evaluator.KERNELBENCH_COMMIT[:7])
                   + "# EVOLVE-BLOCK-START\n" + body.rstrip() + "\n# EVOLVE-BLOCK-END\n")
    return out


def setup_host(home: Path) -> None:
    d = evaluator.remote_dir()
    p = evaluator.ssh(f"mkdir -p ~/{d}/problems ~/{d}/cand ~/{d}/out", timeout=60)
    if p.returncode:
        raise RuntimeError(p.stderr)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(HERE / "eval_kb.py", arcname="eval_kb.py")
        t.add(home / "problems", arcname="problems")
    subprocess.run(["ssh", "-o", "BatchMode=yes", evaluator.host(), f"tar -xzf - -C ~/{d}"], input=buf.getvalue(),
                   check=True)
    img = os.environ.get("KB_IMAGE", evaluator.IMAGE)
    p = evaluator.ssh(f"docker image inspect {img} >/dev/null 2>&1 || docker pull -q {img}", timeout=3600)
    print(f"host {evaluator.host()}:~/{d} ready ({img})" if p.returncode == 0 else f"pull failed: {p.stderr[-400:]}")


def main() -> int:
    home = evaluator.home()
    home.mkdir(parents=True, exist_ok=True)
    fetch(home)
    print(f"{len(list((home / 'problems').glob('*.py')))} problems and the static checker in {home}")
    if os.environ.get("KB_PROBLEM"):
        print(f"initial program: {initial_program(home, evaluator.problem())}")
    if os.environ.get("KB_HOST"):
        setup_host(home)
    return 0


if __name__ == "__main__":
    sys.exit(main())
