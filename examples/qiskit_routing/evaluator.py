# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Evaluator for evolving a Qiskit transpiler pipeline (issue #30).

A candidate is a Python file whose ``build_pass_manager(backend, seed)``
returns a pass manager. Each training circuit from Benchpress
(``split.jsonl``) is transpiled with it for IBM Torino (``FakeTorino``, 133
qubits, CZ) at each seed. The score is relative to Qiskit's preset level 2:

    combined_score = geomean over (circuit, seed) of (CZ_ref + 1) / (CZ_cand + 1)

> 1 means fewer CZ gates than level 2. CZ counts are deterministic for a given
pipeline, seed and Qiskit version, so a score is the same on every machine.

A candidate scores 0 if:
- an output uses a gate outside the target, or a 2-qubit gate off a coupling
  edge;
- an output isn't equivalent to its input (checked by simulation on random
  inputs, for every output with at most 20 active qubits);
- it uses more than ``QISKIT_EVAL_CPU_CAP`` (default 2) times level 2's CPU
  time on the same runs (stage 2), or a worker hangs past its time limit;
- it fails the source check (imports, classes, I/O, introspection or
  approximation inside the EVOLVE-BLOCK).

Cascade: ``evaluate_stage1`` runs the 10 cheapest training circuits at seed 0
with the equivalence check; ``evaluate_stage2`` runs the whole training set.

Run ``prepare.py`` first: it downloads the circuits and measures the
reference on this machine (CPU time is per machine). Environment:

    QISKIT_EVAL_PYTHON  interpreter with requirements.txt installed (required)
    QISKIT_EVAL_HOME    cache dir (default: examples/qiskit_routing/.cache)
    QISKIT_EVAL_JOBS    worker processes (default: CPU count - 2)
    QISKIT_EVAL_SEEDS   seeds (default: 0,1)
    QISKIT_EVAL_SPLIT   set scored by stage 2 (default: train)
    QISKIT_EVAL_CPU_CAP CPU limit as a multiple of level 2's (default: 2)
    QISKIT_EVAL_DETAIL  set to add per-run results ("detail") to the metrics
"""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLIT_FILE = HERE / "split.jsonl"
TARGET_QUBITS = 133
STAGE1_CIRCUITS = 10

FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "input", "breakpoint", "os", "sys", "subprocess", "socket",
                   "importlib", "builtins", "approximation_degree"}


def home() -> Path:
    return Path(os.environ.get("QISKIT_EVAL_HOME", HERE / ".cache"))


def python() -> str:
    p = os.environ.get("QISKIT_EVAL_PYTHON")
    if not p:
        raise RuntimeError("set QISKIT_EVAL_PYTHON to an interpreter with examples/qiskit_routing/requirements.txt")
    return p


def jobs() -> int:
    return int(os.environ.get("QISKIT_EVAL_JOBS", max(1, (os.cpu_count() or 2) - 2)))


def seeds() -> list:
    return [int(x) for x in os.environ.get("QISKIT_EVAL_SEEDS", "0,1").split(",") if x.strip()]


def cpu_cap() -> float:
    return float(os.environ.get("QISKIT_EVAL_CPU_CAP", "2"))


def reference_file() -> Path:
    return home() / "reference.json"


def load_split(name: str | None = None) -> list:
    rows = [json.loads(l) for l in SPLIT_FILE.read_text().splitlines() if l.strip()]
    return [r for r in rows if name is None or r["split"] == name]


def check_source(code: str) -> str | None:
    """Reason the EVOLVE-BLOCK is rejected, or None."""
    start, end = code.find("# EVOLVE-BLOCK-START"), code.find("# EVOLVE-BLOCK-END")
    if start < 0 or end < start:
        return "EVOLVE-BLOCK markers missing"
    try:
        tree = ast.parse(code[start:end])
    except SyntaxError as e:
        return f"syntax error in EVOLVE-BLOCK: {e}"
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_pass_manager"]
    if not funcs:
        return "EVOLVE-BLOCK must define build_pass_manager(backend, seed)"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "imports are not allowed in the EVOLVE-BLOCK"
        if isinstance(node, ast.ClassDef):
            return "classes are not allowed in the EVOLVE-BLOCK (compose Qiskit's own passes)"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return "global/nonlocal are not allowed in the EVOLVE-BLOCK"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"{node.id!r} is not allowed in the EVOLVE-BLOCK"
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN_NAMES):
            return f"attribute {node.attr!r} is not allowed in the EVOLVE-BLOCK"
        if isinstance(node, ast.keyword) and node.arg in FORBIDDEN_NAMES:
            return f"keyword {node.arg!r} is not allowed in the EVOLVE-BLOCK"
    return None


def run_jobs(program: str, job_list: list, check: bool, time_limit: dict | None) -> list:
    """Run jobs ({"circuit", "seed"}) in worker processes; results in input order.

    Jobs are spread over workers by reference CPU (largest first). A worker
    past its time limit (``time_limit[key]`` summed, x cap, + 60 s) is killed
    and its unfinished jobs come back with ``timeout``."""
    qasm = home() / "qasm"
    n = min(jobs(), len(job_list))
    cost = lambda j: (time_limit or {}).get(f"{j['circuit']}@{j['seed']}", 1.0)  # noqa: E731
    order = sorted(range(len(job_list)), key=lambda i: -cost(job_list[i]))
    chunks, loads = [[] for _ in range(n)], [0.0] * n
    for i in order:
        k = loads.index(min(loads))
        chunks[k].append(i)
        loads[k] += cost(job_list[i])

    def work(k):
        lines = "".join(json.dumps({"path": str(qasm / job_list[i]["circuit"]), "seed": job_list[i]["seed"],
                                    "check": check}) + "\n" for i in chunks[k])
        # only catches hangs: equivalence checks add wall time that isn't counted as CPU
        limit = None if time_limit is None else 600 + 3 * cpu_cap() * loads[k]
        try:
            p = subprocess.run([python(), "-I", str(HERE / "worker.py"), program], input=lines,
                               capture_output=True, text=True, timeout=limit)
            out = p.stdout
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        got = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
        res = {}
        for i, r in zip(chunks[k], got):
            res[i] = r
        for i in chunks[k][len(got):]:
            res[i] = {"timeout": True}
        return res

    merged = {}
    with ThreadPoolExecutor(n) as ex:
        for part in ex.map(work, range(n)):
            merged.update(part)
    return [{**job_list[i], **merged[i]} for i in range(len(job_list))]


def score(program_path: str, rows: list, run_seeds: list, check: bool, enforce_cpu: bool = True) -> dict:
    reason = check_source(Path(program_path).read_text())
    if reason:
        return {"combined_score": 0.0, "error": reason}
    ref = json.loads(reference_file().read_text())["runs"]
    job_list = [{"circuit": r["circuit"], "seed": s} for r in rows for s in run_seeds]
    limits = {f"{j['circuit']}@{j['seed']}": ref[f"{j['circuit']}@{j['seed']}"]["cpu_s"] for j in job_list}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="qiskit-eval-") as tmp:   # candidate file as the worker sees it
        prog = Path(tmp) / "candidate.py"
        prog.write_text(Path(program_path).read_text())
        results = run_jobs(str(prog), job_list, check=check, time_limit=limits)
    timeouts = sum(1 for r in results if r.get("timeout"))
    errors = [r for r in results if "error" in r]
    invalid = sum(1 for r in results if r.get("valid") is False)
    inequivalent = sum(1 for r in results if r.get("equivalent") is False)
    checked = sum(1 for r in results if r.get("equivalent") is not None)
    done = [r for r in results if "cz" in r]
    cpu = sum(r["cpu_s"] for r in done)
    ref_cpu = sum(limits[f"{r['circuit']}@{r['seed']}"] for r in done) or 1.0
    metrics = {"runs": len(results), "circuits": len(rows), "invalid": invalid, "inequivalent": inequivalent,
               "checked": checked, "timeouts": timeouts, "errors": len(errors),
               "cpu_ratio": cpu / ref_cpu, "eval_s": round(time.time() - t0, 1)}
    if os.environ.get("QISKIT_EVAL_DETAIL"):
        metrics["detail"] = results
    fail = None
    if errors:
        fail = f"pipeline raised on {len(errors)} runs, e.g. {errors[0]['error']}"
    elif timeouts:
        fail = f"{timeouts} runs past the time limit"
    elif invalid:
        fail = f"{invalid} outputs use gates or qubit pairs the target doesn't have"
    elif inequivalent:
        fail = f"{inequivalent} outputs aren't equivalent to their input"
    elif enforce_cpu and metrics["cpu_ratio"] > cpu_cap():
        fail = f"CPU {metrics['cpu_ratio']:.2f}x level 2, over the {cpu_cap():g}x limit"
    if fail:
        return {"combined_score": 0.0, "error": fail, **metrics}
    logs = [math.log((ref[f"{r['circuit']}@{r['seed']}"]["cz"] + 1) / (r["cz"] + 1)) for r in done]
    return {"combined_score": math.exp(sum(logs) / len(logs)), **metrics}


def evaluate_stage1(program_path: str) -> dict:
    ref = json.loads(reference_file().read_text())["runs"]
    train = load_split("train")
    cheap = sorted(train, key=lambda r: (r["qubits"] > 20, ref[f"{r['circuit']}@0"]["cpu_s"]))[:STAGE1_CIRCUITS]
    # a correctness gate: these runs are too short for the CPU ratio to mean much
    return score(program_path, cheap, [0], check=True, enforce_cpu=False)


def evaluate_stage2(program_path: str) -> dict:
    return score(program_path, load_split(os.environ.get("QISKIT_EVAL_SPLIT", "train")), seeds(), check=True)


def evaluate(program_path: str) -> dict:
    return evaluate_stage2(program_path)
