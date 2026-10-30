# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Evaluator for evolving Kissat's restart policy (issue #13).

A candidate is a complete ``src/restart.c``. The evaluator builds Kissat with
it and runs the training instances of ``hamiltonian_split.jsonl``, each with a
conflict budget (``KISSAT_EVAL_BUDGET`` x stock's conflicts there, default 3x)
and a wall-time safety cap. The score is relative to stock Kissat:

    ticks (default):  combined_score = tick-PAR-2(stock) / tick-PAR-2(candidate)
    time:             combined_score = PAR-2(stock) / PAR-2(candidate)

> 1 is better than stock. Tick-PAR-2 counts Kissat's ``search_ticks`` (its
deterministic effort counter), doubling the ticks of an instance not solved
within its budget. For a given binary it is identical on every run and every
machine, so the score is free of timing noise. Wall-clock PAR-2 (``par2_s``,
``time_score``) is reported in both modes; set ``KISSAT_EVAL_METRIC=time`` to
score on it, e.g. for confirmation runs on a dedicated instance type.

Any wrong answer scores 0: a status that contradicts the instance's known
result, or a SAT witness that doesn't satisfy the formula. A build failure,
or forbidden code inside the EVOLVE-BLOCK, also scores 0.

Cascade: ``evaluate_stage1`` builds and runs the 3 fastest training instances
(cheap rejection of broken or much slower candidates); ``evaluate_stage2``
runs the full training set. ``evaluate`` is stage 2.

Each instance set is run as one parallel batch, and its stock baseline was
measured the same way (``prepare.py``), so load from co-scheduled runs is the
same for stock and candidate. Timings from different machines, or a different
``KISSAT_EVAL_JOBS``, aren't comparable; conflict counts are (Kissat's search is
deterministic), and are reported as ``conflicts_ratio``.

Run ``python examples/kissat/prepare.py`` first: it fetches and builds the
pinned Kissat, downloads the instances and measures the stock baseline on this
machine. Environment (all optional):

    KISSAT_EVAL_HOME   cache dir from prepare.py (default: examples/kissat/.cache)
    KISSAT_EVAL_JOBS   instances run in parallel (default: CPU count)
    KISSAT_EVAL_CAP    per-instance time cap, seconds (default: 300)
    KISSAT_EVAL_SPLIT  split to score (default: train)
    KISSAT_EVAL_METRIC ticks (default) or time
    KISSAT_EVAL_BUDGET conflict budget as a multiple of stock's conflicts (default: 3)
"""

from __future__ import annotations

import json
import lzma
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATUS = {10: "sat", 20: "unsat", 0: "unknown"}
STAGE1_INSTANCES = 3

# Inside the EVOLVE-BLOCK a candidate may only compute a restart decision. It
# must not reach the outside world or the solver's output.
FORBIDDEN = re.compile(
    r"^\s*#|\b(asm|__asm__|system|popen|fork|vfork|exec[lvpe]*|fopen|freopen|open|write|"
    r"printf|fprintf|puts|fputs|putchar|exit|_exit|abort|signal|raise|kill|setjmp|longjmp|"
    r"kissat_signal|kissat_print\w*|kissat_section)\b",
    re.M,
)


def home() -> Path:
    return Path(os.environ.get("KISSAT_EVAL_HOME", HERE / ".cache"))


def cap() -> int:
    return int(os.environ.get("KISSAT_EVAL_CAP", "300"))


def metric() -> str:
    m = os.environ.get("KISSAT_EVAL_METRIC", "ticks")
    if m not in ("ticks", "time"):
        raise ValueError(f"KISSAT_EVAL_METRIC must be ticks or time, not {m!r}")
    return m


def budget_factor() -> float:
    return float(os.environ.get("KISSAT_EVAL_BUDGET", "3"))


def conflict_budget(row: dict) -> int:
    """Deterministic per-instance cap: a multiple of stock's conflicts there."""
    return max(1000, int(budget_factor() * row["conflicts"]))


def load_split(name: str) -> list:
    rows = [json.loads(l) for l in open(HERE / "hamiltonian_split.jsonl") if l.strip()]
    return [r for r in rows if r["split"] == name]


def instance_sets() -> dict:
    """Each set is scored, and its baseline measured, as one parallel batch.

    stage1 is the 3 fastest training instances by the runtimes recorded in the
    split file (Graviton calibration), so the same 3 are chosen on any machine.
    """
    train = load_split("train")
    return {
        "stage1": sorted(train, key=lambda r: r["wall_s"])[:STAGE1_INSTANCES],
        "train": train,
        "heldout": load_split("heldout"),
        "frontier": load_split("frontier"),
    }


def evolve_block(code: str) -> str:
    m = re.search(r"EVOLVE-BLOCK-START\s*\*/(.*?)/\*\s*EVOLVE-BLOCK-END", code, re.S)
    if not m:
        raise ValueError("EVOLVE-BLOCK markers missing")
    return m.group(1)


def check_source(code: str) -> str | None:
    """Reason the candidate is rejected before building, or None."""
    try:
        block = evolve_block(code)
    except ValueError as e:
        return str(e)
    block_nocomments = re.sub(r"/\*.*?\*/|//[^\n]*", "", block, flags=re.S)
    m = FORBIDDEN.search(block_nocomments)
    return f"forbidden construct in EVOLVE-BLOCK: {m.group(0).strip()!r}" if m else None


def build(program_path: str, workdir: Path) -> Path:
    """Copy the prepared Kissat tree (timestamps kept), swap in the candidate,
    rebuild incrementally: only restart.o is recompiled and kissat relinked."""
    tree = workdir / "kissat"
    shutil.copytree(home() / "kissat", tree, symlinks=True, copy_function=shutil.copy2)
    shutil.copyfile(program_path, tree / "src" / "restart.c")
    os.utime(tree / "src" / "restart.c")
    p = subprocess.run(["make", "-C", str(tree / "build"), "kissat"], capture_output=True,
                       text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError((p.stdout + p.stderr)[-3000:])
    return tree / "build" / "kissat"


def satisfies(cnf_xz: Path, witness: list) -> bool:
    true = set(witness)
    clause = []
    with lzma.open(cnf_xz, "rt") as f:
        for line in f:
            if not line or line[0] in "cp%":
                continue
            for tok in line.split():
                lit = int(tok)
                if lit == 0:
                    if clause and not any(l in true for l in clause):
                        return False
                    clause = []
                else:
                    clause.append(lit)
    return not clause or any(l in true for l in clause)


def run_instance(kissat: Path, row: dict, time_cap: int) -> dict:
    cnf = home() / "instances" / f"{row['hash']}.cnf.xz"
    t0 = time.monotonic()
    try:
        p = subprocess.run([str(kissat), "--statistics", f"--time={time_cap}",
                            f"--conflicts={conflict_budget(row)}", str(cnf)],
                           capture_output=True, text=True, timeout=time_cap + 60)
        rc, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        rc, out = -9, ""
    wall = time.monotonic() - t0
    status = STATUS.get(rc, "error")
    def counter(name):
        m = re.search(rf"^c {name}:\s+(\d+)", out, re.M)
        return int(m.group(1)) if m else None

    res = {"hash": row["hash"], "status": status, "wall_s": wall,
           "conflicts": counter("conflicts"), "ticks": counter("search_ticks"), "wrong": False}
    if status in ("sat", "unsat") and row["expected"] in ("sat", "unsat") and status != row["expected"]:
        res["wrong"] = True
    elif status == "sat":
        lits = [int(t) for l in out.splitlines() if l.startswith("v ") for t in l[2:].split()]
        res["wrong"] = not satisfies(cnf, [l for l in lits if l])
    elif status == "error":
        res["wrong"] = True  # crashed, e.g. an assertion or segfault in the candidate
    return res


def par2(results: list, time_cap: int) -> float:
    return sum(r["wall_s"] if r["status"] in ("sat", "unsat") else 2 * time_cap
               for r in results) / len(results)


def tick_par2(results: list) -> float:
    """PAR-2 in search ticks: an unsolved instance counts twice the ticks it used
    before hitting its conflict budget. Deterministic for a given binary."""
    return sum((r["ticks"] or 0) * (1 if r["status"] in ("sat", "unsat") else 2)
               for r in results) / len(results)


def baseline() -> dict:
    path = home() / "baseline.json"
    if not path.is_file():
        raise RuntimeError(f"no stock baseline at {path}; run examples/kissat/prepare.py")
    return json.loads(path.read_text())


def run_batch(kissat: Path, rows: list, time_cap: int) -> list:
    jobs = int(os.environ.get("KISSAT_EVAL_JOBS", os.cpu_count() or 1))
    with ThreadPoolExecutor(max(1, min(jobs, len(rows)))) as ex:
        return list(ex.map(lambda r: run_instance(kissat, r, time_cap), rows))


def score(program_path: str, set_name: str, time_cap: int) -> dict:
    code = Path(program_path).read_text()
    reason = check_source(code)
    if reason:
        return {"combined_score": 0.0, "error": reason}
    base = baseline()
    if base["cap"] != time_cap or set_name not in base["sets"]:
        raise RuntimeError(f"baseline.json has no {set_name!r} set at cap {time_cap} s; "
                           "re-run examples/kissat/prepare.py")
    if base.get("budget_factor") != budget_factor():
        raise RuntimeError(f"baseline.json was measured with conflict budget "
                           f"{base.get('budget_factor')}x, not {budget_factor()}x; re-run prepare.py")
    rows = instance_sets()[set_name]
    with tempfile.TemporaryDirectory(prefix="kissat-eval-") as tmp:
        try:
            kissat = build(program_path, Path(tmp))
        except Exception as e:  # noqa: BLE001 - compile errors are the common case
            return {"combined_score": 0.0, "error": f"build failed: {e}"}
        results = run_batch(kissat, rows, time_cap)

    wrong = [r["hash"] for r in results if r["wrong"]]
    stock = [base["sets"][set_name][r["hash"]] for r in rows]
    stock_par2 = par2(stock, time_cap)
    cand_par2 = par2(results, time_cap)
    stock_ticks, cand_ticks = tick_par2(stock), tick_par2(results)
    time_score = stock_par2 / cand_par2
    tick_score = stock_ticks / cand_ticks if cand_ticks else 0.0
    solved = sum(r["status"] in ("sat", "unsat") for r in results)
    conflicts = sum(r["conflicts"] or 0 for r in results)
    stock_conflicts = sum(s["conflicts"] or 0 for s in stock)
    metrics = {
        "combined_score": 0.0 if wrong else (tick_score if metric() == "ticks" else time_score),
        "tick_score": tick_score,
        "time_score": time_score,
        "par2_s": cand_par2,
        "stock_par2_s": stock_par2,
        "solved": solved,
        "instances": len(rows),
        "wrong_answers": len(wrong),
        "conflicts_ratio": conflicts / stock_conflicts if stock_conflicts else None,
    }
    if wrong:
        metrics["error"] = f"wrong answer on {', '.join(h[:8] for h in wrong)}"
    return metrics


def evaluate_stage1(program_path: str) -> dict:
    return score(program_path, "stage1", cap())


def evaluate_stage2(program_path: str) -> dict:
    return score(program_path, os.environ.get("KISSAT_EVAL_SPLIT", "train"), cap())


def evaluate(program_path: str) -> dict:
    return evaluate_stage2(program_path)
