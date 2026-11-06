#!/usr/bin/env python3
# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Prepare the Qiskit routing example on this machine (issue #30):

1. download the Benchpress circuits at the pinned commit (``feynman`` and
   ``qasmbench-{small,medium,large}``, original files, not the ``_transpiled``
   copies) into ``QISKIT_EVAL_HOME``;
2. with ``--split``, write ``split.jsonl`` (committed): the circuits that fit
   the 133-qubit target and are unitary once final measurements are removed
   (no mid-circuit measurement, reset or control flow), alternately
   train/heldout within each set;
3. measure the reference, Qiskit's preset level 2 (``initial_program.py``),
   on every circuit at seeds 0-3, and write ``reference.json``: CZ counts
   (deterministic for a Qiskit version) and CPU seconds (this machine).

    QISKIT_EVAL_PYTHON=/path/to/qiskit-venv/bin/python python examples/qiskit_routing/prepare.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluator  # noqa: E402

BENCHPRESS_COMMIT = "4c04b38a53fc231a89f1a6bc9541c1b641ed6dcf"   # Qiskit/benchpress main, 2026-11-05
URL = f"https://codeload.github.com/Qiskit/benchpress/tar.gz/{BENCHPRESS_COMMIT}"
SETS = ("feynman", "qasmbench-small", "qasmbench-medium", "qasmbench-large")


def fetch(home: Path) -> Path:
    root = home / "qasm"
    if root.is_dir():
        return root
    print(f"fetching {URL}")
    with urllib.request.urlopen(URL, timeout=300) as r:
        data = r.read()
    prefix = f"benchpress-{BENCHPRESS_COMMIT}/benchpress/qasm/"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        for m in t.getmembers():
            if not m.isfile() or not m.name.startswith(prefix):
                continue
            rel = m.name[len(prefix):]
            if rel.split("/")[0] not in SETS or not (rel.endswith(".qasm") or rel.endswith("LICENSE.md")):
                continue
            if rel.endswith("_transpiled.qasm"):
                continue
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(t.extractfile(m).read())
    return root


def write_split(root: Path) -> None:
    probe = ("import json,sys\nfrom qiskit import QuantumCircuit\n"
             "for p in sys.stdin:\n"
             "    p = p.strip(); qc = QuantumCircuit.from_qasm_file(p).remove_final_measurements(inplace=False)\n"
             "    unitary = all(i.operation.name not in ('measure', 'reset', 'if_else', 'while_loop', 'for_loop',\n"
             "                  'switch_case') for i in qc.data)\n"
             "    print(json.dumps([p, qc.num_qubits, unitary]))\n")
    paths = sorted(str(p) for s in SETS for p in (root / s).rglob("*.qasm"))
    out = subprocess.run([evaluator.python(), "-I", "-c", probe], input="\n".join(paths), capture_output=True,
                         text=True, check=True).stdout
    info = {p: (n, u) for p, n, u in map(json.loads, out.splitlines())}
    qubits = {p: n for p, (n, _) in info.items()}
    rows = []
    for s in SETS:
        # unitary circuits only (after removing final measurements), so every output can be checked
        fit = [p for p in paths if p.startswith(str(root / s) + "/") and info[p][0] <= evaluator.TARGET_QUBITS
               and info[p][1]]
        for i, p in enumerate(fit):
            rows.append({"circuit": str(Path(p).relative_to(root)), "set": s, "qubits": qubits[p],
                         "split": "train" if i % 2 == 0 else "heldout"})
    evaluator.SPLIT_FILE.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"split.jsonl: {len(rows)} circuits ({sum(r['split'] == 'train' for r in rows)} train)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", action="store_true", help="(re)write split.jsonl from the circuits")
    a = ap.parse_args()
    home = evaluator.home()
    home.mkdir(parents=True, exist_ok=True)
    root = fetch(home)
    if a.split or not evaluator.SPLIT_FILE.is_file():
        write_split(root)
    rows = evaluator.load_split()
    jobs = [{"circuit": r["circuit"], "seed": s} for r in rows for s in range(4)]
    print(f"reference: preset level 2 on {len(rows)} circuits x 4 seeds")
    results = evaluator.run_jobs(str(HERE / "initial_program.py"), jobs, check=False, time_limit=None)
    bad = [r for r in results if "cz" not in r or not r.get("valid")]
    if bad:
        print(f"reference failed on {len(bad)} runs, e.g. {bad[0]}", file=sys.stderr)
        return 1
    ver = subprocess.run([evaluator.python(), "-I", "-c", "import qiskit; print(qiskit.__version__)"],
                         capture_output=True, text=True, check=True).stdout.strip()
    ref = {"qiskit": ver, "benchpress": BENCHPRESS_COMMIT, "jobs": evaluator.jobs(),
           "runs": {f"{r['circuit']}@{r['seed']}": {"cz": r["cz"], "cpu_s": r["cpu_s"]} for r in results}}
    evaluator.reference_file().write_text(json.dumps(ref, indent=1))
    print(f"reference.json written (Qiskit {ver}, {sum(r['cpu_s'] for r in results):.0f} CPU-s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
