#!/usr/bin/env python3
# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Prepare the Kissat example on this machine (issue #13):

1. fetch Kissat at the pinned commit from GitHub and build it;
2. download the split's instances from the Global Benchmark Database;
3. measure stock Kissat on them, with the same runner and cap the evaluator
   uses, and write ``baseline.json``.

The baseline belongs to the machine it was measured on: candidates are scored
against it, so re-run this on every machine type that evaluates.

    python examples/kissat/prepare.py [--sets stage1,train,heldout,frontier] [--cap 300]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluator  # noqa: E402

KISSAT_COMMIT = "8af8e56f174b778aef3aa45af9f739b2a5f492c2"  # Kissat 4.0.4
KISSAT_URL = f"https://github.com/arminbiere/kissat/archive/{KISSAT_COMMIT}.tar.gz"
GBD_URL = "https://benchmark-database.de/file/{}?context=cnf"


def fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def build_kissat(home: Path) -> None:
    tree = home / "kissat"
    if (tree / "build" / "kissat").is_file():
        print(f"kissat already built in {tree}")
        return
    tarball = home / f"kissat-{KISSAT_COMMIT[:7]}.tar.gz"
    if not tarball.is_file():
        print(f"fetching {KISSAT_URL}")
        fetch(KISSAT_URL, tarball)
    with tarfile.open(tarball) as t:
        t.extractall(home, filter="data")
    (home / f"kissat-{KISSAT_COMMIT}").rename(tree)
    subprocess.run(["./configure"], cwd=tree, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["make", f"-j{os.cpu_count() or 1}"], cwd=tree, check=True,
                   stdout=subprocess.DEVNULL)
    print(subprocess.run([str(tree / "build" / "kissat"), "--version"], capture_output=True,
                         text=True).stdout.strip(), "built in", tree)


def fetch_instances(home: Path, rows: list) -> None:
    inst = home / "instances"
    inst.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (inst / f"{r['hash']}.cnf.xz").is_file()]

    def one(r):
        dest = inst / f"{r['hash']}.cnf.xz"
        fetch(GBD_URL.format(r["hash"]), dest)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, todo))
    print(f"instances: {len(rows)} ({len(todo)} downloaded) in {inst}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", default="stage1,train,heldout,frontier")
    ap.add_argument("--cap", type=int, default=evaluator.cap())
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("KISSAT_EVAL_JOBS",
                                                                   os.cpu_count() or 1)))
    a = ap.parse_args()

    home = evaluator.home()
    home.mkdir(parents=True, exist_ok=True)
    sets = {name: evaluator.instance_sets()[name] for name in a.sets.split(",")}
    rows = list({r["hash"]: r for rows_ in sets.values() for r in rows_}.values())
    build_kissat(home)
    fetch_instances(home, rows)

    kissat = home / "kissat" / "build" / "kissat"
    os.environ["KISSAT_EVAL_JOBS"] = str(a.jobs)
    print(f"stock baseline: {len(sets)} sets, cap {a.cap} s, up to {a.jobs} in parallel;"
          " each set runs as its own batch, as the evaluator runs it")
    t0 = time.time()
    measured = {}
    for name, set_rows in sets.items():
        results = evaluator.run_batch(kissat, set_rows, a.cap)
        wrong = [r["hash"] for r in results if r["wrong"]]
        if wrong:
            print(f"stock Kissat answered wrongly on {wrong}; not writing a baseline", file=sys.stderr)
            return 1
        measured[name] = {r["hash"]: {k: r[k] for k in ("status", "wall_s", "conflicts", "ticks")}
                          for r in results}
        print(f"  {name:9} solved {sum(r['status'] in ('sat', 'unsat') for r in results)}"
              f"/{len(results)}  PAR-2 {evaluator.par2(results, a.cap):.1f} s"
              f"  tick-PAR-2 {evaluator.tick_par2(results):.3g}"
              f"  ({time.time() - t0:.0f} s elapsed)")
    base = {
        "host": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "jobs": a.jobs,
        "cap": a.cap,
        "budget_factor": evaluator.budget_factor(),
        "kissat_commit": KISSAT_COMMIT,
        "split_sha256": hashlib.sha256((HERE / "hamiltonian_split.jsonl").read_bytes()).hexdigest(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sets": measured,
    }
    (home / "baseline.json").write_text(json.dumps(base, indent=1))
    print(f"baseline.json written ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
