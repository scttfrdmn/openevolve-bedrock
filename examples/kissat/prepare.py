#!/usr/bin/env python3
# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Prepare the Kissat example on this machine (issue #13):

1. fetch Kissat at the pinned commit from GitHub and build it;
2. download the split's instances from the Global Benchmark Database;
3. measure stock Kissat on them, with the same runner and cap the evaluator
   uses, and write ``baseline.json`` (``baseline-<stem>.json`` for a split file
   other than the default, ``KISSAT_EVAL_SPLIT_FILE``).

The baseline belongs to the machine it was measured on: candidates are scored
against it, so re-run this on every machine type that evaluates.

    python examples/kissat/prepare.py [--sets stage1,train,heldout,frontier] [--cap 300]
                                      [--stock-ticks FILE]

``--stock-ticks FILE`` also writes the machine-independent part (status,
conflicts, ticks) in the format of the committed ``stock_ticks.json``.
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
        try:
            t.extractall(home, filter="data")
        except TypeError:  # Python without extraction filters (e.g. AL2023's 3.9)
            t.extractall(home)
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

    mirror = os.environ.get("KISSAT_EVAL_INSTANCES_S3", "").rstrip("/")

    def one(r):
        dest = inst / f"{r['hash']}.cnf.xz"
        if mirror:  # e.g. s3://oeb-kissat-<account>-us-west-2/main_2024, staged in #13
            subprocess.run(["aws", "s3", "cp", "--quiet", f"{mirror}/{r['hash']}.cnf.xz", str(dest)],
                           check=True)
        else:
            fetch(GBD_URL.format(r["hash"]), dest)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, todo))
    print(f"instances: {len(rows)} ({len(todo)} downloaded) in {inst}")


def measure(sets: dict, time_cap: int, jobs: int, log=print) -> dict:
    """Stock Kissat on each set, each set run as one batch as the evaluator runs it.
    Returns the baseline dict; raises if stock answers wrongly."""
    if evaluator.extra_options():
        raise RuntimeError("KISSAT_EVAL_OPTIONS is set; the stock baseline must run with default options")
    home = evaluator.home()
    rows = list({r["hash"]: r for rows_ in sets.values() for r in rows_}.values())
    build_kissat(home)
    fetch_instances(home, rows)
    kissat = home / "kissat" / "build" / "kissat"
    os.environ["KISSAT_EVAL_JOBS"] = str(jobs)
    log(f"stock baseline: {len(sets)} sets, cap {time_cap} s, up to {jobs} in parallel;"
        " each set runs as its own batch, as the evaluator runs it")
    t0 = time.time()
    measured = {}
    for name, set_rows in sets.items():
        results = evaluator.run_batch(kissat, set_rows, time_cap)
        wrong = [r["hash"] for r in results if r["wrong"]]
        if wrong:
            raise RuntimeError(f"stock Kissat answered wrongly on {wrong}")
        measured[name] = {}
        for (row, seed), r in zip(evaluator.runs(set_rows), results):
            m = measured[name][evaluator.run_key(r["hash"], seed)] = {k: r[k] for k in ("status", "wall_s", "conflicts", "ticks")}
            if r["status"] == "unknown":  # ran out of budget (deterministic) or hit the wall cap
                m["budget_reached"] = (r["conflicts"] or 0) >= evaluator.conflict_budget(row)
        log(f"  {name:9} solved {sum(r['status'] in ('sat', 'unsat') for r in results)}"
            f"/{len(results)}  PAR-2 {evaluator.par2(results, time_cap):.1f} s"
            f"  tick-PAR-2 {evaluator.tick_par2(results, results) or float('nan'):.3g}"
            f"  ({time.time() - t0:.0f} s elapsed)")
    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "jobs": jobs,
        "cap": time_cap,
        "budget_factor": evaluator.budget_factor(),
        "kissat_commit": KISSAT_COMMIT,
        "split_sha256": hashlib.sha256(evaluator.split_file().read_bytes()).hexdigest(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sets": measured,
    }


def stock_ticks(base: dict) -> dict:
    """The machine-independent part of a baseline, as committed in stock_ticks.json."""
    return {
        "cap": base["cap"], "budget_factor": base["budget_factor"],
        "kissat_commit": base["kissat_commit"], "split_sha256": base["split_sha256"],
        "measured_on": f"{base['machine']} ({base['host']}, {base['measured_at']}); "
                       "ticks and conflicts are machine-independent",
        "sets": {name: {h: {k: r[k] for k in ("status", "conflicts", "ticks", "budget_reached") if k in r}
                        for h, r in rows.items()}
                 for name, rows in base["sets"].items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", default=",".join(evaluator.instance_sets()))
    ap.add_argument("--cap", type=int, default=evaluator.cap())
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("KISSAT_EVAL_JOBS",
                                                                   os.cpu_count() or 1)))
    ap.add_argument("--stock-ticks", type=Path, help="also write the committed stock-ticks file here")
    a = ap.parse_args()

    home = evaluator.home()
    home.mkdir(parents=True, exist_ok=True)
    sets = {name: evaluator.instance_sets()[name] for name in a.sets.split(",")}
    t0 = time.time()
    try:
        base = measure(sets, a.cap, a.jobs)
    except RuntimeError as e:
        print(f"{e}; not writing a baseline", file=sys.stderr)
        return 1
    evaluator.baseline_file().write_text(json.dumps(base, indent=1))
    print(f"{evaluator.baseline_file().name} written ({time.time() - t0:.0f} s)")
    if a.stock_ticks:
        a.stock_ticks.write_text(json.dumps(stock_ticks(base), indent=1) + "\n")
        print(f"{a.stock_ticks} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
