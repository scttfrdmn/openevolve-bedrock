# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
#58 / #59: a declared parameter space over Qiskit's level-2 pipeline, searched without a model.

    export QISKIT_EVAL_PYTHON=~/qiskit-venv/bin/python      # as for examples/qiskit_routing
    python experiments/two_level/qiskit_space.py enumerate space.jsonl      # every config, train, seeds 0,1
    python experiments/two_level/qiskit_space.py rediscover space.jsonl 1.0093 1.0143 1.0176
    python experiments/two_level/qiskit_space.py heldout picks.json heldout.jsonl   # train + heldout, seeds 0-3

``rediscover`` reports, over shuffled orders, how many random evaluations reach each
score threshold. That is the cost of random search over this space, in evaluations.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import statistics as S
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "qiskit_routing"
SPACE = {"max_iterations": [1, 2, 3, 4, 6], "layout_trials": [10, 20, 40, 80], "swap_trials": [10, 20, 40, 80],
         "vf2post": [False, True], "opt_level3": [False, True]}


def program(c: dict) -> str:
    src = (EXAMPLE / "initial_program.py").read_text()
    a = src.index("# EVOLVE-BLOCK-START\n") + len("# EVOLVE-BLOCK-START\n")
    b = src.index("# EVOLVE-BLOCK-END")
    body = f'''def build_pass_manager(backend, seed):
    pm = generate_preset_pass_manager(2, backend, seed_transpiler=seed)
    layout = PassManager([passes.SabreLayout(backend.target, max_iterations={c["max_iterations"]}, seed=seed,
                                             swap_trials={c["swap_trials"]}, layout_trials={c["layout_trials"]})])
    layout.append(ConditionalController(common.generate_embed_passmanager(backend.target).to_flow_controller(),
                                        condition=lambda ps: ps["final_layout"] is None))
    pm.layout = layout
    if {c["vf2post"]}:
        pm.post_layout = PassManager([passes.VF2PostLayout(target=backend.target, seed=seed)])
    if {c["opt_level3"]}:
        pm.optimization = generate_preset_pass_manager(3, backend, seed_transpiler=seed).optimization
    return pm
'''
    return src[:a] + body + src[b:]


def _evaluator():
    sys.path.insert(0, str(EXAMPLE))
    import evaluator
    return evaluator


def _write(c: dict, d: Path) -> str:
    f = d / f"cfg_{abs(hash(json.dumps(c, sort_keys=True)))}.py"
    f.write_text(program(c))
    return str(f)


def enumerate_space(out: Path) -> None:
    ev = _evaluator()
    done = {json.loads(l)["key"] for l in out.open()} if out.exists() else set()
    rows, t0 = ev.load_split("train"), time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for vals in itertools.product(*SPACE.values()):
            c = dict(zip(SPACE, vals))
            key = json.dumps(c, sort_keys=True)
            if key in done:
                continue
            ts = time.time()
            m = ev.score(_write(c, Path(tmp)), rows, [0, 1], check=True)
            rec = {"key": key, **c, "score": m["combined_score"], "cpu_ratio": m.get("cpu_ratio"),
                   "error": m.get("error"), "eval_s": round(time.time() - ts, 1)}
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            done.add(key)
            print(len(done), c, round(rec["score"], 4), f"{time.time() - t0:.0f}s", flush=True)


def rediscover(space: Path, thresholds, orders: int = 5000) -> None:
    rows = [json.loads(l) for l in space.open()]
    rnd = random.Random(1)
    for thr in thresholds:
        hits = sum(1 for r in rows if not r.get("error") and r["score"] >= thr)
        need = []
        for _ in range(orders):
            order = rows[:]
            rnd.shuffle(order)
            need.append(next((i + 1 for i, r in enumerate(order) if not r.get("error") and r["score"] >= thr),
                             len(rows)))
        need.sort()
        print(f"threshold {thr}: {hits} of {len(rows)} configs reach it; evaluations to first hit: "
              f"median {S.median(need)}, p90 {need[int(0.9 * len(need))]}")


def heldout(picks: Path, out: Path) -> None:
    import os
    ev = _evaluator()
    ref = json.loads(ev.reference_file().read_text())["runs"]
    os.environ["QISKIT_EVAL_DETAIL"] = "1"
    with tempfile.TemporaryDirectory() as tmp, out.open("a") as f:
        for c in json.loads(picks.read_text()):
            res = {"config": c}
            for half in ("train", "heldout"):
                m = ev.score(_write(c, Path(tmp)), ev.load_split(half), [0, 1, 2, 3], check=True)
                per = {}
                for r in m.pop("detail", []):
                    if "cz" in r:
                        per.setdefault(r["circuit"], []).append(
                            math.log((ref[f"{r['circuit']}@{r['seed']}"]["cz"] + 1) / (r["cz"] + 1)))
                mm = [S.mean(v) for v in per.values()]
                rnd = random.Random(1)
                bs = sorted(math.exp(S.mean([rnd.choice(mm) for _ in mm])) for _ in range(4000)) if mm else [0.0]
                res[half] = {"score": m["combined_score"], "ci": [bs[len(bs) // 40], bs[-len(bs) // 40]],
                             "cpu_ratio": m.get("cpu_ratio"), "error": m.get("error")}
            f.write(json.dumps(res) + "\n")
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "enumerate":
        enumerate_space(Path(sys.argv[2]))
    elif cmd == "rediscover":
        rediscover(Path(sys.argv[2]), [float(x) for x in sys.argv[3:]])
    elif cmd == "heldout":
        heldout(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
