# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
#68 part 8 on Qiskit: three structure families, each with its own declared parameter space,
each tuned before the families are compared.

    export QISKIT_EVAL_PYTHON=~/qiskit-venv/bin/python
    python experiments/two_level/qiskit_structures.py measure table.jsonl [sabre_table.jsonl]
    python experiments/two_level/qiskit_structures.py analyse table.jsonl

Structures (all schedules over the same transpilation task; the evaluator's equivalence check
and CPU cap apply to every point):

- ``sabre_l2``: level 2 with Sabre layout knobs, VF2PostLayout, optimisation stage (the #58
  space; its #58 table can be reused instead of re-measured);
- ``vf2_first``: a VF2 perfect-layout search first, Sabre only if none is found;
- ``resynth``: level 2 plus resynthesis rounds before layout and after routing.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics as S
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "qiskit_routing"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qiskit_space  # noqa: E402

from openevolve_bedrock import tuner  # noqa: E402

EMBED = ('    layout.append(ConditionalController(common.generate_embed_passmanager(backend.target)'
         '.to_flow_controller(),\n                                        condition=lambda ps: ps["final_layout"] is None))\n')
RESYNTH = ('passes.ConsolidateBlocks(target=backend.target, force_consolidate=True), '
           'passes.UnitarySynthesis(target=backend.target), '
           'passes.Optimize1qGatesDecomposition(target=backend.target), '
           'passes.CommutativeCancellation(target=backend.target)')
RESYNTH_LOGICAL = ('passes.ConsolidateBlocks(basis_gates=["cz", "rz", "sx", "x"], force_consolidate=True), '
                   'passes.UnitarySynthesis(basis_gates=["cz", "rz", "sx", "x"]), '
                   'passes.Optimize1qGatesDecomposition(basis=["cz", "rz", "sx", "x"]), '
                   'passes.CommutativeCancellation(basis_gates=["cz", "rz", "sx", "x"])')


def _wrap(body: str) -> str:
    src = (EXAMPLE / "initial_program.py").read_text()
    a = src.index("# EVOLVE-BLOCK-START\n") + len("# EVOLVE-BLOCK-START\n")
    b = src.index("# EVOLVE-BLOCK-END")
    return src[:a] + body + src[b:]


def vf2_first(c):
    return _wrap(f'''def build_pass_manager(backend, seed):
    pm = generate_preset_pass_manager(2, backend, seed_transpiler=seed)
    layout = PassManager([passes.VF2Layout(target=backend.target, seed=seed, call_limit=10000000,
                                           max_trials={c["vf2_max_trials"]}, strict_direction=False)])
    layout.append(ConditionalController(
        [passes.SabreLayout(backend.target, max_iterations={c["max_iterations"]}, seed=seed,
                            swap_trials={c["swap_trials"]}, layout_trials={c["layout_trials"]})],
        condition=lambda ps: ps["layout"] is None))
{EMBED}    pm.layout = layout
    return pm
''')


def resynth(c):
    pre = f"    pm.init = PassManager([pm.init.to_flow_controller()] + [{RESYNTH_LOGICAL}] * {c['pre_rounds']})\n" \
        if c["pre_rounds"] else ""
    post = f"    pm.optimization.append([{RESYNTH}] * {c['post_rounds']})\n" if c["post_rounds"] else ""
    return _wrap(f'''def build_pass_manager(backend, seed):
    pm = generate_preset_pass_manager(2, backend, seed_transpiler=seed)
    layout = PassManager([passes.SabreLayout(backend.target, max_iterations={c["max_iterations"]}, seed=seed,
                                             swap_trials=20, layout_trials={c["layout_trials"]})])
{EMBED}    pm.layout = layout
{pre}{post}    return pm
''')


STRUCTURES = {
    "sabre_l2": {"space": qiskit_space.SPACE, "program": qiskit_space.program,
                 "defaults": {"max_iterations": 2, "layout_trials": 20, "swap_trials": 20, "vf2post": False,
                              "opt_level3": False}},
    "vf2_first": {"space": {"vf2_max_trials": [100, 1000, 10000], "max_iterations": [1, 2, 4],
                            "layout_trials": [10, 20, 40], "swap_trials": [10, 20, 40]},
                  "program": vf2_first,
                  "defaults": {"vf2_max_trials": 1000, "max_iterations": 2, "layout_trials": 20, "swap_trials": 20}},
    "resynth": {"space": {"pre_rounds": [0, 1, 2, 3], "post_rounds": [0, 1, 2], "layout_trials": [10, 20, 40],
                          "max_iterations": [2, 4]},
                "program": resynth,
                "defaults": {"pre_rounds": 1, "post_rounds": 1, "layout_trials": 20, "max_iterations": 2}},
}


def _ev():
    sys.path.insert(0, str(EXAMPLE))
    import evaluator
    return evaluator


def measure(out: Path, sabre_table: Path | None = None) -> None:
    ev = _ev()
    done = {(json.loads(l)["structure"], json.dumps(json.loads(l)["params"], sort_keys=True))
            for l in out.open()} if out.exists() else set()
    if sabre_table and sabre_table.exists():
        keys = list(qiskit_space.SPACE)
        with out.open("a") as f:
            for r in map(json.loads, sabre_table.open()):
                p = {k: r[k] for k in keys}
                if ("sabre_l2", json.dumps(p, sort_keys=True)) not in done:
                    f.write(json.dumps({"structure": "sabre_l2", "params": p, "score": 0.0 if r.get("error")
                                        else r["score"], "cpu_ratio": r.get("cpu_ratio"), "error": r.get("error")}) + "\n")
                    done.add(("sabre_l2", json.dumps(p, sort_keys=True)))
    rows = ev.load_split("train")
    with tempfile.TemporaryDirectory() as tmp:
        for name, st in STRUCTURES.items():
            keys = list(st["space"])
            for vals in itertools.product(*st["space"].values()):
                p = dict(zip(keys, vals))
                if (name, json.dumps(p, sort_keys=True)) in done:
                    continue
                f = Path(tmp) / "p.py"
                f.write_text(st["program"](p))
                m = ev.score(str(f), rows, [0, 1], check=True)
                with out.open("a") as fo:
                    fo.write(json.dumps({"structure": name, "params": p, "score": m["combined_score"],
                                         "cpu_ratio": m.get("cpu_ratio"), "error": m.get("error")}) + "\n")
                print(name, p, round(m["combined_score"], 4), m.get("cpu_ratio") and round(m["cpu_ratio"], 2),
                      (m.get("error") or "")[:60], flush=True)


def analyse(table: Path, heldout: bool = True) -> None:
    rows = [json.loads(l) for l in table.open()]
    print(f"{len(rows)} points; over the CPU cap or failing: {sum(1 for r in rows if r.get('error'))}")
    tuned, allv = {}, {}
    print("\n1. inner tuner per structure (GA, budget 30, 20 seeds), against the measured table")
    for name, st in STRUCTURES.items():
        look = {json.dumps(r["params"], sort_keys=True): r["score"] for r in rows if r["structure"] == name}
        valid = [r for r in rows if r["structure"] == name and not r.get("error")]
        best = max(valid, key=lambda r: r["score"])
        default = look.get(json.dumps(st["defaults"], sort_keys=True))
        space = {k: {"values": v} for k, v in st["space"].items()}
        found = [tuner.TUNERS[tuner.DEFAULT](space, lambda c: look[json.dumps(c, sort_keys=True)], 30, seed=s).best()[1]
                 for s in range(20)]
        tuned[name], allv[name] = best, [math.log(r["score"]) for r in valid]
        print(f"  {name:10} {len(look):4d} points ({len(valid)} valid)  default {default:.4f}  best {best['score']:.4f} "
              f"{best['params']}  tuner mean {S.mean(found):.4f}, {sum(f >= best['score'] - 0.003 for f in found)}/20 "
              f"within 0.003")
    within = S.mean(S.pvariance(v) for v in allv.values())
    between = S.pvariance([S.mean(v) for v in allv.values()])
    print(f"\n2. variance of log(score) over valid points, structures weighted equally: within {within:.6f} "
          f"({within / (within + between):.0%}), between {between:.6f} ({between / (within + between):.0%})")
    print("   tuned bests: " + ", ".join(f"{n} {t['score']:.4f}" for n, t in tuned.items()))
    if heldout:
        ev = _ev()
        print("\n3. heldout, seeds 0-3 (mean of per-seed scores; CPU x level 2)")
        with tempfile.TemporaryDirectory() as tmp:
            for name, st in STRUCTURES.items():
                for label, p in (("default", st["defaults"]), ("tuned", tuned[name]["params"])):
                    f = Path(tmp) / "p.py"
                    f.write_text(st["program"](p))
                    per = [ev.evaluate_heldout(str(f), s) for s in range(4)]
                    sc = [m["combined_score"] for m in per]
                    cpu = S.mean(m.get("cpu_ratio") or 0 for m in per)
                    print(f"  {name:10} {label:8} {S.mean(sc):.4f} (seeds {' '.join(f'{x:.4f}' for x in sc)}) "
                          f"cpu {cpu:.2f} {p} {(per[0].get('error') or '')[:50]}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "analyse":
        analyse(Path(sys.argv[2]))
