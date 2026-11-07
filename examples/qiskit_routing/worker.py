# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Worker for evaluator.py: run a candidate pipeline on circuits, in the Qiskit
interpreter (``QISKIT_EVAL_PYTHON``), isolated from the evaluator's process.

    python -I worker.py PROGRAM < jobs.jsonl > results.jsonl

Each job is {"path", "seed", "check"}; each result line has the circuit's CZ
count, CPU seconds (all threads), and whether the output uses only the
target's gates and coupling edges (``valid``) and, if ``check``, whether it
implements the input circuit (``equivalent``).
"""

import importlib.util
import json
import os
import random
import sys
import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeTorino

MAX_CHECK_QUBITS = 20     # active qubits in the output for the statevector check
CHECK_BUDGET = 2e8        # gates x 2**qubits per check: a few seconds of statevector work


def load(program):
    spec = importlib.util.spec_from_file_location("candidate", program)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    limit = os.environ.get("QISKIT_EVAL_MAX_PRESET")
    if limit is not None and hasattr(mod, "generate_preset_pass_manager"):
        preset = mod.generate_preset_pass_manager

        def guarded(*args, **kw):
            level = kw.get("optimization_level", args[0] if args else 2)
            if level > int(limit):
                raise ValueError(f"preset optimization level {level} is not available (QISKIT_EVAL_MAX_PRESET={limit})")
            return preset(*args, **kw)

        mod.generate_preset_pass_manager = guarded
    return mod.build_pass_manager


def valid(out, backend):
    names = set(backend.target.operation_names) | {"barrier"}
    edges = set(backend.coupling_map.get_edges())
    for inst in out.data:
        op = inst.operation
        if op.name in ("barrier", "delay"):
            continue
        if op.name not in names:
            return False
        if op.num_qubits == 2:
            q = tuple(out.find_bit(x).index for x in inst.qubits)
            if q not in edges and q[::-1] not in edges:
                return False
        elif op.num_qubits > 2:
            return False
    return True


def equivalent(original, out, trials=2, seed=0):
    """Random-input check: prep -> transpiled -> inverse(prep + original) must give |0...0>.

    None if the output can't be checked this way (no layout, or more than
    MAX_CHECK_QUBITS active qubits or CHECK_BUDGET of simulation)."""
    if out.layout is None:
        return None
    init = out.layout.initial_index_layout(filter_ancillas=True)
    final = out.layout.final_index_layout(filter_ancillas=True)
    active = sorted({out.find_bit(q).index for inst in out.data for q in inst.qubits} | set(init) | set(final))
    gates = sum(1 for i in out.data if i.operation.name not in ("barrier", "delay")) + 2 * original.size()
    if len(active) > MAX_CHECK_QUBITS or gates * 2 ** len(active) > CHECK_BUDGET:
        return None
    pos = {p: i for i, p in enumerate(active)}
    rng = random.Random(seed)
    for _ in range(trials):
        prep = QuantumCircuit(original.num_qubits)
        for v in range(original.num_qubits):
            prep.u(rng.uniform(0, 3.14159), rng.uniform(0, 6.28318), rng.uniform(0, 6.28318), v)
        test = QuantumCircuit(len(active))
        test.compose(prep, qubits=[pos[init[v]] for v in range(original.num_qubits)], inplace=True)
        for inst in out.data:
            if inst.operation.name in ("barrier", "delay"):
                continue
            test.append(inst.operation, [pos[out.find_bit(q).index] for q in inst.qubits])
        undo = prep.compose(original).inverse()
        test.compose(undo, qubits=[pos[final[v]] for v in range(original.num_qubits)], inplace=True)
        if Statevector(test).probabilities()[0] < 1 - 1e-6:
            return False
    return True


def main(program):
    build = load(program)
    backend = FakeTorino()
    warm = QuantumCircuit(3)    # untimed: the first transpile in a process pays one-off setup
    warm.h(0); warm.cx(0, 1); warm.cx(1, 2)
    try:
        build(backend, 0).run(warm)
    except Exception:  # noqa: BLE001 - reported per job below
        pass
    for line in sys.stdin:
        job = json.loads(line)
        res = {"path": job["path"], "seed": job["seed"]}
        try:
            qc = QuantumCircuit.from_qasm_file(job["path"]).remove_final_measurements(inplace=False)
            c0 = time.process_time()
            out = build(backend, job["seed"]).run(qc)
            res["cpu_s"] = time.process_time() - c0
            res["cz"] = out.count_ops().get("cz", 0)
            res["valid"] = valid(out, backend)
            if job.get("check") and res["valid"]:
                res["equivalent"] = equivalent(qc, out, seed=job["seed"])
        except Exception as e:  # noqa: BLE001 - a broken candidate is the common case
            res["error"] = f"{type(e).__name__}: {e}"[:300]
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
