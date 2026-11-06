# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Transpiler pipeline for openevolve-bedrock's Qiskit example (issue #30).

The evaluator calls ``build_pass_manager(backend, seed)`` once per circuit and
runs the pass manager it returns on the circuit. The pipeline must map the
circuit onto ``backend`` (IBM Torino, 133 qubits, CZ) and should use as few
CZ gates as it can, within twice the CPU time of Qiskit's preset level 2.

Only the function inside the EVOLVE-BLOCK may change. It may compose Qiskit's
own passes, stages and options. It may not define classes, import modules, or
use approximation (``approximation_degree``): every output is checked for
equivalence on small circuits and for basis and connectivity on all.
"""

from qiskit.passmanager.flow_controllers import ConditionalController, DoWhileController  # noqa: F401
from qiskit.transpiler import PassManager, StagedPassManager, generate_preset_pass_manager  # noqa: F401
from qiskit.transpiler import passes  # noqa: F401
from qiskit.transpiler.preset_passmanagers import common  # noqa: F401

# EVOLVE-BLOCK-START
def build_pass_manager(backend, seed):
    """Qiskit's preset optimization level 2, the reference the score is relative to."""
    return generate_preset_pass_manager(2, backend, seed_transpiler=seed)
# EVOLVE-BLOCK-END
