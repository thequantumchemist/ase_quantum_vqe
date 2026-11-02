"""
Tutorial 2: finite-difference (Qiskit) versus analytic (PennyLane) forces (~1 min).

    pip install "ase_quantum_vqe[pennylane]"
    python 02_analytic_forces.py
"""
# --8<-- [start:imports]
import time
import numpy as np
from ase import Atoms

from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator
from ase_quantum_vqe.utils.pyscf import PySCFCalculator
# --8<-- [end:imports]

# --8<-- [start:system]
# a distorted H3+ (two electrons, six qubits), so that all force components are nonzero
h3 = Atoms("H3", positions=[[0, 0, 0], [0.9, 0, 0], [0.4, 0.8, 0]])

calculators = {
    "FCI (reference)": PySCFCalculator(method="fci", charge=1, forces="analytic"),
    "Qiskit, finite differences": QiskitVQECalculator(charge=1, delta=1e-3, n_jobs=1),
    "PennyLane, analytic": PennyLaneAdaptVQECalculator(charge=1),
}
# --8<-- [end:system]

# --8<-- [start:compare]
forces = {}
for name, calc in calculators.items():
    atoms = h3.copy()
    atoms.calc = calc
    t0 = time.time()
    forces[name] = atoms.get_forces()
    print(f"{name:28s} {time.time() - t0:6.1f} s")

ref = forces["FCI (reference)"]
for name in list(calculators)[1:]:
    print(f"{name:28s} max |F - F_FCI| = {np.abs(forces[name] - ref).max():.1e} eV/A")
# --8<-- [end:compare]
