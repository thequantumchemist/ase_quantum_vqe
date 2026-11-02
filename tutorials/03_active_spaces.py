"""
Tutorial 3: frozen-core active spaces and matching classical references (F2, ~30 s).

    python 03_active_spaces.py
"""
# --8<-- [start:imports]
import numpy as np
from ase import Atoms

from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
from ase_quantum_vqe.utils.pyscf import PySCFCalculator
# --8<-- [end:imports]

# --8<-- [start:scan]
# F2/STO-3G: 10 orbitals, 9 doubly occupied. coreorb=8 leaves the sigma_g / sigma_u* pair with
# two electrons, i.e. a (2e, 2o) active space on four qubits.
calcs = {
    "ADAPT-VQE": lambda: QiskitVQECalculator(coreorb=8),
    "FCI (same active space)": lambda: PySCFCalculator(method="fci", coreorb=8),
    "CCSD (same active space)": lambda: PySCFCalculator(method="ccsd", coreorb=8),
}
distances = np.array([1.25, 1.30, 1.35, 1.40])
energies = {}
for name, make in calcs.items():
    energies[name] = []
    for r in distances:
        f2 = Atoms("F2", positions=[[0, 0, 0], [0, 0, r]])
        f2.calc = make()
        energies[name].append(f2.get_potential_energy())
# --8<-- [end:scan]

# --8<-- [start:print]
print("r (A)  " + "  ".join(f"{n:>24s}" for n in calcs))
for i, r in enumerate(distances):
    print(f"{r:5.2f}  " + "  ".join(f"{energies[n][i]:24.5f}" for n in calcs))
dev = np.abs(np.array(energies["ADAPT-VQE"]) - np.array(energies["FCI (same active space)"])).max()
print(f"max |E_VQE - E_FCI| = {dev:.1e} eV")
# --8<-- [end:print]
