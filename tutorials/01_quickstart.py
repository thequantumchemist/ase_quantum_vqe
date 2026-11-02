"""
Tutorial 1: geometry and vibrational frequency of H2 with ADAPT-VQE (Qiskit, ~1 min).

    pip install ase_quantum_vqe
    python 01_quickstart.py
"""
# --8<-- [start:imports]
from ase.build import molecule
from ase.optimize import BFGS
from ase.vibrations import Vibrations

from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
from ase_quantum_vqe.utils.pyscf import PySCFCalculator
# --8<-- [end:imports]

# --8<-- [start:calculator]
atoms = molecule("H2")
atoms.set_distance(0, 1, 0.90)                 # start away from the minimum
atoms.calc = QiskitVQECalculator(basis="sto3g", backend="statevector")
# --8<-- [end:calculator]

# --8<-- [start:optimise]
BFGS(atoms, trajectory="h2_vqe.traj").run(fmax=0.005)
e_vqe = atoms.get_potential_energy()
r_vqe = atoms.get_distance(0, 1)
# --8<-- [end:optimise]

# --8<-- [start:vibrations]
vib = Vibrations(atoms, name="vib_h2_vqe", delta=0.01, nfree=2)
vib.clean()
vib.run()
w_vqe = vib.get_frequencies()[-1].real
# --8<-- [end:vibrations]

# --8<-- [start:reference]
ref = atoms.copy()
ref.calc = PySCFCalculator(basis="sto-3g", method="fci", forces="analytic")
BFGS(ref, logfile=None).run(fmax=0.005)
vib_ref = Vibrations(ref, name="vib_h2_fci", delta=0.01, nfree=2)
vib_ref.clean()
vib_ref.run()

print(f"{'':10s} {'r_e (A)':>8s} {'E (eV)':>10s} {'omega (cm-1)':>13s}")
print(f"{'ADAPT-VQE':10s} {r_vqe:8.4f} {e_vqe:10.4f} {w_vqe:13.0f}")
print(f"{'FCI':10s} {ref.get_distance(0, 1):8.4f} {ref.get_potential_energy():10.4f} "
      f"{vib_ref.get_frequencies()[-1].real:13.0f}")
# --8<-- [end:reference]
