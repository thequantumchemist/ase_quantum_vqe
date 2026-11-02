"""
Tutorial 4: forces under finite sampling (~30 s).

    pip install "ase_quantum_vqe[pennylane,aer]"
    python 04_shot_noise.py
"""
# --8<-- [start:imports]
import numpy as np
from ase import Atoms

from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator
# --8<-- [end:imports]

h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80]])

# --8<-- [start:energy]
# exact versus sampled energies of the same converged state (Qiskit)
exact = h2.copy()
exact.calc = QiskitVQECalculator(backend="statevector")
e_exact = exact.get_potential_energy()

sampled = []
for seed in range(5):
    a = h2.copy()
    a.calc = QiskitVQECalculator(backend="aer_shots", shots=10_000, seed=seed)
    sampled.append(a.get_potential_energy())
print(f"exact energy {e_exact:.4f} eV; sampled (10^4 shots, full VQE under noise): "
      f"{np.mean(sampled):.4f} +- {np.std(sampled):.4f} eV")
# --8<-- [end:energy]

# --8<-- [start:forces]
# analytic forces from a converged state, estimated with finite shots (PennyLane)
a = h2.copy()
a.calc = PennyLaneAdaptVQECalculator()
f_exact = a.get_forces()[1, 2]
for shots in (1_000, 10_000, 100_000):
    f = []
    for seed in range(10):
        b = h2.copy()
        b.calc = PennyLaneAdaptVQECalculator(force_shots=shots, seed=seed)
        f.append(b.get_forces()[1, 2])
    print(f"{shots:>7d} shots: F_z = {np.mean(f):+.3f} +- {np.std(f):.3f} eV/A   (exact {f_exact:+.3f})")
# --8<-- [end:forces]
