"""BeH2 (STO-3G, Be 1s frozen): ADAPT-VQE harmonic frequencies at a tightly converged geometry.

Paper: Results, BeH2 table. Linear BeH2 has four modes (doubly degenerate bend, symmetric and
antisymmetric stretch); all four are printed so that the splitting of the bend can be reported.
Reference values (FCI/CCSD in the same active space): beh2_reference.py.
Runtime: hours on a workstation (12 qubits, 6N+1 VQE solves per force call).
"""
import numpy as np
from ase import Atoms
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.vibrations import Vibrations
from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator

beh2 = Atoms("BeH2", positions=[[0, 0, 0], [1.33, 0, 0], [-1.33, 0.05, 0]])
beh2.calc = QiskitVQECalculator(basis="sto3g", backend="statevector", n_jobs=18,
                                coreorb=1, vqe_eigenvalue=1e-6, maxiter=150)
BFGSLineSearch(beh2, trajectory="beh2_opt.traj", maxstep=0.1).run(fmax=0.005)

vib = Vibrations(beh2, name="vib_beh2", delta=0.01, nfree=2)
vib.run(); vib.summary()
freqs = np.sort(np.real(vib.get_frequencies()))[-4:]
print("BeH2 ADAPT-VQE frequencies (cm-1): bend, bend, sym. stretch, antisym. stretch:", np.round(freqs, 1))
