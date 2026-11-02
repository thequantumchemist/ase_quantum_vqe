"""ASE calculators for variational quantum eigensolver (VQE) energies, forces, and dipole moments.

Backends
--------
QiskitVQECalculator          (ase_quantum_vqe.qiskit_vqe_calculator)   finite-difference forces
PennyLaneAdaptVQECalculator  (ase_quantum_vqe.pennylane_calculator)    analytic forces
PySCFCalculator              (ase_quantum_vqe.utils.pyscf)             classical reference (HF/CCSD/FCI)
DistanceGPR                  (ase_quantum_vqe.surrogates.distance_gpr) force-trained GP surrogate for FALCON

The backends are imported from their submodules so that Qiskit and PennyLane stay optional
dependencies of each other.
"""
__version__ = "1.1.0"
