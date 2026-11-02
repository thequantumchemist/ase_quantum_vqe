# Paper: Table 3 (force accuracy and cost, Qiskit finite differences vs PennyLane analytic). Runtime: ~2 min (one core).
"""Benchmark: force accuracy and cost of the two ASE calculators against FCI (STO-3G).

Qiskit backend: central finite differences of ADAPT-VQE energies.
PennyLane backend: analytic forces (autodiff through differentiable HF), corrected calculator.
Run single-core for a fair timing comparison:  python bench_forces.py
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase import Atoms
from ase.units import Hartree
from pyscf import gto, scf, fci

from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator   # corrected version


def e_fci(atoms, charge):
    mol = gto.M(atom=[(s, p) for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
                basis="sto-3g", charge=charge, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    return fci.FCI(mf).kernel(tol=1e-12)[0] * Hartree


def f_fci(atoms, charge, d=1e-4):
    F = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        for j in range(3):
            a = atoms.copy(); a.positions[i, j] += d; ep = e_fci(a, charge)
            a = atoms.copy(); a.positions[i, j] -= d; em = e_fci(a, charge)
            F[i, j] = -(ep - em) / (2 * d)
    return F


systems = [("H2 r=0.80", Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80]]), 0),
           ("H3+ distorted", Atoms("H3", positions=[[0, 0, 0], [0.9, 0, 0], [0.4, 0.8, 0]]), 1)]

for name, atoms, q in systems:
    ref = f_fci(atoms, q)
    for label, calc in [("Qiskit FD d=1e-3", QiskitVQECalculator(charge=q, delta=1e-3, n_jobs=1)),
                        ("Qiskit FD d=1e-2", QiskitVQECalculator(charge=q, delta=1e-2, n_jobs=1)),
                        ("PennyLane analytic", PennyLaneAdaptVQECalculator(charge=q))]:
        a = atoms.copy(); a.calc = calc
        t = time.time(); f = a.get_forces(); dt = time.time() - t
        print(f"{name:14s} {label:20s} max|dF| = {np.abs(f - ref).max():.2e} eV/A   time = {dt:6.1f} s", flush=True)
