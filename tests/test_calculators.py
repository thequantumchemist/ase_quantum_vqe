"""Fast consistency tests (H2/STO-3G, < 1 min in total). Run with:  pytest -q tests"""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Hartree
from pyscf import gto, scf, fci

from ase_quantum_vqe.utils.pyscf import PySCFCalculator
from ase_quantum_vqe.surrogates import DistanceGPR


def h2(r=0.80):
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, r]])


def e_fci(atoms, charge=0):
    mol = gto.M(atom=[(s, p) for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)],
                basis="sto-3g", charge=charge, unit="Angstrom", verbose=0)
    return fci.FCI(scf.RHF(mol).run(conv_tol=1e-12)).kernel(tol=1e-12)[0] * Hartree


def fd_forces(atoms, d=1e-4):
    F = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        for j in range(3):
            a = atoms.copy(); a.positions[i, j] += d; ep = e_fci(a)
            a = atoms.copy(); a.positions[i, j] -= d; em = e_fci(a)
            F[i, j] = -(ep - em) / (2 * d)
    return F


def test_pyscf_fci_energy_and_forces():
    a = h2(); a.calc = PySCFCalculator(method="fci", forces="analytic")
    assert abs(a.get_potential_energy() - e_fci(a)) < 1e-6
    assert np.abs(a.get_forces() - fd_forces(a)).max() < 1e-4


def test_pyscf_fd_and_analytic_forces_agree():
    a = h2(); a.calc = PySCFCalculator(method="ccsd", forces="fd")
    b = h2(); b.calc = PySCFCalculator(method="ccsd", forces="analytic")
    assert np.abs(a.get_forces() - b.get_forces()).max() < 1e-4


@pytest.mark.parametrize("nfree", [2, 4])
def test_qiskit_statevector_energy_and_forces(nfree):
    pytest.importorskip("qiskit_nature")
    from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
    a = h2(); a.calc = QiskitVQECalculator(backend="statevector", nfree=nfree)
    assert abs(a.get_potential_energy() - e_fci(a)) < 1e-4
    assert np.abs(a.get_forces() - fd_forces(a)).max() < 1e-3


def test_pennylane_analytic_forces():
    pytest.importorskip("pennylane")
    from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator
    a = h2(); a.calc = PennyLaneAdaptVQECalculator()
    assert abs(a.get_potential_energy() - e_fci(a)) < 1e-5
    assert np.abs(a.get_forces() - fd_forces(a)).max() < 1e-5


def test_distance_gpr_forces_are_energy_derivatives():
    data = []
    for r in (0.65, 0.74, 0.85):
        a = h2(r); a.calc = PySCFCalculator(method="fci", forces="analytic")
        e, f = a.get_potential_energy(), a.get_forces()
        a.calc = SinglePointCalculator(a, energy=e, forces=f); data.append(a)
    m = DistanceGPR(use_forces=True); m.train(data)
    x = h2(0.78); d = 1e-5
    xp, xm = x.copy(), x.copy(); xp.positions[1, 2] += d; xm.positions[1, 2] -= d
    num = -(m.predict_energy(xp) - m.predict_energy(xm)) / (2 * d)
    assert abs(num - m.predict_forces(x)[1, 2]) < 1e-5
