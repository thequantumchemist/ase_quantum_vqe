"""F2 (STO-3G) in a (2e, 2o) frozen-core active space: CCSD (= FCI here) vs fixed UCCSD vs ADAPT-VQE.

Paper: Results, F2 paragraph. Eight doubly occupied orbitals are frozen (coreorb=8), leaving the
sigma_g / sigma_u* pair with two electrons (4 qubits). In this space UCCSD is exact, so fixed UCCSD
and ADAPT-VQE must reproduce the CCSD frequency (1673 cm-1).  Runtime: a few minutes.
"""
import warnings
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations
import ase_quantum_vqe.qiskit_vqe_calculator as qc
from ase_quantum_vqe.utils.pyscf import PySCFCalculator

warnings.filterwarnings("ignore")
N_JOBS = 4


def frequency(calc, label, r0=1.3152):
    atoms = Atoms("F2", positions=[[0, 0, 0], [0, 0, r0]])
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=0.005)
    vib = Vibrations(atoms, name=f"vib_f2_{label}", delta=0.01, nfree=2)
    vib.clean(); vib.run()
    w = np.real(vib.get_frequencies()).max()
    print(f"{label:12s} r_e = {atoms.get_distance(0, 1):.4f} A   E = {atoms.get_potential_energy():.3f} eV   "
          f"omega = {w:.1f} cm-1")


frequency(PySCFCalculator(method="ccsd", coreorb=8, n_jobs=N_JOBS), "CCSD")
frequency(qc.QiskitVQECalculator(coreorb=8, n_jobs=N_JOBS), "ADAPT-VQE")

# fixed UCCSD ansatz: use the plain VQE instead of AdaptVQE inside the calculator
_adapt = qc.AdaptVQE
qc.AdaptVQE = lambda vqe, **kw: vqe
try:
    frequency(qc.QiskitVQECalculator(coreorb=8, n_jobs=N_JOBS), "fixed UCCSD")
finally:
    qc.AdaptVQE = _adapt
