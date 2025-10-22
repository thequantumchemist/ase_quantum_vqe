from ase import Atoms
from ase.optimize.bfgslinesearch import BFGSLineSearch as BFGS
from ase_quan tum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
from ase.vibrations import Vibrations, Infrared
from ase_quantum_vqe.utils.pyscf_calc import PySCFCalculator
from ase.vibrations import Vibrations
import numpy as np

# create H2 molecule with a bond distance of 0.9 A
h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
# Calculator using local Aer-simulator
calc = QiskitVQECalculator(
    basis='sto3g',
    backend='aer',  # local simulation
    n_jobs=12,      # number of CPU cores
    charge=0,
    spin=0,
    maxiter=60      # VQE-optimizer steps
)

# attach calculator to atoms object
h2.calc=calc


dyn = BFGS(h2, trajectory='h2_vqe.traj')
dyn.run(fmax=0.005)
e_vqe=h2.get_potential_energy()
vib = Vibrations(h2, name='vibvqe',nfree=4)
vib.run()
vib.summary()

# Copy atoms object to a new one
h2b = h2.copy()
#do classical reference calculation
calcb = PySCFCalculator(basis='sto-3g', method='ccsd', charge=0, spin=0,n_jobs=12)
h2b.calc=calcb

opt = BFGS(h2b, trajectory='h2_pyscf.traj')
opt.run(fmax=0.005)

e_ext=h2b.get_potential_energy()
vibb = Vibrations(h2b, name='vibext',nfree=4)
vibb.run()
vibb.summary()

print('E_exact, E_VQE, Ediff')
print(e_ext,e_vqe,e_ext-e_vqe)
