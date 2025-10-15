from ase import Atoms
#from ase.optimize import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch as BFGS
from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
from ase.vibrations import Vibrations, Infrared
from ase_quantum_vqe.utils.pyscf import PySCFCalculator
from ase.vibrations import Vibrations
import numpy as np
from ase.io import read
from strainjedi.jedi import Jedi
from ase import Atom, Atoms
from ase.optimize.minimahopping import MinimaHopping
from ase_quantum_vqe.utils.utils import random_positions_with_min_distance


positions = random_positions_with_min_distance()
atoms = Atoms("H3", positions=positions)

# Set the calculator.
calc = QiskitVQECalculator(
    basis='sto3g',
    backend='aer',    # <- Lokale Simulation
    n_jobs=18,         # parallele Prozessanzahl für Kräfte
    charge=1,
    spin=0,
    delta=0.01,
    shots=2000,
#    resilience_level=3,
#    optimizer_name='COBYLA',
    nfree=2,
#    coreorb=0,
    maxiter=100        # VQE-Optimierungsschritte
)

atoms.calc = calc

# Instantiate and run the minima hopping algorithm.
hop = MinimaHopping(atoms, Ediff0=2.5, T0=4000.0)
hop(totalsteps=50)


# Calculator mit lokalem Aer-Simulator
calcb = PySCFCalculator(basis='sto-3g', method='ccsd', charge=1, spin=0,n_jobs=9)

dyn = BFGS(atoms, trajectory='h3_vqe.traj')
dyn.run(fmax=0.005)
e_vqe=atoms.get_potential_energy()
vib = Vibrations(atoms, name='vibvqe',nfree=2)
vib.run()
vib.summary()
hessiana=vib.get_vibrations()

# Neutral singlet H2
#h2b = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.9]])
atomsb = atoms.copy()
atomsb.calc=calcb

opt = BFGS(atomsb, trajectory='h3_ext.traj',maxstep=0.05)
opt.run(fmax=0.005)

e_ext=atomsb.get_potential_energy()
vibb = Vibrations(atomsb, name='vibext',nfree=2)
vibb.run()
vibb.summary()
hessianb = vibb.get_vibrations()
#vibb.write_mode()

atomsc=atoms.copy()
atomsc.positions[1][2]+=0.1
atomsc.calc=calc
atomscc.get_potential_energy()

j = Jedi(atoms, atomsc, hessian)
j.set_bond_params(covf=2.0,vdwf=0.9)
j.run()
j.vmd_gen()

atomsd=atomsb.copy()
atomsd.positions[1][2]+=0.1
atomsd.calc=calcb
atomsd.get_potential_energy()

jb = Jedi(atomsb, atomsd, hessianb)
jb.set_bond_params(covf=2.0,vdwf=0.9)
jb.run()

print('E_exact, E_VQE, Ediff')
print(e_ext,e_vqe,e_ext-e_vqe)
