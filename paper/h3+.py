from ase import Atoms
#from ase.optimize import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch as BFGS
from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
from ase.vibrations import Vibrations, Infrared
from ase_quantum_vqe.utils.pyscf import PySCFCalculator
from ase.vibrations import Vibrations
import numpy as np
from ase.io import read, write, Trajectory
from strainjedi.jedi import Jedi
from ase import Atom, Atoms
from ase.optimize.minimahopping import MinimaHopping
from ase_quantum_vqe.utils.utils import random_positions_with_min_distance

print('If you want to repeat the calculation, you have to either delete the file hop.log and qn00....traj or run in different directories!')
###############################
#choose if a classical or a quantum (ADAPT-VQE) calculation should be performed 
usecalculator='classical' # alternative: 'quantum'
num_cpu_cores=9 #number of CPU cores used for numerical force evaluation
num_minima_hopping_steps=10
###############################
print('You are performing a '+usecalculator+' calculation using '+str(num_cpu_cores)+' cpu cores')

# create random positions
positions = random_positions_with_min_distance()
atoms = Atoms("H3", positions=positions)


# Set the calculators.
#VQE
vqe_calc = QiskitVQECalculator(
    basis='sto3g',
    backend='aer',   
    n_jobs=num_cpu_cores,         
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
#Classical
classic_calc=PySCFCalculator(basis='sto-3g', method='ccsd', charge=1, spin=0,n_jobs=num_cpu_cores)

if usecalculator == 'classical':
    print('Use classical PySCF calculator and not the VQE calculator')
    calc=classic_calc
else:
    print('Use VQE_ADAPT Calculator')
    calc=vqe_calc

atoms.calc = calc

# Instantiate and run the minima hopping algorithm.
hop = MinimaHopping(atoms, Ediff0=2.5, T0=4000.0)
hop(totalsteps=num_minima_hopping_steps)

# Analyze the minimum energy configuration
traj = Trajectory("minima.traj")
min_atoms = min(traj, key=lambda atoms: atoms.get_potential_energy())
min_atoms.calc=calc
#local structure optimization
dyn = BFGS(min_atoms, trajectory='localoptimization.traj')
dyn.run(fmax=0.005)

# vibrational analyzes
vib = Vibrations(min_atoms, name='vib',nfree=2)
vib.run()
vib.summary()
hessian=vib.get_vibrations()

# Create a displaced atoms object for the strain anlyzes
displaced_atoms=min_atoms.copy()
displaced_atoms.positions[1][2]+=0.1
displaced_atoms.calc=calc
displaced_atoms.get_potential_energy()
displaced_atoms.get_forces()
write('displaced_atoms.traj',displaced_atoms)

#Perform the JEDI strain analyzes
j = Jedi(min_atoms, displaced_atoms, hessian)
j.set_bond_params(covf=2.0,vdwf=0.9)
j.run()
j.vmd_gen()


