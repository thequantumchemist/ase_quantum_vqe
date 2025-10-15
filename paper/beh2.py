from ase import Atoms
from ase import units
from ase.io import read,write,Trajectory
from ase.optimize import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch as BFGS
from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
import numpy as np
from falcon_md.otf_calculator import FALCON
from falcon_md.models.agox_models import GPR
from falcon_md.utils.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

beh2 = Atoms("BeH2", positions=[[1.4, 0, 0],[0.0,0.,0.],[2.1,1,2.0]])
beh2.center(vacuum=25.0)
calc = QiskitVQECalculator(
    basis='sto3g',
    backend='aer',
    n_jobs=18,
    charge=0,
    spin=0,
    vqe_eigenvalue=1e-06,
    shots=5000,
    coreorb=1,
    maxiter=150
)

beh2.calc=calc


opt = BFGS(beh2, trajectory='beh2opt.traj',maxstep=0.1)
opt.run(fmax=0.05,steps=20)

MaxwellBoltzmannDistribution(beh2, temperature_K=300)

training_data = read('beh2opt.traj@0:')


T = 300              # Temperature in K.
accuracy_e = 0.2    # Accuracy Threshold (Epsilon) in eV.
# Setup of the FALCON-OTF-Calculator

beh2.calc = FALCON(model = GPR(beh2),
                    calc = calcb,
                    train_start=20,
                    training_data = training_data,
                    accuracy_e = accuracy_e)

# Setup of the MD Simulation

dyn = Langevin(beh2, 1 * units.fs, temperature_K=T, friction=0.002)


traj = Trajectory(f'MD.traj', 'w', beh2)
dyn.attach(traj.write)


# Now run the OTF-MD Simulation!
dyn.run(10000)
