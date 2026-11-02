"""BeH2 Langevin MD at 300 K with Qiskit ADAPT-VQE forces and FALCON (energy criterion).

Paper: Results, BeH2 trajectory (Figure 5). This is the setting used for the published run:
restart from start.traj with initial training data train.traj (put both files next to this
script). If they are missing, the script creates them: it optimizes BeH2 with VQE, uses the
optimization frames as training data, and draws Maxwell-Boltzmann velocities at 300 K.
Runtime: long (more than a day on 18 cores for 10 ps).
"""
import os
import numpy as np
from ase import Atoms, units
from ase.io import read, Trajectory
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from falcon_md.otf_calculator import FALCON
from falcon_md.models.agox_models import GPR
from falcon_md.utils.langevin import Langevin
from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator

T, DT, FRICTION, STEPS = 300, 1.0, 0.002, 10000
ACCURACY_E = 0.25            # eV, energy-uncertainty threshold used for the published run

vqe = QiskitVQECalculator(basis="sto3g", backend="statevector", n_jobs=18,
                          coreorb=1, vqe_eigenvalue=1e-5, maxiter=150)

if os.path.exists("start.traj") and os.path.exists("train.traj"):
    beh2 = read("start.traj@-1")
    training_data = read("train.traj@0:")
else:
    beh2 = Atoms("BeH2", positions=[[1.4, 0, 0], [0, 0, 0], [2.1, 1.0, 2.0]])
    beh2.center(vacuum=25.0)
    beh2.calc = vqe
    BFGSLineSearch(beh2, trajectory="train.traj", maxstep=0.1).run(fmax=0.005)
    training_data = read("train.traj@0:")
    MaxwellBoltzmannDistribution(beh2, temperature_K=T, rng=np.random.default_rng(1))
    Stationary(beh2)

beh2.calc = FALCON(model=GPR(beh2), calc=vqe, train_start=0,
                   training_data=training_data, accuracy_e=ACCURACY_E)
dyn = Langevin(beh2, DT * units.fs, temperature_K=T, friction=FRICTION)
dyn.attach(Trajectory("beh2_md.traj", "w", beh2).write)
dyn.run(STEPS)
