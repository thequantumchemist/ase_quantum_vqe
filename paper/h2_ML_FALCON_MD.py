# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

# ASE core objects and units
from ase import Atoms, units

# IO utilities for structures and trajectories
from ase.io import read, write, Trajectory

# Geometry optimization algorithms
from ase.optimize.bfgslinesearch import BFGSLineSearch as BFGS

# Qiskit-based VQE quantum chemistry calculator
from qiskit_vqe_calculator import QiskitVQECalculator

# Numerical utilities
import numpy as np

# FALCON on-the-fly machine-learning calculator
from falcon_md.otf_calculator import FALCON
from falcon_md.models.agox_models import GPR

# Molecular dynamics tools
from falcon_md.utils.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary
)

# Ray for parallel execution (used internally by FALCON)
import ray

# Initialize Ray with a fixed number of CPUs and a custom temporary directory
ray.init(num_cpus=6, _temp_dir='/scratch/wido/tmp/ray/')


# ---------------------------------------------------------------------
# 1. Define the molecular system
# ---------------------------------------------------------------------

# Create an H2 molecule with an initial bond length of ~0.7 Å
h2 = Atoms(
    "H2",
    positions=[
        [0.7, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]
)

# Center the molecule in a large vacuum box to avoid interactions
# with periodic images
h2.center(vacuum=25.0)


# ---------------------------------------------------------------------
# 2. Setup the Qiskit VQE calculator
# ---------------------------------------------------------------------

calc = QiskitVQECalculator(
    basis='sto3g',           # Minimal STO-3G basis set
    backend='aer',           # Local Qiskit Aer simulator backend
    n_jobs=6,                # Number of CPU cores for parallel execution

    # charge=0,              # Total molecular charge
    # spin=0,                # Spin multiplicity (2S)

    vqe_eigenvalue=1e-07,    # Convergence threshold for the VQE eigenvalue

    # shots=5000,            # Number of measurement shots per circuit
    # coreorb=1,             # Number of frozen core orbitals
    # maxiter=250            # Maximum number of VQE optimizer iterations
)

# Attach the quantum chemistry calculator to the Atoms object
h2.calc = calc


# ---------------------------------------------------------------------
# 3. Geometry optimization
# ---------------------------------------------------------------------

# Perform a local geometry optimization using the BFGS optimizer
# The optimization trajectory is written to 'h2opt.traj'
opt = BFGS(
    h2,
    trajectory='h2opt.traj',
    maxstep=0.1              # Maximum atomic displacement per optimization step
)

# Run the optimization until forces are below 0.05 eV/Å
# or until 20 steps are completed
opt.run(fmax=0.05, steps=20)


# ---------------------------------------------------------------------
# 4. Initialize velocities for molecular dynamics
# ---------------------------------------------------------------------

# Assign velocities according to a Maxwell-Boltzmann distribution
# corresponding to a temperature of 600 K
MaxwellBoltzmannDistribution(h2, temperature_K=600)

# Remove any residual center-of-mass translation and rotation
Stationary(h2)


# ---------------------------------------------------------------------
# 5. Prepare training data for on-the-fly ML potential
# ---------------------------------------------------------------------

# Read all structures from the geometry optimization trajectory
# These structures serve as initial training data for the ML model
training_data = read('h2opt.traj@0:')


# ---------------------------------------------------------------------
# 6. Setup the FALCON on-the-fly (OTF) calculator
# ---------------------------------------------------------------------

T = 600               # Target temperature in Kelvin
accuracy_e = 0.05     # Energy accuracy threshold (epsilon) in eV

# Replace the VQE calculator with a FALCON OTF calculator
# FALCON will decide when to call the expensive quantum calculator
# based on the estimated uncertainty of the ML model
h2.calc = FALCON(
    model=GPR(h2),            # Gaussian Process Regression model
    calc=calc,                # Reference calculator (VQE)
    train_start=50,           # Number of steps before ML model is activated
    training_data=training_data,
    accuracy_e=accuracy_e
)


# ---------------------------------------------------------------------
# 7. Molecular dynamics simulation
# ---------------------------------------------------------------------

# Setup a Langevin molecular dynamics integrator
dyn = Langevin(
    h2,
    0.5 * units.fs,           # Time step of 0.5 femtoseconds
    temperature_K=T,          # Thermostat temperature
    friction=0.002            # Langevin friction coefficient
)

# Write the MD trajectory to file
traj = Trajectory('MD.traj', 'w', h2)
dyn.attach(traj.write)


# ---------------------------------------------------------------------
# 8. Run the on-the-fly molecular dynamics simulation
# ---------------------------------------------------------------------

# Run MD for 5000 time steps
dyn.run(5000)
