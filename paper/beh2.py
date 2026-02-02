# ASE core objects for defining atomic structures
from ase import Atoms
from ase import units

# IO utilities for reading/writing structures and trajectories
from ase.io import read, write, Trajectory

# Geometry optimization algorithms
from ase.optimize import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch as BFGS

# Qiskit-based VQE quantum chemistry calculator
from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator

# Numerical utilities
import numpy as np

# FALCON on-the-fly machine-learning potential
from falcon_md.otf_calculator import FALCON
from falcon_md.models.agox_models import GPR

# Molecular dynamics utilities
from falcon_md.utils.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


# ---------------------------------------------------------------------
# 1. Define the BeH2 molecule
# ---------------------------------------------------------------------

# Create a BeH2 molecule with an initial (non-optimized) geometry
beh2 = Atoms(
    "BeH2",
    positions=[
        [1.4, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.1, 1.0, 2.0]
    ]
)

# Center the molecule in a large simulation box to avoid interactions
# with periodic images (important for isolated molecules)
beh2.center(vacuum=25.0)


# ---------------------------------------------------------------------
# 2. Setup the Qiskit VQE quantum chemistry calculator
# ---------------------------------------------------------------------

# Initialize a VQE-based calculator using a local Aer simulator backend
calc = QiskitVQECalculator(
    basis='sto3g',          # Minimal Gaussian basis set
    backend='aer',          # Local quantum circuit simulator
    n_jobs=18,              # Number of CPU cores for parallel execution
    charge=0,               # Total molecular charge
    spin=0,                 # Spin multiplicity (2S)
    vqe_eigenvalue=1e-06,   # Convergence threshold for VQE eigenvalue
    shots=5000,             # Number of measurement shots per circuit
    coreorb=1,              # Number of frozen core orbitals
    maxiter=150             # Maximum number of VQE optimizer iterations
)

# Attach the quantum calculator to the BeH2 Atoms object
beh2.calc = calc


# ---------------------------------------------------------------------
# 3. Geometry optimization using ASE
# ---------------------------------------------------------------------

# Perform a geometry optimization using the BFGS optimizer
# The trajectory of the optimization is stored in 'beh2opt.traj'
opt = BFGS(
    beh2,
    trajectory='beh2opt.traj',
    maxstep=0.1             # Maximum step size per optimization step
)

# Run the optimization until forces are below 0.05 eV/Å
# or a maximum of 20 steps is reached
opt.run(fmax=0.05, steps=20)


# ---------------------------------------------------------------------
# 4. Initialize velocities for molecular dynamics
# ---------------------------------------------------------------------

# Assign initial velocities according to a Maxwell-Boltzmann distribution
# corresponding to 300 K
MaxwellBoltzmannDistribution(beh2, temperature_K=300)


# ---------------------------------------------------------------------
# 5. Prepare training data for on-the-fly ML potential
# ---------------------------------------------------------------------

# Read all structures from the optimization trajectory
# These structures are used as initial training data for the ML model
training_data = read('beh2opt.traj@0:')


# ---------------------------------------------------------------------
# 6. Setup the FALCON on-the-fly (OTF) calculator
# ---------------------------------------------------------------------

T = 300               # Target temperature in Kelvin
accuracy_e = 0.2      # Energy accuracy threshold (epsilon) in eV

# Replace the quantum calculator with a FALCON OTF calculator
# FALCON will decide when to call the expensive reference calculator
# (e.g. Qiskit VQE) based on the estimated ML uncertainty
beh2.calc = FALCON(
    model=GPR(beh2),        # Gaussian Process Regression model
    calc=calcb,             # Reference calculator (e.g. VQE)  <-- must exist
    train_start=20,         # Number of steps before ML model is activated
    training_data=training_data,
    accuracy_e=accuracy_e
)


# ---------------------------------------------------------------------
# 7. Molecular dynamics simulation
# ---------------------------------------------------------------------

# Setup a Langevin molecular dynamics integrator
dyn = Langevin(
    beh2,
    1 * units.fs,           # Time step of 1 femtosecond
    temperature_K=T,        # Thermostat temperature
    friction=0.002          # Langevin friction coefficient
)

# Write the MD trajectory to file
traj = Trajectory('MD.traj', 'w', beh2)
dyn.attach(traj.write)


# ---------------------------------------------------------------------
# 8. Run the on-the-fly molecular dynamics simulation
# ---------------------------------------------------------------------

# Run MD for 10,000 steps
dyn.run(10000)
