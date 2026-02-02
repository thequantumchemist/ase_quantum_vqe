# ASE core object for defining atomic structures
from ase import Atoms

# Geometry optimization algorithm with line search
from ase.optimize.bfgslinesearch import BFGSLineSearch as BFGS

# Qiskit-based VQE quantum chemistry calculator
from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator

# Vibrational analysis tools from ASE
from ase.vibrations import Vibrations, Infrared

# Classical reference quantum chemistry calculator based on PySCF
from ase_quantum_vqe.utils.pyscf import PySCFCalculator

# Numerical utilities
import numpy as np


# ---------------------------------------------------------------------
# 1. Define the H2 molecule
# ---------------------------------------------------------------------

# Create an H2 molecule with an initial bond distance of 0.9 Å
h2 = Atoms(
    "H2",
    positions=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.9]
    ]
)


# ---------------------------------------------------------------------
# 2. Setup the VQE-based quantum chemistry calculator
# ---------------------------------------------------------------------

calc = QiskitVQECalculator(
    basis='sto3g',          # Basis set for the electronic structure problem

    # optimizer=None,       # Qiskit optimizer object (if provided directly)
    # optimizer_name='SLSQP',# Name of the Qiskit optimizer ('SLSQP', 'COBYLA', 'L_BFGS_B')

    # delta=1e-3,           # Finite-difference displacement (Å) for numerical forces
    n_jobs=12,              # Number of CPU cores used for parallel execution
    charge=0,               # Total molecular charge
    spin=0,                 # Spin multiplicity (2S = N_alpha - N_beta)
    maxiter=60,             # Maximum number of VQE optimizer iterations

    backend='aer',          # Backend: 'aer' (local simulator) or IBMQ backend name
    # shots=4000,           # Number of measurement shots per quantum circuit
    # resilience_level=0,   # Error mitigation level (0–3)

    # ibmq_token=None,      # IBM Quantum API token (required if backend != 'aer')
    # estimator_override=None,  # External estimator (e.g. fixed runtime estimator)

    # nfree=2,              # Finite displacements per direction per atom (2 or 4)
    # coreorb=0,            # Number of frozen core orbitals
    # vqe_eigenvalue=1e-07, # Eigenvalue convergence threshold for ADAPT-VQE

    # **kwargs              # Additional keyword arguments passed internally
)

# Attach the VQE calculator to the ASE Atoms object
h2.calc = calc


# ---------------------------------------------------------------------
# 3. Geometry optimization using the VQE calculator
# ---------------------------------------------------------------------

# Optimize the molecular geometry using a BFGS optimizer
# The optimization trajectory is written to 'h2_vqe.traj'
dyn = BFGS(h2, trajectory='h2_vqe.traj')

# Run the optimization until forces are below 0.005 eV/Å
dyn.run(fmax=0.005)

# Store the VQE total energy at the optimized geometry
e_vqe = h2.get_potential_energy()


# ---------------------------------------------------------------------
# 4. Vibrational analysis using the VQE potential energy surface
# ---------------------------------------------------------------------

# Compute vibrational modes and frequencies using finite differences
vib = Vibrations(
    h2,
    name='vibvqe',
    nfree=4
)

vib.run()
vib.summary()


# ---------------------------------------------------------------------
# 5. Classical reference calculation using PySCF
# ---------------------------------------------------------------------

# Create a copy of the optimized structure to avoid modifying the VQE result
h2b = h2.copy()

# Initialize a high-accuracy classical quantum chemistry calculator
calcb = PySCFCalculator(
    basis='sto-3g',
    method='ccsd',
    charge=0,
    spin=0,
    n_jobs=12
)

# Attach the classical reference calculator
h2b.calc = calcb


# ---------------------------------------------------------------------
# 6. Geometry optimization using the classical reference calculator
# ---------------------------------------------------------------------

opt = BFGS(h2b, trajectory='h2_pyscf.traj')
opt.run(fmax=0.005)

# Store the reference (exact) total energy
e_ext = h2b.get_potential_energy()


# ---------------------------------------------------------------------
# 7. Vibrational analysis using the classical reference potential
# ---------------------------------------------------------------------

vibb = Vibrations(
    h2b,
    name='vibext',
    nfree=4
)

vibb.run()
vibb.summary()


# ---------------------------------------------------------------------
# 8. Compare VQE and classical reference results
# ---------------------------------------------------------------------

print('E_exact, E_VQE, E_diff')
print(e_ext, e_vqe, e_ext - e_vqe)
