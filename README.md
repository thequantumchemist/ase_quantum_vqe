# ASE–Qiskit VQE Interface

This repository provides an interface between the Atomic Simulation Environment (ASE) and Qiskit, enabling atomistic simulations using variational quantum algorithms (VQE) on either simulated quantum backends or real quantum hardware in a hybrid classical–quantum workflow.

![Alt text](qcmol.jpeg?raw=true "ASE_Quiskit")

---

## Authors

Wilke Dononelli


---

## Requirements

This package is intended for scientific use and depends on several libraries with strict version constraints.

### Core requirements

- Python **3.8 or later**
- NumPy
- ASE (Atomic Simulation Environment)
- Qiskit (with specific version restrictions, cee below)
- PySCF (required for quantum chemistry integrals)

### Qiskit version restrictions

This package is tested with the following versions:

- `qiskit == 1.4.4`
- `qiskit-ibm-runtime == 0.41.1`
- `qiskit-algorithms == 0.3.1`
- `qiskit-nature == 0.7.2`
- `qiskit-nature-pyscf >= 0.4.0`

⚠️ **Important:**
The new Qiskit packaging (Qiskit ≥ 1.0) is **not backward compatible** with older Qiskit installations.
If you already have Qiskit installed, you **mst use a fresh environment** to avoid dependency conflicts.

---

## Installation (Recommended: Conda)

### Why Conda?

This package depends on **PySCF**, which includes compiled C/Fortran extensions and frequently fails to install via `pip` alone.
For this reason, **using Conda is strongly recommended**, especially on Linux and HPC systems.

---

### Step 1: Create a fresh Conda environment

We consider creating a **new environment** with Python 3.11:

    conda create -n ase-qiskit-vqe python=3.11
    conda activate ase-qiskit-vqe

---

### Step 2: Install PySCF via Conda

    conda install -c conda-forge pyscf

This ensures that PySCF is linked correctly against BLAS/LAPACK and avoids compilation errors.

---

### Step 3: Install this package

Clone the repository and install it using `pip` in one step:
    
    pip install git+https://github.com/thequantumchemist/ase_quantum_vqe.git

Or alternately first clone the repository and then install it using `pip`:
    
    git clone https://github.com/thequantumchemist/ase_quantum_vqe.git
    cd ase_quantum_vqe
    python -m pip install .

For development, you may prefer an editable installation:

    python -m pip install -e .

---

## Notes on Existing Qiskit Installations

- Do **not** install this package into an environment that already contains an older Qiskit version.
- Mixing legacy Qiskit (`qiskit < 1.0`) with the new modular Qiskit stack will lead to import and runtime errors.
- Always use a **clean Conda environment** when working with this package.
- In addition, the package hasn't been tested with the new Qiskit (`qiskit > 2.0`). For the moment usage with `1.0 < qiskit < 2.0` is tested and recommended

---

## Minimal Example

The following example demonstrates a geometry optimization and vibrational analysis of an H2 molecule using a VQE-based quantum chemistry calculator interfaced with ASE.

    from ase import Atoms
    from ase.optimize.bfgslinesearch import BFGSLineSearch as BFGS
    from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
    from ase.vibrations import Vibrations
    import numpy as np

    # Create H2 molecule with a bond distance of 0.9 Æ

    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])

    # Define VQEcalculator using a local Aer simulator backend
    calc = QiskitVQECalculator(
        basis='sto3g',
        backend='aer',
        n_jobs=12,
        charge=0,
        spin=0,
        maxiter=60
    )

    # Attach calculator to ASE Atoms object
    h2.calc = calc

    # Geometry optimization
    dyn = BFGS(h2, trajectory='h2_vqe.traj')
    dyn.run(fmax=0.005)

    # Compute total energy
    e_vqe = h2.get_potential_energy()
    print(f"VQE total energy: {e_vqe:.8f} Ha")

    # Vibrational analysis
    vib = Vibrations(h2, name='vibvqe', nfree=4)
    vib.run()
    vib.summary()

---

## License

This project is licensed under the **GPL-3.0 License**. See the `LICENSE` file for details.
