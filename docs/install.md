## Installation

PySCF is most reliable from conda, and a fresh environment avoids conflicts with older Qiskit
installations (Qiskit Nature currently requires Qiskit < 2.0):

```bash
conda create -n ase-vqe python=3.12
conda activate ase-vqe
conda install -c conda-forge pyscf
pip install "ase_quantum_vqe[all] @ git+https://github.com/thequantumchemist/ase_quantum_vqe.git"
```

Install only the parts you need from a clone of the repository:

```bash
git clone https://github.com/thequantumchemist/ase_quantum_vqe.git
cd ase_quantum_vqe
pip install -e .                # Qiskit backend and PySCF reference
pip install -e ".[pennylane]"   # + PennyLane backend (analytic forces)
pip install -e ".[aer]"         # + finite-shot sampling with qiskit-aer
pip install -e ".[falcon]"      # + FALCON active learning (pins ASE <= 3.26)
pip install -e ".[jedi]"        # + JEDI strain analysis
pip install -e ".[all]"         # everything, including matplotlib and pytest
```

`ase_quantum_vqe` requires Python 3.10 or later and depends on `numpy`, `scipy`, `ase` (3.26 or later),
`pyscf`, `joblib`, `qiskit` 1.4.4, `qiskit-algorithms` 0.3.1, `qiskit-nature` 0.7.2 and
`qiskit-ibm-runtime` 0.41.1. The exact versions used for the paper are listed in `requirements.txt`.

Check the installation:

```bash
pytest -q tests
```

!!! warning "AGOX and NumPy 2"
    FALCON's default GPR model comes from AGOX. With NumPy 2, AGOX 3.11 stops during hyperparameter
    training with `TypeError: only 0-dimensional arrays can be converted to Python scalars`. Until this
    is fixed upstream, change one line in `agox/models/GPR/GPR.py`:

    ```python
    P, grad_P = -float(np.squeeze(P)), -np.asarray(grad_P, dtype="float64")
    ```

    The `DistanceGPR` surrogate of this package does not depend on AGOX.
