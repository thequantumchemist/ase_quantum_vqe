# 1. Quickstart: VQE geometry and frequency of H₂

The hydrogen molecule in a minimal basis is small enough to be solved exactly, which makes it the ideal
first test. This tutorial optimizes H₂ with ADAPT-VQE, computes its harmonic frequency, and compares
both with full configuration interaction (FCI).

This tutorial demonstrates:

- Attaching a VQE calculator to an ASE `Atoms` object
- Geometry optimization with ASE's BFGS optimizer
- A harmonic frequency with ASE's `Vibrations` module
- A classical FCI reference in the same basis with `PySCFCalculator`

```bash
python 01_quickstart.py
```

## Step 1: Define the calculator

```python
--8<-- "tutorials/01_quickstart.py:imports"

--8<-- "tutorials/01_quickstart.py:calculator"
```

`QiskitVQECalculator` builds the qubit Hamiltonian from PySCF integrals, maps it with the Jordan–Wigner
transformation and solves it with ADAPT-VQE from the Hartree–Fock reference. `backend="statevector"`
computes expectation values exactly.

## Step 2: Optimize the geometry

```python
--8<-- "tutorials/01_quickstart.py:optimise"
```

Every force call of the Qiskit backend solves the VQE at the geometry and at the 6N displaced
geometries of a central finite difference.

## Step 3: Vibrational frequency

```python
--8<-- "tutorials/01_quickstart.py:vibrations"
```

## Step 4: Compare with FCI

```python
--8<-- "tutorials/01_quickstart.py:reference"
```

```text
            r_e (A)     E (eV)  omega (cm-1)
ADAPT-VQE    0.7349   -30.9477          5003
FCI          0.7349   -30.9477          5003
```

VQE reproduces FCI. The minimal basis, not the quantum algorithm, limits the accuracy here: in a large
basis, H₂ has a bond length of 0.742 Å and a frequency of about 4400 cm⁻¹.
