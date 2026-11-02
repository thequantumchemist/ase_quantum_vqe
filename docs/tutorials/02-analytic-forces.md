# 2. Finite-difference versus analytic forces

Workflows consume forces. This tutorial computes the forces on a distorted H₃⁺ ion with both backends and
compares them with exact FCI forces.

This tutorial demonstrates:

- Finite-difference forces from the Qiskit backend
- Analytic forces from the PennyLane backend
- Their accuracy and cost against an exact reference

```bash
python 02_analytic_forces.py
```

## Step 1: Set up the system and the calculators

```python
--8<-- "tutorials/02_analytic_forces.py:imports"

--8<-- "tutorials/02_analytic_forces.py:system"
```

The PennyLane backend builds the Hamiltonian with a differentiable Hartree–Fock solver. The force is the
expectation value of the derivative of the Hamiltonian, −⟨ψ(θ\*)|∂Ĥ/∂R|ψ(θ\*)⟩, obtained by automatic
differentiation at the converged circuit parameters θ\*.

## Step 2: Compare

```python
--8<-- "tutorials/02_analytic_forces.py:compare"
```

```text
FCI (reference)                 0.1 s
Qiskit, finite differences     49.2 s
PennyLane, analytic            34.1 s
Qiskit, finite differences   max |F - F_FCI| = 8.4e-04 eV/A
PennyLane, analytic          max |F - F_FCI| = 2.9e-08 eV/A
```

Analytic forces are exact to numerical precision and need one VQE solve per force call. Finite
differences need 6N+1 solves (19 for H₃⁺, which `n_jobs` distributes over cores); their accuracy depends
on the step `delta` and on how tightly every displaced solve is converged.

!!! note
    Analytic forces equal the derivative of the energy only if the circuit parameters are variationally
    converged. The PennyLane defaults (BFGS with exact gradients, `conv_tol=1e-9`) are chosen for this.
