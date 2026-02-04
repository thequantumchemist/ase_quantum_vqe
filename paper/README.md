This tutorial page provides an overview of the example scripts included in this repository.
Each example demonstrates a different level of complexity for combining the Atomic Simulation
Environment (ASE) with quantum chemistry calculations based on Qiskit VQE and classical
reference methods.

The examples are designed to be self-contained, well-documented, and suitable both for
learning and for benchmarking quantum–classical workflows.

---

## Overview of the Examples

- h2.py   : Minimal VQE workflow (geometry optimization, vibrations, CCSD reference) (fast/ a few minutes)
- h3+.py  : Global structure search using minima hopping (classical or quantum) (medium /depending on number of cores ~0.5-5 hours)
- h2_ML_FALCON_MD.py: Advanced example combining VQE, machine learning, and molecular dynamics (long / up to several hours)
- beh2.py : Advanced example combining VQE, machine learning, and molecular dynamics (long / more than a day)

---

## Prerequisites

Please install the code as introduced one layer above

---

## Example 1: h2.py — Minimal Quantum Chemistry Workflow

This example demonstrates:
- Geometry optimization of H2 using a VQE-based calculator
- Vibrational analysis using finite differences
- Comparison against a classical CCSD reference (PySCF)

Run with:
    python h2.py

---

## Example 2: h3+.py — Global Optimization with Minima Hopping

This example demonstrates:
- Global structure search using minima hopping
- Switching between classical and ADAPT-VQE energy evaluations
- Selection of the lowest-energy structure from a trajectory
- Vibrational analysis using finite differences
- JEDI strain analysis

Run with:
    python h3+.py

Note:
Delete hop.log and qn00*.traj before rerunning, or use a fresh directory.

---

## Example 3: h2_ML_FALCON_MD.py — On-the-Fly Quantum / ML Molecular Dynamics
This example demonstrates
- Geometry optimization of H₂ using a VQE-based quantum chemistry calculator
- Generation of high-quality quantum reference data from VQE calculations
- Construction of an on-the-fly machine-learning potential using FALCON
- Molecular dynamics simulations accelerated by a Gaussian Process Regression (GPR) model
- Dynamic switching between quantum reference evaluations and ML predictions
- Efficient sampling of finite-temperature dynamics at 600 K

Run with:
    python h2_ML_FALCON_MD.py
    
---

## Example 4: beh2.py — On-the-Fly Quantum / ML Molecular Dynamics

This example demonstrates:
- Geometry optimization of BeH2 using VQE
- Generation of quantum training data
- On-the-fly Machine Learning acceleration using FALCON
- Molecular dynamics with a Langevin thermostat

Run with:
    python beh2.py

---

## Recommended Learning Path

1. h2.py   – Learn the basic ASE–VQE workflow
2. h3+.py  – Explore global optimization and quantum/classical switching
3. h2_ML_FALCON_MD.py – Advanced quantum/ML molecular dynamics
4. beh2.py – Advanced quantum/ML molecular dynamics with more realistic molecule

---

## Scientific Motivation

The examples provided in this repository are designed to illustrate,
in a progressive manner, how quantum algorithms for electronic structure
calculations can be integrated into atomistic simulation workflows using
the Atomic Simulation Environment (ASE).

Together, the examples address key challenges in quantum chemistry and
materials modeling, including accuracy benchmarking, exploration of
potential energy surfaces, and scalability toward finite-temperature
dynamics using hybrid quantum–classical approaches.

Each example focuses on a distinct scientific question and computational
challenge.

---

### Example 1: H2 — Benchmarking Quantum Chemistry with VQE

The hydrogen molecule (H2) represents the simplest non-trivial molecular
system and serves as an ideal benchmark problem for quantum chemistry
methods.

The primary scientific motivation of this example is to validate the
Variational Quantum Eigensolver (VQE) against high-accuracy classical
reference methods. By comparing VQE results with coupled-cluster (CCSD)
calculations, the example provides a controlled assessment of the accuracy
of quantum-derived potential energy surfaces for geometry optimization
and vibrational analysis.

This example establishes a clear baseline for evaluating the capabilities
and limitations of near-term quantum algorithms in molecular simulations.

---

### Example 2: H3+ — Global Structure Search with Quantum Potentials

The H3+ cation is a paradigmatic system in molecular physics, characterized
by a highly fluxional potential energy surface with multiple competing
minima.

The scientific motivation of this example is to demonstrate how quantum 
calculations can be embedded into global structure search
algorithms, such as minima hopping. The example highlights the use of
quantum-derived energies and forces beyond local geometry optimization,
enabling the exploration of complex potential energy landscapes.

By allowing a direct switch between classical and quantum calculators,
this example provides insight into differences between classical and
quantum energy landscapes within an identical structural search framework.

---

### Example 3: H2 + FALCON — On-the-Fly Quantum / ML Molecular Dynamics

While quantum chemistry methods offer high accuracy, their computational
cost limits their direct application to long-time molecular dynamics
simulations.

The motivation of this example is to address this scalability challenge
by combining VQE-based quantum reference calculations with on-the-fly
machine-learning potentials using FALCON. The machine-learning model
dynamically decides when expensive quantum evaluations are required,
significantly reducing the overall computational cost.

Using H2 as a minimal test system allows systematic validation of this
hybrid quantum–machine-learning approach, including force consistency,
energy conservation, and finite-temperature sampling.

---

### Example 4: BeH2 + FALCON — Scaling Quantum–ML Dynamics to Larger Systems

The BeH2 molecule represents a more complex molecular system with increased
electronic structure complexity and a larger number of nuclear degrees
of freedom.

The scientific motivation of this example is to demonstrate the scalability
of hybrid quantum–machine-learning workflows beyond minimal test systems.
By combining VQE-based reference calculations with on-the-fly machine
learning, the example illustrates how finite-temperature molecular
dynamics can be performed efficiently while retaining quantum-chemical
accuracy.

This example serves as a proof of concept for extending quantum–ML
molecular dynamics toward chemically and physically relevant systems.

---

## License

This tutorial and all example scripts are provided under the GPL-3.0 License.
