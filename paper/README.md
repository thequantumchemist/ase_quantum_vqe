This tutorial page provides an overview of the example scripts included in this repository.
Each example demonstrates a different level of complexity for combining the Atomic Simulation
Environment (ASE) with quantum chemistry calculations based on Qiskit VQE and classical
reference methods.

The examples are designed to be self-contained, well-documented, and suitable both for
learning and for benchmarking quantum–classical workflows.

---

## Overview of the Examples

- h2.py   : Minimal VQE workflow (geometry optimization, vibrations, CCSD reference) (fast/ a few minutes)
- h3+.py  : Global structure search using minima hopping (classical or quantum) (medium /depending on number of cores ~1-5 hours)
- beh2.py : Advanced example combining VQE, machine learning, and molecular dynamics (long / more than a day)

---

## Prerequisites

Please install the code as introduced one layer above

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
- Local optimization, vibrational analysis, and JEDI strain analysis

Run with:
    python h3+.py

Note:
Delete hop.log and qn00*.traj before rerunning, or use a fresh directory.

---

## Example 3: beh2.py — On-the-Fly Quantum / ML Molecular Dynamics

This example demonstrates:
- Geometry optimization of BeH2 using VQE
- Generation of quantum training data
- On-the-fly ML acceleration using FALCON
- Molecular dynamics with a Langevin thermostat

Run with:
    python beh2.py

---

## Recommended Learning Path

1. h2.py   – Learn the basic ASE–VQE workflow
2. h3+.py  – Explore global optimization and quantum/classical switching
3. beh2.py – Advanced quantum/ML molecular dynamics

---

## License

This tutorial and all example scripts are provided under the GPL-3.0 License.
