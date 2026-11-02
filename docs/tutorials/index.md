# Tutorials

The tutorials build on each other, from a first VQE calculation to molecular dynamics with active
learning. Every page includes its code from a script in the
[`tutorials/`](https://github.com/thequantumchemist/ase_quantum_vqe/tree/main/tutorials) folder. These
scripts run as tests, so the code on these pages always works with the current version. Runtimes are for
one CPU core.

| Tutorial | What you learn | Extras | Runtime |
|---|---|---|---|
| [1. Quickstart](01-quickstart.md) | VQE geometry and frequency of H₂, comparison with FCI | – | ~1 min |
| [2. Analytic forces](02-analytic-forces.md) | finite-difference vs analytic forces, accuracy and cost | `pennylane` | ~1.5 min |
| [3. Active spaces](03-active-spaces.md) | frozen-core active spaces and matching classical references | – | seconds |
| [4. Shot noise](04-shot-noise.md) | sampled energies and forces | `pennylane`, `aer` | ~30 s |
| [5. Active-learning MD](05-active-learning-md.md) | MD with FALCON and a force-trained surrogate | `pennylane`, `falcon` | ~10 s |
| [6. IBM Quantum](06-ibm-quantum.md) | hardware execution, tested with a fake backend | `aer` | ~30 s |
| [7. Workflows](07-workflows.md) | optimization, vibrations and JEDI strain analysis of H₃⁺ | `pennylane`, `jedi` | ~2.5 min |

Install everything for the tutorials with

```bash
pip install "ase_quantum_vqe[all]"
```

All tutorials use the minimal STO-3G basis and small molecules, so that every VQE result can be
checked against exact references.
