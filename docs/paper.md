# Reproducing the paper

The [`paper/`](https://github.com/thequantumchemist/ase_quantum_vqe/tree/main/paper) folder of the
repository contains a script for every figure and table of *Variational Quantum Forces in Atomistic
Simulation Workflows* and the data of all MD runs.

| Paper item | Script | Runtime |
|---|---|---|
| H₂ validation (Qiskit, PennyLane) | `h2.py`, `h2_pennylane.py` | minutes |
| F₂ in a (2e, 2o) active space | `f2.py` | minutes |
| BeH₂ frequencies (ADAPT-VQE; FCI and CCSD references) | `beh2_frequencies.py`, `beh2_reference.py` | hours; < 1 min |
| Force accuracy and cost | `bench_forces.py` | ~2 min |
| Sampling statistics and Figure 2 | `noise_analysis/exact_variance.py`, `plot_exact.py` | ~3 min |
| Geometry optimization with sampled forces | `noise_analysis/noisy_opt.py` | minutes |
| H₃⁺ minima hopping, vibrations, JEDI (Figure 3) | `h3+.py` | hours |
| Figure 4 and the 10 ps MD table | `md_analysis/run_md_table.sh` | ~30 min |
| BeH₂ trajectory (Figure 5) | `beh2_md.py` | > 1 day, 18 cores |
| Figure S1 | `legacy/h2_ML_FALCON_MD.py` | hours |

`md_analysis/plot_md_final.py 4072` and `md_analysis/summarize_md.py` reproduce Figure 4 and the MD
table from the archived runs in `paper/data/md/` without new calculations.
