# Scripts and data for the paper

*Variational Quantum Forces in Atomistic Simulation Workflows: Analytic Gradients, Measurement Noise, and
Force-Controlled Active Learning* (W. Dononelli).

Install the package with all extras first (see the top-level README), then run each script from its own
directory. Runtimes are for one CPU core unless stated otherwise.

## Where each result comes from

| Paper item | Script | Archived data | Runtime |
| --- | --- | --- | --- |
| H₂ validation, Qiskit (text, Results) | `h2.py` | – | minutes |
| H₂ validation, PennyLane (text, Results) | `h2_pennylane.py` | – | < 1 min |
| F₂ in a (2e, 2o) active space (text, Results) | `f2.py` | – | minutes |
| BeH₂ frequencies, ADAPT-VQE (table, Results) | `beh2_frequencies.py` | – | hours, 18 cores |
| BeH₂ frequencies, FCI and CCSD (table, Results) | `beh2_reference.py` | – | < 1 min |
| Force accuracy and cost (table) | `bench_forces.py` | `data/bench_forces_output.txt` | ~2 min |
| Sampling statistics (table) and Figure 2 | `noise_analysis/exact_variance.py`, `noise_analysis/plot_exact.py` | `data/noise/exact_variance.json` | ~3 min |
| Validation of the variances against qiskit-aer sampling (text) | `noise_analysis/validate_fd_and_seed.py`, `noise_analysis/validate_analytic.py` | – | ~5 min |
| Geometry optimization with sampled forces (table) | `noise_analysis/noisy_opt.py` (`SYS=H2` or `SYS=H3+`) | `data/noise/noisy_opt_*.json` | H₂ ~5 min; H₃⁺ ~2 min per run |
| H₃⁺ minima hopping, vibrations, JEDI (Figure 3, table) | `h3+.py` | – | hours |
| Figure 4 and the 10 ps MD table | `md_analysis/run_md_table.sh` (calls `h2_md.py`, `plot_md_final.py`, `summarize_md.py`) | `data/md/*.npz` | ~30 min in total |
| Offline test of surrogate settings (text, MD section) | `md_analysis/pseudo_al.py` | `data/md/pseudo_al.json` | ~2 min |
| BeH₂ trajectory (Figure 5) | `beh2_md.py` (needs `start.traj` and `train.traj` of the published run) | – | > 1 day, 18 cores |
| Figure S1 (original FALCON setup) | `legacy/h2_ML_FALCON_MD.py` | – | hours |

`noise_analysis/noise_forces_run.py` is an optional direct sampling benchmark; the paper uses the exact
variances from `exact_variance.py` instead.

## Archived data

`data/md/` holds every MD run used in Figure 4 and the MD table as compressed NumPy archives (time,
predicted and exact energies, bond length, temperature, VQE-call flags, predicted and exact bond forces).
`md_analysis/runio.py` loads a run from a fresh `<name>.json` written by `h2_md.py` if present, otherwise
from the archive, so `plot_md_final.py 4072` and `summarize_md.py` reproduce the figure and table without
rerunning anything. `md_analysis/h2_fci_grid.json` is the exact H₂/STO-3G potential (PySCF FCI energies and
gradients on a dense grid) used to validate trajectories; `oracle.py` rebuilds it if it is missing.

## `h2_md.py` options (environment variables)

| Variable | Values | Meaning |
| --- | --- | --- |
| `MODE` | `falcon`, `direct` | FALCON active learning, or dynamics with exact forces |
| `REF` | `pennylane`, `qiskit`, `fci`, `oracle` | reference calculator (`oracle`: exact FCI spline, fast) |
| `MODEL` | `agox`, `distance` | FALCON's default AGOX GPR (energies) or `DistanceGPR` (energies and forces) |
| `ACC_E`, `ACC_F` | eV, eV/Å | energy and force uncertainty thresholds (default 0.05, none) |
| `TRAIN_START`, `TRAIN_LOG` | int, 0/1 | initial VQE steps; FALCON's additional calls at steps 2ⁿ |
| `THERMO` | `falcon` | use FALCON's Langevin implementation in `direct` mode |
| `SEED`, `STEPS`, `OUT` | | random seed, number of 0.5 fs steps, output name |

All runs use Langevin dynamics at 600 K with a 0.5 fs time step and friction 0.002, as in the paper.
