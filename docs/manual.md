# Manual

## 1. Overview

| Class or function | Module | Purpose |
|---|---|---|
| `QiskitVQECalculator` | `ase_quantum_vqe.qiskit_vqe_calculator` | ADAPT-VQE with Qiskit; finite-difference forces |
| `make_ibm_estimator` | `ase_quantum_vqe.qiskit_vqe_calculator` | V1-compatible estimator for IBM (or fake) backends |
| `PennyLaneAdaptVQECalculator` | `ase_quantum_vqe.pennylane_calculator` | ADAPT-VQE with PennyLane; analytic forces |
| `PySCFCalculator` | `ase_quantum_vqe.utils.pyscf` | classical references (HF, CCSD, FCI) in the same active space |
| `DistanceGPR` | `ase_quantum_vqe.surrogates` | force-trained Gaussian process for FALCON |
| `random_positions_with_min_distance` | `ase_quantum_vqe.utils.utils` | random start structures for global search |

All calculators implement `energy` (eV) and `forces` (eV/Å); the VQE calculators also implement
`dipole` (Debye), so ASE's `Infrared` module works directly.

## 2. `QiskitVQECalculator`

The geometry is passed to Qiskit Nature's `PySCFDriver`. An optional frozen-core active space is built
with `ActiveSpaceTransformer`, the Hamiltonian is mapped with Jordan–Wigner, and ADAPT-VQE (operator
pool: UCCSD singles and doubles; Hartree–Fock reference; parameters initialized at zero) finds the
ground state. Forces are central finite differences of the total energy.

| Parameter | Default | Meaning |
|---|---|---|
| `basis` | `"sto3g"` | Gaussian basis set |
| `charge`, `spin` | `0`, `0` | total charge and 2S = N_α − N_β |
| `coreorb` | `0` | number of frozen doubly occupied orbitals |
| `optimizer_name` | `"SLSQP"` | `"SLSQP"`, `"COBYLA"` or `"L_BFGS_B"` |
| `optimizer` | `None` | a Qiskit optimizer object (overrides `optimizer_name`) |
| `maxiter` | `200` | optimizer iterations per VQE solve |
| `vqe_eigenvalue` | `1e-7` | ADAPT stops when the energy changes less than this (Ha); the gradient threshold is `1e-5` Ha |
| `delta` | `1e-3` | finite-difference step (Å) |
| `nfree` | `2` | 2- or 4-point stencil |
| `n_jobs` | `1` | parallel displaced solves (joblib) |
| `backend` | `"statevector"` | see below |
| `shots` | `4000` | shots per measurement setting (`"aer_shots"` and IBM backends) |
| `seed` | `None` | sampling seed (`"aer_shots"`) |
| `resilience_level` | `0` | Qiskit Runtime error mitigation level |
| `ibmq_token`, `ibm_channel`, `ibm_instance` | `None`, `"ibm_quantum_platform"`, `None` | Qiskit Runtime account |
| `estimator_override` | `None` | any V1 estimator, e.g. from `make_ibm_estimator` |

| `backend` | Execution |
|---|---|
| `"statevector"` | exact expectation values (Qiskit reference `Estimator`); `shots` ignored |
| `"aer_shots"` | finite-shot sampling with qiskit-aer, qubit-wise commuting groups |
| `"aer"` | deprecated alias of `"statevector"` |
| other | name of an IBM Quantum backend, via Qiskit Runtime and a V2→V1 adapter |

Cost per force call: 6N + 1 VQE solves (12N + 1 with `nfree=4`) for N atoms.

## 3. `make_ibm_estimator(backend, shots=4000, resilience_level=0, optimization_level=1)`

Wraps a Qiskit Runtime `EstimatorV2` for `backend` (a real or fake IBM backend) in the V1 interface
used by `qiskit-algorithms`. Circuits are transpiled with a preset pass manager and observables are
mapped onto the transpiled layout.

## 4. `PennyLaneAdaptVQECalculator`

The Hamiltonian is built with PennyLane's differentiable Hartree–Fock solver. ADAPT-VQE selects
single and double excitation gates (Givens rotations) by their energy gradient and re-optimizes all
parameters after each addition. Forces are F = −⟨ψ(θ\*)|∂Ĥ/∂R|ψ(θ\*)⟩ by automatic differentiation,
including the response of the Hartree–Fock orbitals; they equal the derivative of the energy for
converged parameters θ\*.

| Parameter | Default | Meaning |
|---|---|---|
| `basis` | `"sto-3g"` | Gaussian basis set |
| `charge`, `spin` | `0`, `0` | total charge and 2S |
| `coreorb` | `0` | number of frozen doubly occupied orbitals |
| `optimizer_name` | `"BFGS"` | `"BFGS"`, `"SLSQP"`, `"L-BFGS-B"` (exact simulators) or `"ADAM"`, `"GradientDescent"` |
| `maxiter`, `conv_tol` | `200`, `1e-9` | optimizer iterations and energy tolerance (Ha) |
| `vqe_lr` | `0.1` | step size for `"ADAM"` and `"GradientDescent"` |
| `vqe_eigenvalue` | `1e-5` | ADAPT gradient threshold (Ha) |
| `adapt_max_operators` | `30` | maximum number of ADAPT operators |
| `use_ADAPT_memory` | `True` | reuse the operator list and warm-start the parameters at the next geometry |
| `backend` | `"default.qubit"` | PennyLane device (`"lightning.qubit"` for speed) |
| `shots` | `None` | finite shots for everything (requires `"ADAM"` or `"GradientDescent"`) |
| `force_shots`, `seed` | `None`, `None` | noise studies: exact parameters, sampled forces |

With `use_ADAPT_memory=True`, the operator list chosen at the first geometry is kept. After large
geometry changes (minima hopping, long MD), create a new calculator to repeat the operator selection.

## 5. `PySCFCalculator`

| Parameter | Default | Meaning |
|---|---|---|
| `basis` | `"sto-3g"` | Gaussian basis set |
| `method` | `"ccsd"` | `"hf"`, `"ccsd"` or `"fci"` |
| `charge`, `spin` | `0`, `0` | total charge and 2S (open shells use ROHF orbitals) |
| `coreorb` | `0` | frozen orbitals; with `"fci"`, FCI in the frozen-core active space (CASCI) |
| `forces` | `"fd"` | `"fd"` (central differences) or `"analytic"` (PySCF gradients) |
| `delta`, `n_jobs` | `1e-3`, `1` | finite-difference step (Å) and parallel jobs |

## 6. `DistanceGPR`

A Gaussian process on all pairwise interatomic distances with a squared-exponential kernel, implementing
FALCON's model interface (`train`, `predict_energy`, `predict_forces`, `predict_uncertainty`,
`predict_uncertainty_forces`, `descriptor`).

| Parameter | Default | Meaning |
|---|---|---|
| `use_forces` | `True` | train on energies and forces (gradient-enhanced GP) |
| `noise_E`, `noise_F` | `1e-3`, `1e-2` | noise of energies (eV) and forces (eV/Å) |
| `length_scales` | 16 values, 0.03–1 Å | grid for the length scale (maximum marginal likelihood) |
| `s2_min` | `0.1` | lower bound of the kernel amplitude (eV²) |
| `n_fixed`, `l_default`, `s2_default` | `3`, `0.3`, `1.0` | fixed hyperparameters for fewer than `n_fixed` structures |

Use it with FALCON's `accuracy_f` (force threshold, e.g. 0.1 eV/Å), `train_log=False` and a small
`train_start`. The features suit small molecules with a fixed atom order.

## 7. Practical advice

* **Converge tightly.** Finite-difference forces amplify any residual error of the displaced solves by
  1/(2δ). Analytic forces require converged parameters.
* **Choose the force route by execution mode.** On exact simulators both routes work; with shot noise,
  only analytic forces are practical.
* **Use matching references.** Compare VQE with `PySCFCalculator(method="fci", coreorb=...)` in the same
  basis and active space.
* **Under noise, force-based convergence criteria are unreliable.** A noisy `fmax` test can stop too
  early (few force components) or never (many components).
