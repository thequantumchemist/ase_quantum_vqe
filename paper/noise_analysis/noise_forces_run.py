# Optional: direct sampled force benchmark (qiskit-aer and PennyLane shots); superseded by exact_variance.py.
"""Shot-noise analysis of VQE forces (measurement noise only).

For each system, the ADAPT-VQE state is first converged on an exact simulator. Forces are then
estimated with finite sampling:
  * Qiskit backend: central finite differences, every displaced energy sampled with the
    qiskit-aer Estimator (approximation=False, qubit-wise commuting grouping).
  * PennyLane backend: analytic forces F = -<psi|dH/dR|psi>, expectation values sampled with
    PennyLane's finite-shot simulator (parameter-shift / Hamiltonian-term splitting).
Errors are measured against FCI forces (STO-3G). Results -> noise_forces.json, noise_forces.pdf/png
"""
import json, time, warnings
warnings.filterwarnings("ignore")
import numpy as onp
from ase import Atoms
from ase.units import Hartree, Bohr
from pyscf import gto, scf, fci

from qiskit import transpile
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.primitives import Estimator as ExactEstimator
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_algorithms.minimum_eigensolvers import VQE, AdaptVQE
from qiskit_algorithms.optimizers import SLSQP

import pennylane as qml
from pennylane import numpy as pnp
from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator

SHOTS = [1_000, 10_000, 100_000, 1_000_000]   # shots per measurement setting (QWC group)
import os
REPS = int(os.environ.get("REPS", 20))
FD_SHOTS = [int(float(x)) for x in os.environ.get("FD_SHOTS", "1e3,1e4,1e5,1e6").split(",")]
DELTAS = [float(x) for x in os.environ.get("FD_DELTAS", "1e-3,1e-2").split(",")]
SYSTEMS = [("H2", Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80]]), 0),
           ("H3+", Atoms("H3", positions=[[0, 0, 0], [0.9, 0, 0], [0.4, 0.8, 0]]), 1)]


def e_fci(atoms, charge):
    mol = gto.M(atom=[(s, p) for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
                basis="sto-3g", charge=charge, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    return fci.FCI(mf).kernel(tol=1e-12)[0] * Hartree


def f_fci(atoms, charge, d=1e-4):
    F = onp.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        for j in range(3):
            a = atoms.copy(); a.positions[i, j] += d; ep = e_fci(a, charge)
            a = atoms.copy(); a.positions[i, j] -= d; em = e_fci(a, charge)
            F[i, j] = -(ep - em) / (2 * d)
    return F


def adapt_state(atoms, charge):
    """Noise-free ADAPT-VQE (same settings as QiskitVQECalculator); returns bound circuit, qubit H, constant."""
    atom = [f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)]
    problem = PySCFDriver(atom=atom, basis="sto3g", charge=charge, spin=0, unit=DistanceUnit.ANGSTROM).run()
    mapper = JordanWignerMapper()
    qop = mapper.map(problem.hamiltonian.second_q_op())
    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    ans = UCC(problem.num_spatial_orbitals, problem.num_particles, "sd", mapper, initial_state=hf)
    vqe = VQE(ExactEstimator(), ans, optimizer=SLSQP(maxiter=200)); vqe.initial_point = onp.zeros(ans.num_parameters)
    res = AdaptVQE(vqe, gradient_threshold=1e-5, eigenvalue_threshold=1e-7).compute_minimum_eigenvalue(qop)
    circ = res.optimal_circuit.assign_parameters(res.optimal_point)
    circ = transpile(circ, basis_gates=["u", "cx"], optimization_level=1)
    const = sum(problem.hamiltonian.constants.values())
    return circ, qop, const


def n_qwc_groups(qop):
    return len(qop.paulis.group_qubit_wise_commuting())


ONLY = os.environ.get("ONLY")
SYSTEMS = [x for x in SYSTEMS if ONLY is None or x[0] == ONLY]
OUT = os.environ.get("OUT", "noise_forces.json")
results = {"shots_per_setting": SHOTS, "reps": REPS, "systems": {}}
for name, atoms, q in SYSTEMS:
    t0 = time.time()
    N = len(atoms); ref = f_fci(atoms, q)
    sysres = {"n_atoms": N}

    # ---------------- Qiskit: finite differences with Aer sampling ----------------
    for delta in DELTAS:
        jobs = []                                    # (i, j, sign, circuit, op, const)
        for i in range(N):
            for j in range(3):
                for sgn in (+1, -1):
                    a = atoms.copy(); a.positions[i, j] += sgn * delta
                    c, op, k = adapt_state(a, q)
                    jobs.append((i, j, sgn, c, op, k))
        groups = n_qwc_groups(jobs[0][4])
        circs = [jb[3] for jb in jobs]; ops = [jb[4] for jb in jobs]
        exact = onp.array(ExactEstimator().run(circs, ops).result().values) + onp.array([jb[5] for jb in jobs])
        F0 = onp.zeros((N, 3))
        for (i, j, sgn, *_), e in zip(jobs, exact * Hartree):
            F0[i, j] -= sgn * e / (2 * delta)
        est = AerEstimator(approximation=False, skip_transpilation=True)
        rms, mx = [], []
        for shots in FD_SHOTS:
            r_rms, r_max = [], []
            for rep in range(REPS):
                # independent seed per circuit: qiskit-aer reuses one seed for all circuits in a batch, which makes
                # the +delta/-delta circuits of symmetry-equivalent displacements return identical samples
                vals = onp.array([est.run([c], [o], shots=shots, seed=100000 * rep + 100 * k + 7).result().values[0]
                                  for k, (c, o) in enumerate(zip(circs, ops))])
                E = (vals + onp.array([jb[5] for jb in jobs])) * Hartree
                F = onp.zeros((N, 3))
                for (i, j, sgn, *_), e in zip(jobs, E):
                    F[i, j] -= sgn * e / (2 * delta)
                d = F - ref
                r_rms.append(float(onp.sqrt(onp.mean(d ** 2)))); r_max.append(float(onp.abs(d).max()))
            rms.append(float(onp.mean(r_rms))); mx.append(float(onp.mean(r_max)))
            print(f"{name} FD d={delta:g} shots={shots:>8}: RMS err {rms[-1]:.3e} eV/A", flush=True)
        sysres[f"qiskit_fd_{delta:g}"] = {
            "rms_error": rms, "max_error": mx,
            "noise_free_rms_error": float(onp.sqrt(onp.mean((F0 - ref) ** 2))),
            "settings_per_force": 6 * N * groups, "qwc_groups": groups, "shots": FD_SHOTS}

    # ---------------- PennyLane: analytic forces with finite shots ----------------
    calc = PennyLaneAdaptVQECalculator(charge=q)
    a = atoms.copy(); a.calc = calc
    F0 = a.get_forces()                               # converges the state (exact)
    w = pnp.array(calc.vqe_weights, requires_grad=False); chosen = calc.chosen_operators
    symbols = a.get_chemical_symbols()
    H0, nq = qml.qchem.molecular_hamiltonian(symbols, pnp.array(a.positions / Bohr, requires_grad=False),
                                             charge=q, method="dhf")
    groups_pl = len(qml.pauli.group_observables(H0.terms()[1], grouping_type="qwc"))
    n_el = int(sum(a.get_atomic_numbers()) - q); hf = qml.qchem.hf_state(n_el, nq)
    rms, mx = [], []
    for shots in SHOTS:
        r_rms, r_max = [], []
        for rep in range(REPS):
            dev = qml.device("default.qubit", wires=nq, seed=1000 * rep + 7)

            @qml.qnode(dev, diff_method="parameter-shift", shots=shots)
            def circ(Hm):
                qml.BasisState(hf, wires=range(nq))
                for wi, op in zip(w, chosen):
                    (qml.SingleExcitation if len(op) == 2 else qml.DoubleExcitation)(wi, wires=op)
                return qml.expval(Hm)

            def loss(r):
                return circ(qml.qchem.molecular_hamiltonian(symbols, r, charge=q, method="dhf")[0])

            g = qml.grad(loss)(pnp.array(a.positions / Bohr, requires_grad=True))
            d = onp.array(-g * Hartree / Bohr) - ref
            r_rms.append(float(onp.sqrt(onp.mean(d ** 2)))); r_max.append(float(onp.abs(d).max()))
        rms.append(float(onp.mean(r_rms))); mx.append(float(onp.mean(r_max)))
        print(f"{name} PennyLane analytic shots={shots:>8}: RMS err {rms[-1]:.3e} eV/A", flush=True)
    sysres["pennylane_analytic"] = {
        "rms_error": rms, "max_error": mx,
        "noise_free_rms_error": float(onp.sqrt(onp.mean((F0 - ref) ** 2))),
        "settings_per_force": groups_pl, "qwc_groups": groups_pl}
    results["systems"][name] = sysres
    print(f"{name} done in {time.time() - t0:.0f} s", flush=True)
    json.dump(results, open(OUT, "w"), indent=1)

