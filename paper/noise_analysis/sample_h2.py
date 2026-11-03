# Paper: Supporting Information, Section S3 (sampled validation of Figure 2). Runtime: about 45 min on one core.
"""Sampled RMS force errors of H2 (r = 0.80 A) with qiskit-aer, to validate the exact curves of Figure 2.

Converged states (exact simulation) are measured with a finite number of shots per measurement setting
(qubit-wise commuting groups, as in the variance formula). Every circuit gets its own random seed.
  finite differences: 12 displaced energies per force, delta = 1e-3 and 1e-2 A, 10 repetitions
  analytic:          -<dH/dR> for each force component, 30 repetitions
Point = sqrt(mean squared deviation from FCI forces over components and repetitions); error bars from a
bootstrap over repetitions (16th-84th percentile).  Output: sampled_h2.json
"""
import json, time, warnings
warnings.filterwarnings("ignore")
import numpy as onp
from ase import Atoms
from ase.units import Hartree, Bohr
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer.primitives import Estimator as AerEstimator
import pennylane as qml
from pennylane import numpy as pnp
from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator
from exact_variance import adapt_state, f_fci

SHOTS = [1_000, 10_000, 100_000, 1_000_000]
REPS_FD, REPS_AN = 10, 30
rng = onp.random.default_rng(2026)
atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80]]); N = 2
ref = f_fci(atoms, 0)
est = AerEstimator(approximation=False, skip_transpilation=True)


def state_circuit(psi):
    qc = QuantumCircuit(psi.num_qubits); qc.prepare_state(psi.data, range(psi.num_qubits))
    return transpile(qc, basis_gates=["u", "cx"])


def summarize(sq_err_by_rep):
    """sq_err_by_rep: (reps, n_components). Returns RMS and bootstrap 16/84 percentiles."""
    a = onp.asarray(sq_err_by_rep)
    rms = float(onp.sqrt(a.mean()))
    boot = [onp.sqrt(a[rng.integers(0, len(a), len(a))].mean()) for _ in range(2000)]
    return rms, float(onp.percentile(boot, 16)), float(onp.percentile(boot, 84))


out = {"system": "H2", "r": 0.80, "shots_per_setting": SHOTS}

# ---------------- finite differences ----------------
for delta in (1e-3, 1e-2):
    jobs = []
    for i in range(N):
        for j in range(3):
            for sgn in (+1, -1):
                a = atoms.copy(); a.positions[i, j] += sgn * delta
                psi, op, k = adapt_state(a, 0)
                jobs.append((i, j, sgn, state_circuit(psi), op.simplify(atol=1e-12), k))
    res = []
    for shots in SHOTS:
        t0 = time.time(); sq = []
        for rep in range(REPS_FD):
            F = onp.zeros((N, 3))
            for (i, j, sgn, c, op, k) in jobs:
                e = (est.run(c, op, shots=shots, seed=int(rng.integers(1 << 31))).result().values[0] + k) * Hartree
                F[i, j] -= sgn * e / (2 * delta)
            sq.append(((F - ref) ** 2).ravel())
        res.append(summarize(sq))
        print(f"FD d={delta:g} shots={shots:>8}: RMS {res[-1][0]:.3e} [{res[-1][1]:.3e}, {res[-1][2]:.3e}]  ({time.time()-t0:.0f}s)", flush=True)
    out[f"qiskit_fd_{delta:g}"] = {"reps": REPS_FD, "rms": [r[0] for r in res], "lo": [r[1] for r in res], "hi": [r[2] for r in res]}
    json.dump(out, open("sampled_h2.json", "w"), indent=1)

# ---------------- analytic ----------------
calc = PennyLaneAdaptVQECalculator(); a = atoms.copy(); a.calc = calc; a.get_forces()
sym = a.get_chemical_symbols(); r0 = a.positions / Bohr
H, nq = qml.qchem.molecular_hamiltonian(sym, pnp.array(r0, requires_grad=False), method="dhf")
words = [qml.pauli.pauli_word_to_string(o, wire_map={w: w for w in range(nq)}) for o in H.terms()[1]]
J = onp.array(qml.jacobian(lambda r: qml.math.stack(qml.qchem.molecular_hamiltonian(
    sym, r, method="dhf")[0].terms()[0]))(pnp.array(r0, requires_grad=True)))
w = pnp.array(calc.vqe_weights, requires_grad=False); hf = qml.qchem.hf_state(2, nq)


@qml.qnode(qml.device("default.qubit", wires=nq))
def state():
    qml.BasisState(hf, wires=range(nq))
    for wi, op in zip(w, calc.chosen_operators):
        (qml.SingleExcitation if len(op) == 2 else qml.DoubleExcitation)(wi, wires=op)
    return qml.state()


psi = Statevector(onp.array(state()))          # PL wire w <-> qiskit qubit nq-1-w: labels coincide
circ = state_circuit(psi)
comps = []
for i in range(N):
    for j in range(3):
        coef = J[:, i, j] * Hartree / Bohr
        op = SparsePauliOp(words, coef).simplify(atol=1e-10)
        comps.append((i, j, op))
res = []
for shots in SHOTS:
    t0 = time.time(); sq = []
    for rep in range(REPS_AN):
        F = onp.zeros((N, 3))
        for (i, j, op) in comps:
            if len(op) == 1 and not op.paulis[0].x.any() and not op.paulis[0].z.any():
                F[i, j] = -float(onp.real(op.coeffs[0]))            # constant only: exact
            else:
                F[i, j] = -est.run(circ, op, shots=shots, seed=int(rng.integers(1 << 31))).result().values[0]
        sq.append(((F - ref) ** 2).ravel())
    res.append(summarize(sq))
    print(f"analytic shots={shots:>8}: RMS {res[-1][0]:.3e} [{res[-1][1]:.3e}, {res[-1][2]:.3e}]  ({time.time()-t0:.0f}s)", flush=True)
out["pennylane_analytic"] = {"reps": REPS_AN, "rms": [r[0] for r in res], "lo": [r[1] for r in res], "hi": [r[2] for r in res]}
json.dump(out, open("sampled_h2.json", "w"), indent=1)
print("done", flush=True)
