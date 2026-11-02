# Paper: validation of the exact analytic-force variance against direct qiskit-aer sampling.
"""Validate the exact analytic-force variance against direct qiskit-aer sampling (H3+, largest component)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as onp
from ase import Atoms
from ase.units import Hartree, Bohr
import pennylane as qml
from pennylane import numpy as pnp
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer.primitives import Estimator as AerEstimator
from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator
from exact_variance import qwc_groups

a = Atoms("H3", positions=[[0, 0, 0], [0.9, 0, 0], [0.4, 0.8, 0]]); q = 1
calc = PennyLaneAdaptVQECalculator(charge=q); a.calc = calc; F = a.get_forces()
sym = a.get_chemical_symbols(); r0 = a.positions / Bohr
H, nq = qml.qchem.molecular_hamiltonian(sym, pnp.array(r0, requires_grad=False), charge=q, method="dhf")
c, ops = H.terms()
words = [qml.pauli.pauli_word_to_string(o, wire_map={w: w for w in range(nq)}) for o in ops]
J = onp.array(qml.jacobian(lambda r: qml.math.stack(qml.qchem.molecular_hamiltonian(sym, r, charge=q, method="dhf")[0].terms()[0]))(pnp.array(r0, requires_grad=True)))
i, j = onp.unravel_index(onp.argmax(onp.abs(F)), F.shape)
coef = J[:, i, j] * Hartree / Bohr
O = SparsePauliOp(words, coef).simplify(atol=1e-8)   # PL wire w <-> Qiskit qubit nq-1-w: labels coincide
w = pnp.array(calc.vqe_weights, requires_grad=False); hf = qml.qchem.hf_state(2, nq)
@qml.qnode(qml.device("default.qubit", wires=nq))
def state():
    qml.BasisState(hf, wires=range(nq))
    for wi, op in zip(w, calc.chosen_operators):
        (qml.SingleExcitation if len(op) == 2 else qml.DoubleExcitation)(wi, wires=op)
    return qml.state()
psi_pl = onp.array(state())
# PL: wire 0 = most significant bit; Qiskit: qubit 0 = least significant -> reversing wire labels makes them identical
psi = Statevector(psi_pl)
print("check <O> vs -F:", float(onp.real(psi.expectation_value(O))), -F[i, j])
shots = 10_000
var = sum(float(onp.real(psi.expectation_value(g.compose(g).simplify()) - psi.expectation_value(g) ** 2)) for g in qwc_groups(O))
qc = QuantumCircuit(nq); qc.prepare_state(psi.data, range(nq)); qc = transpile(qc, basis_gates=["u", "cx"])
est = AerEstimator(approximation=False, skip_transpilation=True)
samples = [est.run(qc, O, shots=shots, seed=int(s)).result().values[0] for s in onp.random.default_rng(1).integers(1e9, size=200)]
print(f"component ({i},{j}), {shots} shots: predicted sigma = {onp.sqrt(var / shots):.4f}, sampled std = {onp.std(samples):.4f}, mean - exact = {onp.mean(samples) + F[i, j]:+.4f} eV/A")
