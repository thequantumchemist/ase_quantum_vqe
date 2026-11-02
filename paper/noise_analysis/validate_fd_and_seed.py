"""Paper: validation of the exact finite-difference force variance against direct qiskit-aer sampling,
and the shared-seed check (qiskit-aer reuses one seed for identical circuits in a batch)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as onp
from ase import Atoms
from ase.units import Hartree
from qiskit import transpile
from qiskit_aer.primitives import Estimator as AerEstimator
import exact_variance as ev   # reuses adapt_state and var_qiskit (main loop is guarded)
from qiskit.circuit import QuantumCircuit
d = 1e-3; shots = 1000; R = 200
circs, ops, ks, Vs = [], [], [], []
for sgn in (+1, -1):
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80 + sgn * d]])
    psi, op, k = ev.adapt_state(a, 0)
    qc = QuantumCircuit(op.num_qubits); qc.prepare_state(psi.data, range(op.num_qubits))
    circs.append(transpile(qc, basis_gates=["u", "cx"])); ops.append(op); ks.append(k); Vs.append(ev.var_qiskit(psi, op))
pred = onp.sqrt(sum(Vs) / shots) * Hartree / (2 * d)
est = AerEstimator(approximation=False, skip_transpilation=True)
def fz(vals): return -((vals[0] + ks[0]) - (vals[1] + ks[1])) * Hartree / (2 * d)
same = [fz(est.run(circs, ops, shots=shots, seed=1000 * r + 7).result().values) for r in range(R)]
indep = [fz([est.run([c], [o], shots=shots, seed=onp.random.randint(1e9)).result().values[0] for c, o in zip(circs, ops)]) for r in range(R)]
print(f"predicted sigma (independent sampling): {pred:.1f} eV/A")
print(f"Aer, one seed per batch (as in benchmark): std = {onp.std(same):.1f} eV/A")
print(f"Aer, independent seeds per circuit:         std = {onp.std(indep):.1f} eV/A")
