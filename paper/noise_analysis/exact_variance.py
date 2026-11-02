# Paper: Table 5 and Figure 2 (exact sampling statistics of the force estimators). Runtime: ~3 min.
"""Exact sampling statistics of VQE force estimators (no sampling needed).

For a converged state |psi>, measuring a qubit-wise-commuting (QWC) group g of Pauli terms with s shots
gives an estimator of <O_g> with variance Var_psi(O_g)/s, where Var_psi(O) = <O^2> - <O>^2.
Summing over groups gives the exact variance of every force component:
  finite differences: F = -(E+ - E-)/(2 delta)  ->  sigma^2 = (V+ + V-)/(4 delta^2) / s
  analytic:           F = -<dH/dR>              ->  sigma^2 = sum_g Var(sum_{k in g} dc_k/dR P_k) / s
Both use Qiskit's QWC grouping (the grouping of the qiskit-aer Estimator), so measurement-setting
counts are directly comparable. Output: exact_variance.json
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as onp
from functools import reduce
from ase import Atoms
from ase.units import Hartree, Bohr
from pyscf import gto, scf, fci
from qiskit.quantum_info import Statevector, SparsePauliOp, PauliList
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

TARGETS = [0.05, 0.005]   # eV/A
SYSTEMS = [("H2", Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80]]), 0),
           ("H3+", Atoms("H3", positions=[[0, 0, 0], [0.9, 0, 0], [0.4, 0.8, 0]]), 1)]


def e_fci(atoms, charge):
    mol = gto.M(atom=[(s, p) for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
                basis="sto-3g", charge=charge, unit="Angstrom", verbose=0)
    return fci.FCI(scf.RHF(mol).run(conv_tol=1e-12)).kernel(tol=1e-12)[0] * Hartree


def f_fci(atoms, charge, d=1e-4):
    F = onp.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        for j in range(3):
            a = atoms.copy(); a.positions[i, j] += d; ep = e_fci(a, charge)
            a = atoms.copy(); a.positions[i, j] -= d; em = e_fci(a, charge)
            F[i, j] = -(ep - em) / (2 * d)
    return F


def adapt_state(atoms, charge):
    atom = [f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)]
    problem = PySCFDriver(atom=atom, basis="sto3g", charge=charge, spin=0, unit=DistanceUnit.ANGSTROM).run()
    mapper = JordanWignerMapper()
    qop = mapper.map(problem.hamiltonian.second_q_op())
    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    ans = UCC(problem.num_spatial_orbitals, problem.num_particles, "sd", mapper, initial_state=hf)
    vqe = VQE(ExactEstimator(), ans, optimizer=SLSQP(maxiter=200)); vqe.initial_point = onp.zeros(ans.num_parameters)
    res = AdaptVQE(vqe, gradient_threshold=1e-5, eigenvalue_threshold=1e-7).compute_minimum_eigenvalue(qop)
    psi = Statevector(res.optimal_circuit.assign_parameters(res.optimal_point))
    return psi, qop, sum(problem.hamiltonian.constants.values())


def qwc_groups(op: SparsePauliOp):
    """QWC groups as SparsePauliOps, identity-only groups dropped (need no measurement)."""
    groups = []
    for g in op.simplify(atol=1e-8).group_commuting(qubit_wise=True):
        g = g.simplify()
        if all(not p.x.any() and not p.z.any() for p in g.paulis):
            continue
        groups.append(g)
    return groups


def var_qiskit(psi, op):
    return sum(float(onp.real(psi.expectation_value(g.compose(g).simplify()) - psi.expectation_value(g) ** 2))
               for g in qwc_groups(op))


PAULI = {"I": onp.eye(2), "X": onp.array([[0, 1], [1, 0]]), "Y": onp.array([[0, -1j], [1j, 0]]),
         "Z": onp.diag([1, -1])}


def pmat(word):   # PennyLane order: wire 0 is the leftmost/most significant factor
    return reduce(onp.kron, [PAULI[c] for c in word])


out = {}
if __name__ == '__main__':
  for name, atoms, q in SYSTEMS:
      N = len(atoms); ref = f_fci(atoms, q); res = {}

      # ---------- finite differences (Qiskit) ----------
      for delta in (1e-3, 1e-2):
          F0 = onp.zeros((N, 3)); var = onp.zeros((N, 3)); G = 0
          for i in range(N):
              for j in range(3):
                  for sgn in (+1, -1):
                      a = atoms.copy(); a.positions[i, j] += sgn * delta
                      psi, op, k = adapt_state(a, q)
                      e = (float(onp.real(psi.expectation_value(op))) + k) * Hartree
                      F0[i, j] -= sgn * e / (2 * delta)
                      var[i, j] += var_qiskit(psi, op) * Hartree ** 2 / (4 * delta ** 2)
                      G += len(qwc_groups(op))
          res[f"qiskit_fd_{delta:g}"] = dict(bias=(F0 - ref).tolist(), var_per_shot=var.tolist(),
                                            groups=G / (6 * N), settings_per_force=G)
          print(f"{name} FD d={delta:g}: settings={G}, sigma*sqrt(s) = {onp.sqrt(var).max():.3g} eV/A (max comp)", flush=True)

      # ---------- analytic (PennyLane) ----------
      calc = PennyLaneAdaptVQECalculator(charge=q); a = atoms.copy(); a.calc = calc
      F0 = a.get_forces()
      symbols = a.get_chemical_symbols(); r0 = a.positions / Bohr
      H, nq = qml.qchem.molecular_hamiltonian(symbols, pnp.array(r0, requires_grad=False), charge=q, method="dhf")
      coeffs, ops = H.terms()
      words = [qml.pauli.pauli_word_to_string(o, wire_map={w: w for w in range(nq)}) for o in ops]
      J = onp.array(qml.jacobian(lambda r: qml.math.stack(qml.qchem.molecular_hamiltonian(
          symbols, r, charge=q, method="dhf")[0].terms()[0]))(pnp.array(r0, requires_grad=True)))   # (K, N, 3) Ha/Bohr
      w = pnp.array(calc.vqe_weights, requires_grad=False); chosen = calc.chosen_operators
      n_el = int(sum(a.get_atomic_numbers()) - q); hf = qml.qchem.hf_state(n_el, nq)

      @qml.qnode(qml.device("default.qubit", wires=nq))
      def state():
          qml.BasisState(hf, wires=range(nq))
          for wi, op in zip(w, chosen):
              (qml.SingleExcitation if len(op) == 2 else qml.DoubleExcitation)(wi, wires=op)
          return qml.state()
      psi = onp.array(state())
      keep = [k for k in range(len(words)) if max(abs(float(coeffs[k])), onp.abs(J[k]).max()) > 1e-8]
      words = [words[k] for k in keep]; J = J[keep]
      plist = PauliList(words)
      idx = {wd: k for k, wd in enumerate(words)}
      groups = []
      for g in plist.group_qubit_wise_commuting():
          members = [idx[p.to_label()] for p in g if p.to_label() != "I" * nq]
          if members:
              groups.append(members)
      var = onp.zeros((N, 3))
      mats = {k: pmat(words[k]) for k in range(len(words))}
      for i in range(N):
          for j in range(3):
              for g in groups:
                  O = sum(J[k, i, j] * mats[k] for k in g) * Hartree / Bohr
                  v = psi.conj() @ O @ O @ psi - (psi.conj() @ O @ psi) ** 2
                  var[i, j] += float(onp.real(v))
      res["pennylane_analytic"] = dict(bias=(F0 - ref).tolist(), var_per_shot=var.tolist(),
                                       groups=len(groups), settings_per_force=len(groups))
      print(f"{name} analytic: groups={len(groups)}, sigma*sqrt(s) = {onp.sqrt(var).max():.3g} eV/A (max comp)", flush=True)

      # ---------- shots required for target RMS ----------
      for key, r in res.items():
          b2 = float(onp.mean(onp.array(r["bias"]) ** 2)); v = float(onp.mean(r["var_per_shot"]))
          r["rms_bias"] = b2 ** 0.5; r["rms_sigma_sqrt_shots"] = v ** 0.5
          r["total_shots_for_rms"] = {str(t): (v / (t ** 2 - b2) * r["settings_per_force"] if t ** 2 > b2 else None)
                                      for t in TARGETS}
          print(f"  {key:22s} bias {r['rms_bias']:.2e}  sigma*sqrt(s) {r['rms_sigma_sqrt_shots']:.3g}  "
                f"total shots for 0.05: {r['total_shots_for_rms']['0.05']:.2e}", flush=True)
      out[name] = res
      json.dump(out, open("exact_variance.json", "w"), indent=1)
  print("done")
