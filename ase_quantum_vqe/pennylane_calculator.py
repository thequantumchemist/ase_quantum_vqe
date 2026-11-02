import pennylane as qml
from pennylane import numpy as np
import numpy as onp
from scipy.optimize import minimize
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Debye, _e, Bohr


class PennyLaneAdaptVQECalculator(Calculator):
    """
    ASE calculator based on PennyLane ADAPT-VQE in an (optional) frozen-core active space.

    Energies are expectation values of the qubit Hamiltonian built with PennyLane's
    differentiable Hartree-Fock ("dhf"). Forces are obtained analytically as
    -dE/dR = -<psi(theta*)| dH(R)/dR |psi(theta*)>, i.e. by automatic differentiation of
    the energy with respect to the nuclear coordinates at fixed, converged circuit
    parameters theta*. This is exact only if theta* is variationally converged
    (dE/dtheta = 0); the optimizer settings below are chosen with that in mind.
    """
    implemented_properties = ['energy', 'forces', 'dipole']

    def __init__(self,
                 basis="sto-3g",
                 charge=0,
                 spin=0,                     # 2S = N_alpha - N_beta
                 coreorb=0,                  # number of frozen core orbitals
                 maxiter=200,                # max optimizer iterations per VQE solve
                 conv_tol=1e-9,              # energy convergence threshold (Hartree)
                 backend="default.qubit",    # "default.qubit" or "lightning.qubit"
                 vqe_eigenvalue=1e-5,        # ADAPT gradient threshold for adding operators
                 adapt_max_operators=30,     # max number of ADAPT operators
                 optimizer_name="BFGS",      # 'BFGS' / 'SLSQP' (scipy, exact simulators) or 'ADAM' / 'GradientDescent'
                 vqe_lr=0.1,                 # step size for ADAM / GradientDescent
                 use_ADAPT_memory=True,      # reuse operator list and warm-start parameters across geometries
                 shots=None,                 # None = exact expectation values
                 force_shots=None,           # noise studies: exact VQE, forces sampled with this many shots
                 seed=None,                  # seed for finite-shot sampling
                 **kwargs):
        super().__init__(**kwargs)
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.coreorb = coreorb
        self.maxiter = maxiter
        self.conv_tol = conv_tol
        self.backend = backend
        self.vqe_eigenvalue = vqe_eigenvalue
        self.adapt_max_operators = adapt_max_operators
        self.optimizer_name = optimizer_name.upper()
        self.vqe_lr = vqe_lr
        self.use_ADAPT_memory = use_ADAPT_memory
        self.shots = shots
        self.force_shots = force_shots
        self.seed = seed

        if self.optimizer_name in ("BFGS", "SLSQP", "L-BFGS-B"):
            if shots is not None:
                raise ValueError("scipy optimizers require exact expectation values (shots=None); "
                                 "use optimizer_name='ADAM' with shots.")
        elif self.optimizer_name not in ("ADAM", "GRADIENTDESCENT"):
            raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

        self.chosen_operators = []
        self.vqe_weights = np.array([], requires_grad=True)
        self.n_vqe_evaluations = 0   # bookkeeping: number of circuit executions (energy) in last call

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _apply(op, w):
        if len(op) == 2:
            qml.SingleExcitation(w, wires=op)
        else:
            qml.DoubleExcitation(w, wires=op)

    def _optimize(self, cost, weights):
        """Minimise cost(weights); returns converged weights."""
        if len(weights) == 0:
            return weights
        if self.optimizer_name in ("BFGS", "SLSQP", "L-BFGS-B"):
            grad = qml.grad(cost)
            f = lambda x: float(cost(np.array(x, requires_grad=True)))
            g = lambda x: onp.array(grad(np.array(x, requires_grad=True)), dtype=float)
            method = "BFGS" if self.optimizer_name == "BFGS" else self.optimizer_name
            res = minimize(f, onp.array(weights, dtype=float), jac=g, method=method,
                           options={"maxiter": self.maxiter, "gtol": 1e-7} if method == "BFGS"
                           else {"maxiter": self.maxiter, "ftol": self.conv_tol})
            return np.array(res.x, requires_grad=True)
        opt_cls = qml.AdamOptimizer if self.optimizer_name == "ADAM" else qml.GradientDescentOptimizer
        opt = opt_cls(stepsize=self.vqe_lr)
        e_old = cost(weights)
        for _ in range(self.maxiter):
            weights, e_new = opt.step_and_cost(cost, weights)
            if abs(e_new - e_old) < self.conv_tol:
                break
            e_old = e_new
        return weights

    # ---------------------------------------------------------------- calculate
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        symbols = atoms.get_chemical_symbols()
        angstrom_to_bohr = 1.0 / Bohr
        static_coords = np.array(atoms.get_positions() * angstrom_to_bohr, requires_grad=False)
        multiplicity = self.spin + 1

        # active space
        mol = qml.qchem.Molecule(symbols, static_coords, charge=self.charge, mult=multiplicity,
                                 basis_name=self.basis)
        total_electrons = sum(atoms.get_atomic_numbers()) - self.charge
        if self.coreorb > 0:
            active_orbitals = mol.n_orbitals - self.coreorb
            active_electrons = total_electrons - 2 * self.coreorb
        else:
            active_orbitals = None
            active_electrons = None

        ham_kwargs = dict(basis=self.basis, charge=self.charge, mult=multiplicity, method="dhf",
                          active_electrons=active_electrons, active_orbitals=active_orbitals)
        H_static, qubits = qml.qchem.molecular_hamiltonian(symbols, static_coords, **ham_kwargs)
        n_el = active_electrons if active_electrons is not None else total_electrons
        hf_init = qml.qchem.hf_state(n_el, qubits)

        dev = qml.device(self.backend, wires=qubits, seed=self.seed)
        diff_method = "backprop" if self.shots is None else "parameter-shift"

        @qml.qnode(dev, diff_method=diff_method, shots=self.shots)
        def energy_circuit(weights, hamiltonian, ops, extra_op=None, extra_w=0.0):
            qml.BasisState(hf_init, wires=range(qubits))
            for w, op in zip(weights, ops):
                self._apply(op, w)
            if extra_op is not None:
                self._apply(extra_op, extra_w)
            return qml.expval(hamiltonian)

        # ADAPT-VQE: grow the ansatz, re-optimising all parameters after each addition
        if not self.use_ADAPT_memory or not self.chosen_operators:
            singles, doubles = qml.qchem.excitations(n_el, qubits)
            pool = doubles + singles
            self.chosen_operators = []
            weights = np.array([], requires_grad=True)
            for _ in range(self.adapt_max_operators):
                grads = []
                for op in pool:
                    g = qml.grad(lambda x: energy_circuit(weights, H_static, self.chosen_operators, op, x))(
                        np.array(0.0, requires_grad=True))
                    grads.append(abs(float(g)))
                k = int(onp.argmax(grads))
                if grads[k] < self.vqe_eigenvalue:
                    break
                self.chosen_operators.append(pool[k])
                weights = np.array(onp.append(onp.array(weights, dtype=float), 0.0), requires_grad=True)
                weights = self._optimize(lambda w: energy_circuit(w, H_static, self.chosen_operators), weights)
            self.vqe_weights = weights
        else:
            # same operators, warm start from previous geometry; re-converge parameters
            self.vqe_weights = self._optimize(
                lambda w: energy_circuit(w, H_static, self.chosen_operators),
                np.array(self.vqe_weights, requires_grad=True))

        weights = np.array(self.vqe_weights, requires_grad=False)
        energy_hartree = energy_circuit(weights, H_static, self.chosen_operators)

        # analytic forces: dE/dR at fixed converged parameters (Hellmann-Feynman for the circuit)
        if 'forces' in properties:
            force_circuit = energy_circuit
            if self.force_shots is not None:
                # measurement-noise study: converged (exact) parameters, sampled force estimator
                @qml.qnode(dev, diff_method="parameter-shift", shots=self.force_shots)
                def force_circuit(weights, hamiltonian, ops):
                    qml.BasisState(hf_init, wires=range(qubits))
                    for w, op in zip(weights, ops):
                        self._apply(op, w)
                    return qml.expval(hamiltonian)

            def loss(r):
                H, _ = qml.qchem.molecular_hamiltonian(symbols, r, **ham_kwargs)
                return force_circuit(weights, H, self.chosen_operators)
            coords = np.array(atoms.get_positions() * angstrom_to_bohr, requires_grad=True)
            grad_coords = qml.grad(loss)(coords)
            self.results['forces'] = onp.array(-grad_coords * Hartree * angstrom_to_bohr, dtype=float)

        # dipole moment
        dipole_ops = qml.qchem.molecular_dipole(mol, method="dhf",
                                                active_electrons=active_electrons,
                                                active_orbitals=active_orbitals)
        dip = onp.zeros(3)
        for i in range(3):
            dip[i] = float(onp.real(energy_circuit(weights, dipole_ops[i], self.chosen_operators)))

        self.results['energy'] = float(energy_hartree) * Hartree
        self.results['dipole'] = dip * (_e * Bohr) / Debye
