import pennylane as qml
from pennylane import numpy as np
from ase.calculators.calculator import Calculator, all_properties
from ase.units import Hartree, Debye, _e, Bohr

class PennyLaneAdaptVQECalculator(Calculator):
    """
    ASE-Calculator utilizing PennyLane ADAPT-VQE in an Active Space.
    Fully synchronized with Qiskit parameters including backend selection and maxiter.
    Robust against complex-type expectation values for dipole moments.
    """
    implemented_properties = ['energy', 'forces', 'dipole']

    def __init__(self,
                 basis="sto-3g",
                 charge=0,
                 spin=0,                  # 2S = #alpha - #beta Electrons
                 coreorb=0,               # Number of core orbitals to freeze
                 maxiter=5,               # Max iterations of VQE-optimizer per geometry step (maps to Qiskit maxiter)
                 backend="default.qubit", # "default.qubit" (Python) or "lightning.qubit" (Fast C++)
                 vqe_eigenvalue=1e-05,    # Gradient/Eigenvalue threshold for ADAPT VQE circuit growth
                 adapt_max_operators=4,  # Max number of operator layers in ADAPT circuit
                 optimizer_name="SLSQP",  # 'SLSQP' or 'ADAM' or 'GradientDescent'
                 vqe_lr=0.1,              # Learning rate for classical steps
                 use_ADAPT_memory=True,   # If True, reuses the circuit structure across geometry updates
                 shots=None,              # Total measurements per quantum circuit (None = exact analytic)
                 **kwargs):
        super().__init__(**kwargs)
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.coreorb = coreorb
        self.maxiter = maxiter
        self.backend = backend
        self.vqe_eigenvalue = vqe_eigenvalue
        self.adapt_max_operators = adapt_max_operators
        self.vqe_lr = vqe_lr
        self.use_ADAPT_memory = use_ADAPT_memory
        self.shots = shots

        # Select classical optimizer
        if optimizer_name.upper() == 'SLSQP' or optimizer_name.upper() == 'GRADIENTDESCENT':
            self.vqe_optimizer_cls = qml.GradientDescentOptimizer
        elif optimizer_name.upper() == 'ADAM':
            self.vqe_optimizer_cls = qml.AdamOptimizer
        else:
            raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

        self.chosen_operators = []  # Stores selected excitation operators
        self.vqe_weights = None      # Stores optimized VQE amplitudes

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_properties):
        super().calculate(atoms, properties, system_changes)

        symbols = atoms.get_chemical_symbols()
        angstrom_to_bohr = 1.8897259886
        coords_bohr = np.array(atoms.get_positions() * angstrom_to_bohr, requires_grad=True)
        multiplicity = self.spin + 1

        # 1. Total molecular properties calculation
        total_electrons = sum(atoms.get_atomic_numbers()) - self.charge
        static_coords = np.array(atoms.get_positions() * angstrom_to_bohr, requires_grad=False)

        _, total_qubits = qml.qchem.molecular_hamiltonian(
            symbols, static_coords, basis=self.basis, charge=self.charge, mult=multiplicity, method="dhf"
        )
        total_orbitals = total_qubits // 2

        # 2. Active Space Setup
        if self.coreorb > 0:
            active_orbitals = total_orbitals - self.coreorb
            active_electrons = total_electrons - 2 * self.coreorb
        else:
            active_orbitals = None
            active_electrons = None

        H_static, qubits = qml.qchem.molecular_hamiltonian(
            symbols, static_coords, basis=self.basis,
            charge=self.charge, mult=multiplicity, method="dhf",
            active_electrons=active_electrons,
            active_orbitals=active_orbitals
        )

        electrons_in_circuit = active_electrons if active_electrons is not None else total_electrons
        hf_init = qml.qchem.hf_state(electrons_in_circuit, qubits)

        # 3. Dynamic Device and Differentiation Configuration
        dev = qml.device(self.backend, wires=qubits)
        diff_method_to_use = "backprop" if self.shots is None else "parameter-shift"

        # 4. ADAPT-VQE logic
        if not self.use_ADAPT_memory or not self.chosen_operators:
            print("   [ADAPT-VQE] Generating and searching operator pool...")
            singles, doubles = qml.qchem.excitations(electrons_in_circuit, qubits)
            operator_pool = singles + doubles

            self.chosen_operators = []
            current_weights = np.array([], requires_grad=True)

            for _ in range(self.adapt_max_operators):
                pool_gradients = []
                for op in operator_pool:
                    def adapt_loss(excitation_weight):
                        @qml.qnode(dev, shots=self.shots)
                        def circuit():
                            qml.BasisState(hf_init, wires=range(qubits))
                            for w, chosen_op in zip(current_weights, self.chosen_operators):
                                if len(chosen_op) == 2:
                                    qml.SingleExcitation(w, wires=chosen_op)
                                else:
                                    qml.DoubleExcitation(w, wires=chosen_op)
                            if len(op) == 2:
                                qml.SingleExcitation(excitation_weight, wires=op)
                            else:
                                qml.DoubleExcitation(excitation_weight, wires=op)
                            return qml.expval(H_static)
                        return circuit()

                    grad_val = qml.grad(adapt_loss)(np.array(0.0, requires_grad=True))
                    pool_gradients.append(abs(grad_val))

                max_grad_idx = np.argmax(pool_gradients)
                max_grad = pool_gradients[max_grad_idx]

                if max_grad < self.vqe_eigenvalue:
                    break

                best_op = operator_pool[max_grad_idx]
                self.chosen_operators.append(best_op)
                current_weights = np.append(current_weights, 0.0)

            self.vqe_weights = np.array(current_weights, requires_grad=True)
        else:
            pass

        # 5. Final QNode execution layer for energy
        @qml.qnode(dev, diff_method=diff_method_to_use, shots=self.shots)
        def adapt_energy_circuit(weights, current_hamiltonian):
            qml.BasisState(hf_init, wires=range(qubits))
            for w, op in zip(weights, self.chosen_operators):
                if len(op) == 2:
                    qml.SingleExcitation(w, wires=op)
                else:
                    qml.DoubleExcitation(w, wires=op)
            return qml.expval(current_hamiltonian)

        # 6. VQE optimization loop
        if self.maxiter > 0 and len(self.vqe_weights) > 0:
            opt = self.vqe_optimizer_cls(stepsize=self.vqe_lr)
            for _ in range(self.maxiter):
                self.vqe_weights = opt.step(lambda w: adapt_energy_circuit(w, H_static), self.vqe_weights)

        # 7. Differentiable loss for analytic forces
        def loss_function(r_coords):
            H, _ = qml.qchem.molecular_hamiltonian(
                symbols, r_coords, basis=self.basis,
                charge=self.charge, mult=multiplicity, method="dhf",
                active_electrons=active_electrons,
                active_orbitals=active_orbitals
            )
            return adapt_energy_circuit(self.vqe_weights, H)

        energy_hartree = loss_function(coords_bohr)
        grad_coords = qml.grad(loss_function)(coords_bohr)

        # 8. Dipole moment calculation
        mol_obj = qml.qchem.Molecule(symbols, static_coords, charge=self.charge, mult=multiplicity)
        dipole_operators = qml.qchem.molecular_dipole(
            mol_obj, method="dhf",
            active_electrons=active_electrons,
            active_orbitals=active_orbitals
        )

        dipole_vec_au = np.zeros(3)
        for idx in range(3):
            @qml.qnode(dev, shots=self.shots)
            def eval_dipole_component(weights, op_component):
                qml.BasisState(hf_init, wires=range(qubits))
                for w, op in zip(weights, self.chosen_operators):
                    if len(op) == 2:
                        qml.SingleExcitation(w, wires=op)
                    else:
                        qml.DoubleExcitation(w, wires=op)
                return qml.expval(op_component)

            # FIX: Safely extract the real part before float casting
            res_val = eval_dipole_component(self.vqe_weights, dipole_operators[idx])
            dipole_vec_au[idx] = float(np.real(res_val))

        # 9. Convert units back to ASE standard
        self.results['energy'] = float(energy_hartree) * Hartree
        forces_ev_angstrom = -grad_coords * (Hartree * angstrom_to_bohr)
        self.results['forces'] = np.array(forces_ev_angstrom)

        a_u_to_Debye = (_e * Bohr) / Debye
        self.results['dipole'] = dipole_vec_au * a_u_to_Debye

