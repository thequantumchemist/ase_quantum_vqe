import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Debye, _e, Bohr
from joblib import Parallel, delayed

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD,UCC
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from qiskit_algorithms.minimum_eigensolvers import VQE,AdaptVQE
from qiskit_algorithms.optimizers import SLSQP, COBYLA, L_BFGS_B

from qiskit.primitives import Estimator as LocalEstimator
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as RuntimeEstimator

class QiskitVQECalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'dipole']

    def __init__(self, basis='sto3g', optimizer=None, optimizer_name='SLSQP',
                 delta=1e-3, n_jobs=1, charge=0, spin=0, maxiter=200,
                 backend='aer', shots=4000, resilience_level=0,
                 ibmq_token=None, estimator_override=None, nfree=2,
                 coreorb=0,vqe_eigenvalue=1e-07, **kwargs):
        """
        basis:            Basis set (z.B. 'sto3g')
        optimizer:        Qiskit-Optimizer (directly loaded)
        optimizer_name:   Name of Qiskit-Optimizer ('SLSQP', 'COBYLA', 'L_BFGS_B')
        delta:            Finite-Difference displacement in Å
        n_jobs:           Parallization ofer cores
        charge:           Total charge of syste.
        spin:             2S = #alpha - #beta Electrons
        maxiter:          max. iterationens of VQE-optimizer
        backend:          'aer' or IBMQ Backend-Name
        shots:            Measurments per QPU-Analizes
        resilience_level: Error-Mitigation-Level (0–3)
        ibmq_token:       Optional: IBMQ API-Token (if backend != 'aer')
        estimator_override: Optional: Exteral Estimator (e.g. keep fixed if running on a real device)
        nfree:            finite displacements per direction per atom for numerical forces (2 or 4)
        coreorb:          number of core orbitals
        vqe_eigenvalue:   Eigenvalue of UCCSD Ansatz for ADAPT VQE
        """
        super().__init__(**kwargs)
        self.basis = basis
        self.delta = delta
        self.n_jobs = n_jobs
        self.charge = charge
        self.spin = spin
        self.backend = backend
        self.shots = shots
        self.resilience_level = resilience_level
        self.ibmq_token = ibmq_token
        self.estimator_override = estimator_override
        self.nfree = nfree
        self.coreorb = coreorb
        self.vqe_eigenvalue = vqe_eigenvalue

        # Optimizer-Auswahl
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            if optimizer_name.upper() == 'SLSQP':
                self.optimizer = SLSQP(maxiter=maxiter)
            elif optimizer_name.upper() == 'COBYLA':
                self.optimizer = COBYLA(maxiter=maxiter)
            elif optimizer_name.upper() == 'L_BFGS_B':
                self.optimizer = L_BFGS_B(maxiter=maxiter)
            else:
                raise ValueError(f"Unbekannter optimizer_name: {optimizer_name}")

        self.service = None
        if backend != 'aer' and estimator_override is None:
            if ibmq_token:
                self.service = QiskitRuntimeService(channel="ibm_quantum", token=ibmq_token)
            else:
                self.service = QiskitRuntimeService(channel="ibm_quantum")

    def _single_point(self, atoms):
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        atom_strs = [
            f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}"
            for sym, pos in zip(symbols, positions)
        ]

        driver = PySCFDriver(
            atom=atom_strs,
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
            unit=DistanceUnit.ANGSTROM
        )
        problem = driver.run()
        mapper = JordanWignerMapper()

        # freze core orbitals (Active Space)
        if self.coreorb > 0:
            n_orb = problem.num_spatial_orbitals
            n_elec = sum(problem.num_particles)
            n_active_orb = n_orb - self.coreorb
            n_active_elec = n_elec - 2 * self.coreorb  # 2 e- per Core-Orbital
            transformer = ActiveSpaceTransformer(
                num_electrons=n_active_elec,
                num_spatial_orbitals=n_active_orb
            )
            problem = transformer.transform(problem)

        # UCCSD-Ansatz
        hf_init = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
        uccsd_ansatz = UCC(problem.num_spatial_orbitals, problem.num_particles, 'sd',mapper, alpha_spin=True, beta_spin=True, max_spin_excitation=None,initial_state=hf_init)

        # Estimator
        if self.estimator_override is not None:
            estimator = self.estimator_override
        elif self.backend == 'aer':
            estimator = LocalEstimator()
        else:
            backend_instance = self.service.backend(self.backend)
            estimator = RuntimeEstimator(
                backend=backend_instance,
                options={"shots": self.shots, "resilience_level": self.resilience_level}
            )

#        vqea = VQE(estimator, full_ansatz, optimizer=self.optimizer)
        vqea = VQE(estimator, uccsd_ansatz, optimizer=self.optimizer)
        vqea.initial_point=np.zeros(uccsd_ansatz.num_parameters)
        vqe=AdaptVQE(vqea,gradient_threshold=1e-05, eigenvalue_threshold=self.vqe_eigenvalue)
        vqe.supports_aux_opperators= lambda: True
        solver = GroundStateEigensolver(mapper, vqe)
        result = solver.solve(problem)

        # Energie [eV]
        energy_ev = result.total_energies[0].real * Hartree

        # Dipolmoment [Debye]
        mux_au = float(result.dipole_moment[0][0])
        muy_au = float(result.dipole_moment[0][1])
        muz_au = float(result.dipole_moment[0][2])
        a_u_to_Debye = (_e * Bohr) / Debye
        dipole_vec = np.array([mux_au, muy_au, muz_au]) * a_u_to_Debye

        return energy_ev, dipole_vec

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy_ev, dipole_vec = self._single_point(atoms)
        self.results['energy'] = energy_ev
        self.results['dipole'] = dipole_vec

        if 'forces' in properties:
            positions = atoms.get_positions()
            delta = self.delta
            tasks = []
            if self.nfree == 2:
                displacements = [+delta, -delta]
            elif self.nfree == 4:
                displacements = [+delta, -delta, +2 * delta, -2 * delta]
            else:
                raise ValueError(f"nfree={self.nfree} wird nicht unterstützt (nur 2 oder 4 erlaubt)")
            for i_atom in range(len(positions)):
                for j_coord in range(3):
                    for d in displacements:
                        tasks.append((i_atom, j_coord, d))

            def displaced_energy(i_atom, j_coord, d):
                displaced = atoms.copy()
                displaced.positions[i_atom, j_coord] += d
                e_ev, _ = self._single_point(displaced)
                return e_ev

            energies = Parallel(n_jobs=self.n_jobs)(
                delayed(displaced_energy)(i, j, d) for (i, j, d) in tasks
            )

            forces = np.zeros_like(positions)
            idx = 0
            n_disp = len(displacements)
            for i_atom in range(len(positions)):
                for j_coord in range(3):
                    disp_energies = energies[idx:idx + n_disp]
                    idx += n_disp
                    if self.nfree == 2:
                        e_plus, e_minus = disp_energies
                        forces[i_atom, j_coord] = -(e_plus - e_minus) / (2 * delta)
                    elif self.nfree == 4:
                        e_plus1, e_minus1, e_plus2, e_minus2 = disp_energies
                        forces[i_atom, j_coord] = (-e_plus2 + 8 * e_plus1 - 8 * e_minus1 + e_minus2) / (12 * delta)

            self.results['forces'] = forces
