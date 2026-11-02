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

import warnings
from qiskit.primitives import Estimator as LocalEstimator      # exact statevector expectation values
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as RuntimeEstimator
from qiskit.primitives import BaseEstimatorV1, EstimatorResult
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class _EstimatorV1FromV2(BaseEstimatorV1):
    """Expose a Qiskit Runtime EstimatorV2 through the V1 interface used by qiskit-algorithms.

    Circuits are transpiled to the backend's instruction set and observables are mapped onto
    the resulting qubit layout before submission.
    """

    def __init__(self, estimator_v2, backend, optimization_level=1):
        super().__init__()
        self._est = estimator_v2
        self._pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)

    def _call(self, pubs):
        result = self._est.run(pubs).result()
        values = np.array([float(np.real(r.data.evs)) for r in result])
        return EstimatorResult(values, [{} for _ in values])

    def _run(self, circuits, observables, parameter_values, **run_options):
        pubs = []
        for circ, obs, params in zip(circuits, observables, parameter_values):
            isa = self._pm.run(circ)
            pubs.append((isa, obs.apply_layout(isa.layout), params))
        job = PrimitiveJob(self._call, pubs)
        job._submit()
        return job


def make_ibm_estimator(backend, shots=4000, resilience_level=0, optimization_level=1):
    """V1-compatible estimator for an IBM Quantum backend (or a fake backend for local testing)."""
    est = RuntimeEstimator(mode=backend, options={"default_shots": shots,
                                                  "resilience_level": resilience_level})
    return _EstimatorV1FromV2(est, backend, optimization_level=optimization_level)


class QiskitVQECalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'dipole']

    def __init__(self, basis='sto3g', optimizer=None, optimizer_name='SLSQP',
                 delta=1e-3, n_jobs=1, charge=0, spin=0, maxiter=200,
                 backend='statevector', shots=4000, resilience_level=0,
                 ibmq_token=None, estimator_override=None, nfree=2,
                 coreorb=0, vqe_eigenvalue=1e-07, seed=None,
                 ibm_channel='ibm_quantum_platform', ibm_instance=None, **kwargs):
        """
        basis:            Basis set (z.B. 'sto3g')
        optimizer:        Qiskit-Optimizer (directly loaded)
        optimizer_name:   Name of Qiskit-Optimizer ('SLSQP', 'COBYLA', 'L_BFGS_B')
        delta:            Finite-Difference displacement in Å
        n_jobs:           Parallization ofer cores
        charge:           Total charge of syste.
        spin:             2S = #alpha - #beta Electrons
        maxiter:          max. iterationens of VQE-optimizer
        backend:          'statevector' : exact expectation values (qiskit.primitives.Estimator), shots ignored
                          'aer_shots'   : finite-shot sampling with qiskit-aer (shots, seed used)
                          'aer'         : deprecated alias of 'statevector' (kept for old scripts)
                          any other     : name of an IBM Quantum backend (Qiskit Runtime)
        shots:            Measurements per measurement setting ('aer_shots' and IBM backends only)
        resilience_level: Error-Mitigation-Level (0–3)
        ibmq_token:       Optional IBM Quantum API token (IBM backends only)
        ibm_channel:      Qiskit Runtime channel ('ibm_quantum_platform' or 'ibm_cloud')
        ibm_instance:     Optional Qiskit Runtime instance (CRN or name)
        seed:             Seed for finite-shot sampling ('aer_shots')
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
        self.seed = seed
        if backend == 'aer':
            warnings.warn("backend='aer' computes exact expectation values and ignores 'shots'. "
                          "Use backend='statevector' (same behaviour) or backend='aer_shots' for "
                          "finite-shot sampling.", DeprecationWarning)
        self._local = backend in ('statevector', 'aer', 'aer_shots')

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
        if not self._local and estimator_override is None:
            kw = {"channel": ibm_channel}
            if ibmq_token:
                kw["token"] = ibmq_token
            if ibm_instance:
                kw["instance"] = ibm_instance
            self.service = QiskitRuntimeService(**kw)

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
        elif self.backend in ('statevector', 'aer'):
            estimator = LocalEstimator()
        elif self.backend == 'aer_shots':
            from qiskit_aer.primitives import Estimator as AerEstimator
            run_options = {"shots": self.shots}
            if self.seed is not None:
                run_options["seed"] = self.seed
            estimator = AerEstimator(run_options=run_options, approximation=False)
        else:
            backend_instance = self.service.backend(self.backend)
            estimator = make_ibm_estimator(backend_instance, shots=self.shots,
                                           resilience_level=self.resilience_level)

#        vqea = VQE(estimator, full_ansatz, optimizer=self.optimizer)
        vqea = VQE(estimator, uccsd_ansatz, optimizer=self.optimizer)
        vqea.initial_point=np.zeros(uccsd_ansatz.num_parameters)
        vqe=AdaptVQE(vqea,gradient_threshold=1e-05, eigenvalue_threshold=self.vqe_eigenvalue)
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
                raise ValueError(f"nfree={self.nfree} is not supported (use 2 or 4)")
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
                        forces[i_atom, j_coord] = -(-e_plus2 + 8 * e_plus1 - 8 * e_minus1 + e_minus2) / (12 * delta)

            self.results['forces'] = forces
