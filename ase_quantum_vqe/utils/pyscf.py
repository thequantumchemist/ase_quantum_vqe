import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Bohr
from joblib import Parallel, delayed

from pyscf import gto, scf, cc, mcscf


class PySCFCalculator(Calculator):
    """Classical reference calculator based on PySCF (HF, CCSD, or FCI).

    Used to produce reference energies and forces in exactly the same basis and frozen-core
    active space as the VQE calculators.

    Parameters
    ----------
    basis : str
        Gaussian basis set, e.g. 'sto-3g'.
    method : str
        'hf', 'ccsd', or 'fci'. With coreorb > 0, 'fci' is FCI in the frozen-core active space
        (CASCI with all non-core orbitals active), i.e. the exact reference for VQE in that space.
    charge : int
        Total charge.
    spin : int
        2S = N_alpha - N_beta. Open shells use ROHF orbitals.
    coreorb : int
        Number of lowest doubly occupied orbitals to freeze.
    forces : str
        'fd' (central finite differences with step `delta`, parallelized with joblib) or
        'analytic' (PySCF nuclear gradients).
    delta : float
        Finite-difference step in Angstrom (forces='fd').
    n_jobs : int
        Number of parallel jobs for finite-difference forces.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, basis='sto-3g', method='ccsd', charge=0, spin=0,
                 delta=1e-3, n_jobs=1, coreorb=0, forces='fd', **kwargs):
        super().__init__(**kwargs)
        self.basis = basis
        self.method = method.lower()
        if self.method not in ('hf', 'ccsd', 'fci'):
            raise ValueError(f"Unknown method {method}")
        self.charge = charge
        self.spin = spin
        self.delta = delta
        self.n_jobs = n_jobs
        self.coreorb = coreorb
        if forces not in ('fd', 'analytic'):
            raise ValueError("forces must be 'fd' or 'analytic'")
        self.force_mode = forces

    def _solve(self, atoms):
        """Run the electronic-structure calculation; returns the PySCF solver object."""
        mol = gto.M(atom=[(s, p) for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
                    basis=self.basis, charge=self.charge, spin=self.spin, unit='Angstrom', verbose=0)
        mf = (scf.RHF(mol) if self.spin == 0 else scf.ROHF(mol)).run(conv_tol=1e-10)
        if self.method == 'hf':
            return mf
        if self.method == 'ccsd':
            solver = cc.CCSD(mf, frozen=list(range(self.coreorb)) if self.coreorb > 0 else None)
            solver.conv_tol = 1e-10
            solver.kernel()
            return solver
        # FCI in the frozen-core active space (all non-core orbitals active)
        ncas = mol.nao - self.coreorb
        nelecas = mol.nelectron - 2 * self.coreorb
        solver = mcscf.CASCI(mf, ncas, nelecas)
        solver.fcisolver.conv_tol = 1e-12
        solver.kernel()
        return solver

    def _energy_from_atoms(self, atoms):
        """Total energy in eV (includes nuclear repulsion and frozen-core energy)."""
        return self._solve(atoms).e_tot * Hartree

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if 'forces' in properties and self.force_mode == 'analytic':
            solver = self._solve(atoms)
            self.results['energy'] = solver.e_tot * Hartree
            grad = solver.nuc_grad_method().kernel()                 # Hartree / Bohr
            self.results['forces'] = -np.asarray(grad) * Hartree / Bohr
            return

        self.results['energy'] = self._energy_from_atoms(atoms)

        if 'forces' in properties:
            positions = atoms.get_positions()
            delta = self.delta
            tasks = [(i, j, d) for i in range(len(positions)) for j in range(3) for d in (+delta, -delta)]

            def disp_energy(i_atom, j_coord, d):
                displaced = atoms.copy()
                displaced.positions[i_atom, j_coord] += d
                return self._energy_from_atoms(displaced)

            energies = Parallel(n_jobs=self.n_jobs)(delayed(disp_energy)(i, j, d) for (i, j, d) in tasks)
            forces = np.zeros_like(positions)
            for k, (i_atom, j_coord, _) in enumerate(tasks[::2]):
                forces[i_atom, j_coord] = -(energies[2 * k] - energies[2 * k + 1]) / (2 * delta)
            self.results['forces'] = forces
