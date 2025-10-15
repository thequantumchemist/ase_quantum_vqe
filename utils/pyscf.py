import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree
from joblib import Parallel, delayed

from pyscf import gto, scf, cc

class PySCFCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, basis='sto-3g', method='ccsd',
                 charge=0, spin=0,
                 delta=1e-3, n_jobs=1, coreorb=0, **kwargs):
        """
        ASE calculator using PySCF.

        Parameters
        ----------
        basis : str
            Gaussian basis set name, e.g. 'sto-3g'
        method : str
            'hf', 'ccsd', or 'fci'
        charge : int
            Total charge of the system (electrons removed are positive)
        spin : int
            2S, the difference (#alpha - #beta electrons).
        delta : float
            Displacement step for finite-difference forces (Å)
        n_jobs : int
            Number of parallel jobs for forces (via joblib)
        coreorb : int
            Number of lowest-lying doubly occupied orbitals to freeze (frozen core)
        """
        super().__init__(**kwargs)
        self.basis = basis
        self.method = method.lower()
        self.charge = charge
        self.spin = spin
        self.delta = delta
        self.n_jobs = n_jobs
        self.coreorb = coreorb

    def _energy_from_atoms(self, atoms):
        """Return total energy in eV from PySCF calculation."""
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        # Build PySCF molecule
        mol = gto.Mole()
        mol.unit = 'Angstrom'
        mol.basis = self.basis
        mol.charge = self.charge
        mol.spin = self.spin
        mol.atom = ";".join(
            f"{sym} {pos[0]} {pos[1]} {pos[2]}" for sym, pos in zip(symbols, positions)
        )
        mol.build()

        # Hartree–Fock
        mf = scf.RHF(mol).run()

        # Select correlated method
        if self.method == 'hf':
            e_tot = mf.e_tot

        elif self.method == 'ccsd':
            frozen_list = list(range(self.coreorb)) if self.coreorb > 0 else None
            ccsd_obj = cc.CCSD(mf, frozen=frozen_list)
            ccsd_val = ccsd_obj.run()
            e_tot = ccsd_val.e_tot

        elif self.method == 'fci':
            from pyscf import fci
            cisolver = fci.FCI(mf)
            if self.coreorb > 0:
                cisolver.frozen = list(range(self.coreorb))
            e_tot = mf.energy_nuc() + cisolver.kernel()[0]

        else:
            raise ValueError(f"Unknown method {self.method}")

        return e_tot * Hartree

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Energy calculation
        energy_ev = self._energy_from_atoms(atoms)
        self.results['energy'] = energy_ev

        # Numerical forces if requested
        if 'forces' in properties:
            positions = atoms.get_positions()
            delta = self.delta

            # list of displacements
            tasks = []
            for i_atom in range(len(positions)):
                for j_coord in range(3):
                    tasks.append((i_atom, j_coord, +delta))
                    tasks.append((i_atom, j_coord, -delta))

            def disp_energy(i_atom, j_coord, d):
                displaced = atoms.copy()
                displaced.positions[i_atom, j_coord] += d
                return self._energy_from_atoms(displaced)

            energies = Parallel(n_jobs=self.n_jobs)(
                delayed(disp_energy)(i, j, d) for (i, j, d) in tasks
            )

            forces = np.zeros_like(positions)
            idx = 0
            for i_atom in range(len(positions)):
                for j_coord in range(3):
                    e_plus = energies[idx]
                    e_minus = energies[idx + 1]
                    idx += 2
                    forces[i_atom, j_coord] = -(e_plus - e_minus) / (2 * delta)

            self.results['forces'] = forces
