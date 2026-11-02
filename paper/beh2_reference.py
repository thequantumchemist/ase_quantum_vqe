# Paper: Results, BeH2 table (FCI and CCSD rows). Runtime: < 1 min.
"""BeH2 STO-3G, one frozen core orbital: FCI (= CASCI(4e,6o)) and CCSD optimized geometry and harmonic frequencies."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from pyscf import gto, scf, mcscf, cc

class Ref(Calculator):
    implemented_properties = ["energy", "forces"]
    def __init__(self, method): super().__init__(); self.method = method
    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        mol = gto.M(atom=[(s, p) for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)], basis="sto-3g", unit="Angstrom", verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        if self.method == "fci":
            m = mcscf.CASCI(mf, mol.nao - 1, mol.nelectron - 2); m.fcisolver.conv_tol = 1e-12; m.kernel(); e = m.e_tot
        else:
            m = cc.CCSD(mf, frozen=[0]); m.conv_tol = 1e-12; m.kernel(); e = m.e_tot
        g = m.nuc_grad_method().kernel()
        self.results["energy"] = e * units.Hartree; self.results["forces"] = -g * units.Hartree / units.Bohr

for method in ["ccsd", "fci"]:
    a = Atoms("BeH2", positions=[[0, 0, 0], [1.33, 0, 0], [-1.33, 0.05, 0]]); a.calc = Ref(method)
    BFGS(a, logfile=None).run(fmax=1e-4, steps=200)
    vib = Vibrations(a, name=f"vib_{method}", delta=0.01, nfree=2); vib.clean(); vib.run()
    e = vib.get_frequencies()
    real = np.sort(np.real(e[np.abs(np.imag(e)) < 1e-6]))[-4:]
    print(f"{method.upper():5s} r(Be-H) = {a.get_distance(0,1):.4f} A, angle = {a.get_angle(1,0,2):.2f} deg, E = {a.get_potential_energy():.4f} eV; "
          f"frequencies (cm-1): {np.round(real, 1)}", flush=True)
