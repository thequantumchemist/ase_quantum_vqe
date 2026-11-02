"""Exact H2/STO-3G (FCI) energy and bond force from a dense PySCF grid + spline (accurate to <1e-6)."""
import numpy as np, os, json
from scipy.interpolate import CubicSpline
from ase.units import Hartree, Bohr
from pyscf import gto, scf, ci
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h2_fci_grid.json")
if not os.path.exists(F):
    rs = np.linspace(0.35, 1.6, 251); E = []; G = []
    for r in rs:
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12); m = ci.CISD(mf).run(conv_tol=1e-12)
        E.append(m.e_tot * Hartree); G.append(m.nuc_grad_method().kernel()[1, 2] * Hartree / Bohr)
    json.dump({"r": rs.tolist(), "E": E, "dEdr": G}, open(F, "w"))
g = json.load(open(F)); _E = CubicSpline(g["r"], g["E"]); _G = CubicSpline(g["r"], g["dEdr"])
def energy(r): return float(_E(r))
def dEdr(r): return float(_G(r))
