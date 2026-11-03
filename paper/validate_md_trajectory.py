"""Validate an MD trajectory against exact forces: FCI (or CCSD) in the same frozen-core active space.

Works for any ASE trajectory whose frames carry the energies and forces used by the MD, e.g. the
published BeH2 run (MD.traj) or beh2_distance_md.traj. Prints force errors and temperatures and writes
<name>_validation.json.   Usage:  python validate_md_trajectory.py MD.traj --coreorb 1 [--every 5]
"""
import argparse, json
import numpy as np
from ase.io import read
from ase_quantum_vqe.utils.pyscf import PySCFCalculator

p = argparse.ArgumentParser()
p.add_argument("traj"); p.add_argument("--coreorb", type=int, default=0)
p.add_argument("--method", default="fci", choices=["fci", "ccsd"]); p.add_argument("--every", type=int, default=1)
p.add_argument("--charge", type=int, default=0)
a = p.parse_args()

frames = read(f"{a.traj}@::{a.every}")
ref = PySCFCalculator(method=a.method, coreorb=a.coreorb, charge=a.charge, forces="analytic")
dF, dE, T = [], [], []
for at in frames:
    f_md, e_md = at.get_forces(), at.get_potential_energy()
    T.append(float(at.get_temperature()) if at.get_momenta() is not None else float("nan"))
    b = at.copy(); b.calc = ref
    dF.append(float(np.abs(f_md - b.get_forces()).max())); dE.append(float(e_md - b.get_potential_energy()))
dF, dE = np.array(dF), np.array(dE)
print(f"{len(frames)} frames: max |dF| = {dF.max():.3f} eV/A, mean max|dF| = {dF.mean():.4f} eV/A, "
      f"max |dE| = {np.abs(dE).max():.3f} eV, mean T = {np.nanmean(T):.0f} K")
json.dump({"max_force_error_per_frame": dF.tolist(), "energy_error": dE.tolist(), "T": T},
          open(a.traj.rsplit(".", 1)[0] + "_validation.json", "w"))
