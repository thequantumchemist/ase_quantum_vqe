# Paper: Table 6 (geometry optimization with sampled analytic forces). Runtime: H2 ~5 min, H3+ ~2 min per run.
"""Geometry optimization with sampled analytic VQE forces (PennyLane backend, force_shots).

Circuit parameters are converged exactly at every step; only the forces are sampled. For each shot
level and seed, BFGS (ASE, maxstep 0.1 A) runs from a distorted start for at most MAXSTEPS steps or until
fmax < 0.05 eV/A (as judged by the noisy forces). Reported: final bond-length error relative to the
exact (noise-free) optimum, and whether the noisy convergence criterion was met.
Usage: SYS=H2|H3+ python noisy_opt.py   -> noisy_opt_<SYS>.json
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as onp
from ase import Atoms
from ase.optimize import BFGS
from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator

SYS = os.environ.get("SYS", "H2")
SHOTS = [int(float(x)) for x in os.environ.get("SHOTS", "1e3,1e4,1e5").split(",")]
SEEDS = int(os.environ.get("SEEDS", 5))
MAXSTEPS = int(os.environ.get("MAXSTEPS", 30))
FMAX = 0.05

if SYS == "H2":
    start = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.90]]); q = 0
else:
    start = Atoms("H3", positions=[[0, 0, 0], [0.9, 0, 0], [0.4, 0.8, 0]]); q = 1


def bonds(a):
    d = a.get_all_distances()
    return onp.sort(d[onp.triu_indices(len(a), 1)])


# exact optimum with noise-free analytic forces
ref = start.copy(); ref.calc = PennyLaneAdaptVQECalculator(charge=q)
BFGS(ref, logfile=None).run(fmax=1e-4, steps=200)
b_ref = bonds(ref)
out = {"system": SYS, "ref_bonds": b_ref.tolist(), "fmax": FMAX, "maxsteps": MAXSTEPS, "runs": []}
print(SYS, "reference bonds (A):", onp.round(b_ref, 5), flush=True)

for shots in SHOTS:
    for seed in range(SEEDS):
        t = time.time()
        a = start.copy()
        a.calc = PennyLaneAdaptVQECalculator(charge=q, force_shots=(shots or None), seed=seed)
        opt = BFGS(a, maxstep=0.1, logfile=None)
        converged = opt.run(fmax=FMAX, steps=MAXSTEPS)
        err = float(onp.abs(bonds(a) - b_ref).max())
        out["runs"].append(dict(shots=shots, seed=seed, converged=bool(converged), steps=opt.nsteps,
                                max_bond_error=err))
        print(f"{SYS} shots={shots:>7} seed={seed}: converged={converged} steps={opt.nsteps:2d} "
              f"max bond error={err:.4f} A  ({time.time() - t:.0f}s)", flush=True)
        json.dump(out, open(f"noisy_opt_{SYS.replace('+', 'p')}.json", "w"), indent=1)
print("done", flush=True)
