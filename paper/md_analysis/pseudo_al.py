"""Paper: Results, MD section (offline test of surrogate settings on exact trajectories).

Pseudo active learning: replay saved exact-force H2 trajectories (bond length per step) and decide,
frame by frame, whether the surrogate would have requested a VQE (= exact) calculation."""
import json, sys, time, numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase_quantum_vqe.surrogates import DistanceGPR
from runio import load_run
import oracle

def h2(r, labeled=False):
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, r]])
    if labeled:
        g = oracle.dEdr(r); a.calc = SinglePointCalculator(a, energy=oracle.energy(r), forces=np.array([[0, 0, g], [0, 0, -g]]))
    return a

def replay(rs, use_forces, eps_E, eps_F, n_init=1):
    data = [h2(r, True) for r in rs[:n_init]]; m = DistanceGPR(use_forces=use_forces); m.train(data)
    calls, fe, ee = n_init, [], []
    for r in rs[n_init:]:
        a = h2(r)
        sE = m.predict_uncertainty(a); sF = m.predict_uncertainty_forces(a).max()
        if sE > eps_E or sF > eps_F:
            data.append(h2(r, True)); m.train(data); calls += 1
        else:
            fe.append(abs(m.predict_forces(a)[1, 2] - (-oracle.dEdr(r)))); ee.append(abs(m.predict_energy(a) - oracle.energy(r)))
    return calls, max(fe) if fe else 0, np.mean(fe) if fe else 0, max(ee) if ee else 0

res = {}
for f in ["direct_1", "direct_2"]:
    rs = np.array(load_run(f)["r"])
    for label, uf, eE, eF in [("energy-only GP, eps_E=0.05 eV", False, 0.05, np.inf),
                              ("energy+force GP, eps_E=0.05 eV", True, 0.05, np.inf),
                              ("energy+force GP, eps_F=0.10 eV/A", True, 0.05, 0.10),
                              ("energy+force GP, eps_F=0.05 eV/A", True, 0.05, 0.05)]:
        t = time.time(); c, fmax, fmean, emax = replay(rs, uf, eE, eF)
        res[f"{f} | {label}"] = dict(calls=c, max_force_err=fmax, mean_force_err=fmean, max_energy_err=emax)
        print(f"{f:9s} {label:36s} calls={c:4d}/{len(rs)}  max|dF|={fmax:.3f}  mean|dF|={fmean:.4f} eV/A  max|dE|={emax:.4f} eV  ({time.time()-t:.0f}s)", flush=True)
json.dump(res, open("pseudo_al.json", "w"), indent=1)
