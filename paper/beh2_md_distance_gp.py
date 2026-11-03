"""BeH2 Langevin MD at 300 K with VQE forces and FALCON, using the force-trained distance GP.

Follow-up to the published BeH2 run (fingerprint GPR, energy criterion, 160 VQE calculations in 320 fs).
Same start structure and thermostat; the surrogate is the distance GP with energy and force thresholds.
Writes beh2_distance_md.traj (structures with the energies and forces used by the MD),
beh2_distance_training.traj (all VQE calculations), and beh2_distance_md_log.json (per-step log).
Validate afterwards with:  python validate_md_trajectory.py beh2_distance_md.traj --coreorb 1
"""
import json
import numpy as np
from ase import Atoms, units
from ase.io import read, Trajectory
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from falcon_md.otf_calculator import FALCON
from falcon_md.utils.langevin import Langevin
from ase_quantum_vqe.surrogates import DistanceGPR

# ---------------- settings ----------------
BACKEND = "qiskit"            # "qiskit" (finite differences, n_jobs cores) or "pennylane" (analytic forces)
N_JOBS = 18
T, DT_FS, FRICTION, STEPS = 300, 1.0, 0.002, 10000
ACC_E, ACC_F = 0.05, 0.1      # eV and eV/A; the force threshold is essential
SEED = 1

if BACKEND == "qiskit":
    from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
    vqe = QiskitVQECalculator(basis="sto3g", backend="statevector", n_jobs=N_JOBS,
                              coreorb=1, vqe_eigenvalue=1e-5, maxiter=150)
else:
    from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator
    vqe = PennyLaneAdaptVQECalculator(basis="sto-3g", coreorb=1, backend="lightning.qubit")

NCALLS = [0]


class Counting(Calculator):
    """Counts VQE calls (module-level counter, because FALCON copies the calculator)."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, calc):
        super().__init__(); self.calc = calc

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        NCALLS[0] += 1
        a = atoms.copy(); a.calc = self.calc
        self.results["energy"] = a.get_potential_energy(); self.results["forces"] = a.get_forces()


ref = Counting(vqe)

# ---------------- geometry optimization (initial training data) ----------------
beh2 = Atoms("BeH2", positions=[[1.4, 0, 0], [0.0, 0.0, 0.0], [2.1, 1.0, 2.0]])
beh2.center(vacuum=25.0)
beh2.calc = ref
BFGSLineSearch(beh2, trajectory="beh2_distance_opt.traj", maxstep=0.1).run(fmax=0.005)
n_opt = NCALLS[0]
MaxwellBoltzmannDistribution(beh2, temperature_K=T, rng=np.random.default_rng(SEED))
Stationary(beh2)
np.random.seed(SEED)

# ---------------- FALCON with the distance GP ----------------
beh2.calc = FALCON(model=DistanceGPR(use_forces=True), calc=ref,
                   training_data=read("beh2_distance_opt.traj@:"),
                   train_start=2, train_log=False,
                   accuracy_e=ACC_E, accuracy_f=ACC_F,
                   td_filename="beh2_distance_training.traj")
dyn = Langevin(beh2, DT_FS * units.fs, temperature_K=T, friction=FRICTION)
traj = Trajectory("beh2_distance_md.traj", "w", beh2)
dyn.attach(traj.write, interval=1)

log = {"t_fs": [], "epot": [], "tkin": [], "vqe_calls": [], "n_opt": n_opt,
       "settings": dict(backend=BACKEND, T=T, dt_fs=DT_FS, friction=FRICTION, accuracy_e=ACC_E, accuracy_f=ACC_F)}


def record():
    log["t_fs"].append(dyn.nsteps * DT_FS)
    log["epot"].append(float(beh2.get_potential_energy()))
    log["tkin"].append(float(beh2.get_temperature()))
    log["vqe_calls"].append(NCALLS[0])
    if dyn.nsteps % 100 == 0:
        json.dump(log, open("beh2_distance_md_log.json", "w"))
        print(f"step {dyn.nsteps}: VQE calculations so far {NCALLS[0]} (optimization {n_opt}), "
              f"T = {np.mean(log['tkin'][-100:]):.0f} K", flush=True)


dyn.attach(record, interval=1)
dyn.run(STEPS)
json.dump(log, open("beh2_distance_md_log.json", "w"))
print(f"done: {STEPS * DT_FS:.0f} fs, {NCALLS[0]} VQE calculations in total ({n_opt} in the optimization)")
