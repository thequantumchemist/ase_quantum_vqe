"""
Tutorial 5: molecular dynamics with VQE forces and FALCON active learning (~30 s).

    pip install "ase_quantum_vqe[pennylane,falcon]"
    python 05_active_learning_md.py
"""
# --8<-- [start:imports]
import numpy as np
from ase import Atoms, units
from ase.io import read
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from falcon_md.otf_calculator import FALCON
from falcon_md.utils.langevin import Langevin

from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator
from ase_quantum_vqe.surrogates import DistanceGPR
# --8<-- [end:imports]

# --8<-- [start:setup]
np.random.seed(2)
h2 = Atoms("H2", positions=[[0.7, 0, 0], [0, 0, 0]])
h2.center(vacuum=25.0)
vqe = PennyLaneAdaptVQECalculator()

# the frames of a short VQE geometry optimization are the initial training data
h2.calc = vqe
BFGS(h2, trajectory="h2_opt.traj", logfile=None).run(fmax=0.05)
MaxwellBoltzmannDistribution(h2, temperature_K=600, rng=np.random.default_rng(2))
Stationary(h2)
# --8<-- [end:setup]

# --8<-- [start:falcon]
h2.calc = FALCON(
    model=DistanceGPR(use_forces=True),   # trained on energies and forces
    calc=vqe,                             # VQE is called only when the surrogate is uncertain
    training_data=read("h2_opt.traj@:"),
    train_start=2, train_log=False,
    accuracy_e=0.05,                      # eV
    accuracy_f=0.1,                       # eV/A: the force criterion is essential
    td_filename="training.traj",          # every VQE calculation is written here
)
# --8<-- [end:falcon]

# --8<-- [start:run]
dyn = Langevin(h2, 0.5 * units.fs, temperature_K=600, friction=0.002)
r = []
dyn.attach(lambda: r.append(h2.get_distance(0, 1)), interval=1)
dyn.run(2000)                             # 1 ps
n_vqe = len(read("training.traj@:"))
print(f"1 ps of MD: H-H distance {np.mean(r):.3f} +- {np.std(r):.3f} A; "
      f"training set: {n_vqe} VQE calculations in total")
# --8<-- [end:run]
