"""BeH2 (STO-3G, Be 1s frozen): harmonic frequencies with the PennyLane backend (analytic forces).

Paper: Results, BeH2 table (optional PennyLane row). The ADAPT operator selection is done once at the
first geometry and reused for all later geometries (use_ADAPT_memory=True), so the first force call
dominates the cost. 'lightning.qubit' speeds up the VQE optimization; the nuclear gradient is always
evaluated by backpropagation on 'default.qubit'.
Runtime: several hours on one core (12 qubits).
"""
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator

beh2 = Atoms("BeH2", positions=[[0, 0, 0], [1.3164, 0, 0], [-1.3164, 0.02, 0]])
beh2.calc = PennyLaneAdaptVQECalculator(basis="sto-3g", coreorb=1, backend="lightning.qubit")
BFGS(beh2, trajectory="beh2_pennylane_opt.traj").run(fmax=0.005)

vib = Vibrations(beh2, name="vib_beh2_pennylane", delta=0.01, nfree=2)
vib.run(); vib.summary()
freqs = np.sort(np.real(vib.get_frequencies()))[-4:]
print("BeH2 PennyLane frequencies (cm-1): bend, bend, sym. stretch, antisym. stretch:", np.round(freqs, 1))
