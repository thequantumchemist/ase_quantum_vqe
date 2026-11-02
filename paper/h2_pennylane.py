"""H2 (STO-3G) with the PennyLane backend and analytic forces: geometry, energy, frequency.

Paper: Results, H2 validation ("The PennyLane backend with analytic forces gives 0.735 A,
-30.948 eV, and 5001 cm-1").  Runtime: < 1 min.
"""
from ase.build import molecule
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator

h2 = molecule("H2")
h2.set_distance(0, 1, 0.90)
h2.calc = PennyLaneAdaptVQECalculator(basis="sto-3g")
BFGS(h2, trajectory="h2_pennylane.traj").run(fmax=0.005)

vib = Vibrations(h2, name="vib_h2_pennylane", delta=0.01, nfree=4)
vib.clean(); vib.run(); vib.summary()
print(f"r_e = {h2.get_distance(0, 1):.4f} A, E = {h2.get_potential_energy():.4f} eV, "
      f"omega = {vib.get_frequencies()[-1].real:.0f} cm-1")
