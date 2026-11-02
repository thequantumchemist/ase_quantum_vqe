"""
Tutorial 7: optimization, vibrations and JEDI strain analysis of H3+ with analytic VQE forces (~2 min).

    pip install "ase_quantum_vqe[pennylane,jedi]"
    python 07_workflows.py
"""
# --8<-- [start:imports]
from importlib.metadata import version

from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from strainjedi.jedi import Jedi

from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator
# --8<-- [end:imports]

# --8<-- [start:optimise]
h3 = Atoms("H3", positions=[[0, 0, 0], [0.95, 0, 0], [0.45, 0.85, 0]])
h3.center(vacuum=10.0)                         # a box, so that JEDI can handle the geometry
calc = PennyLaneAdaptVQECalculator(charge=1)
h3.calc = calc
BFGS(h3, trajectory="h3_opt.traj").run(fmax=0.005)
# --8<-- [end:optimise]

# --8<-- [start:vibrations]
vib = Vibrations(h3, name="vib_h3", delta=0.01, nfree=2)
vib.clean()
vib.run()
vib.summary()
hessian = vib.get_vibrations()
# --8<-- [end:vibrations]

# --8<-- [start:jedi]
strained = h3.copy()
strained.positions[1, 0] += 0.1                # stretch one bond by 0.1 A
strained.calc = calc
strained.get_potential_energy()

jedi = Jedi(h3, strained, hessian)
# The H-H bonds of H3+ (~1 A) exceed JEDI's default covalent cutoff. strainjedi 1.0 finds them with a
# larger covalent factor; strainjedi 1.1 ignores that factor and needs the bonds explicitly.
if tuple(int(x) for x in version("strainjedi").split(".")[:2]) >= (1, 1):
    jedi.add_custom_bonds([[0, 1], [0, 2], [1, 2]])
else:
    jedi.set_bond_params(covf=2.0, vdwf=0.9)
jedi.run()
# --8<-- [end:jedi]
