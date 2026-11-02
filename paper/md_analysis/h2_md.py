"""H2 Langevin MD at 600 K with FALCON active learning or exact forces, with full diagnostics.

Paper: Figure 4 and Table 7 (see run_md_table.sh for the exact commands).

MODE=falcon : FALCON on-the-fly learning, reference = REF (pennylane | qiskit | fci)
MODE=direct : Langevin MD driven directly by the exact FCI forces (reference trajectory)
Env: MODE, REF, SEED, STEPS (default 5000), OUT
Per step: time (fs), predicted Epot, exact FCI Epot (computed afterwards), r(H-H), T_kin, reference call flag.
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from pyscf import gto, scf, ci

MODE = os.environ.get("MODE", "falcon"); REF = os.environ.get("REF", "pennylane")
SEED = int(os.environ.get("SEED", 1)); STEPS = int(os.environ.get("STEPS", 5000))
OUT = os.environ.get("OUT", f"md_{MODE}_{REF}_{SEED}")
T, DT, FRICTION = 600, 0.5, 0.002
EPS = float(os.environ.get("ACC_E", 0.05))
ACC_F = float(os.environ.get("ACC_F", "inf"))
MODEL = os.environ.get("MODEL", "agox")
TRAIN_START = int(os.environ.get("TRAIN_START", 50))
TRAIN_LOG = os.environ.get("TRAIN_LOG", "1") == "1"


class FCICalc(Calculator):
    """Exact (FCI = CISD for two electrons) STO-3G energy and analytic forces from PySCF."""
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        mol = gto.M(atom=[(s, p) for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)],
                    basis="sto-3g", unit="Angstrom", verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-11)
        mc = ci.CISD(mf).run(conv_tol=1e-10)
        g = mc.nuc_grad_method().kernel()
        self.results["energy"] = mc.e_tot * units.Hartree
        self.results["forces"] = -g * units.Hartree / units.Bohr


NCALLS = [0]   # module-level: FALCON may copy the calculator object


class Counting(Calculator):
    """Wraps the reference calculator and counts calls."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, calc):
        super().__init__(); self.calc = calc

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        NCALLS[0] += 1
        a = atoms.copy(); a.calc = self.calc
        self.results["energy"] = a.get_potential_energy(); self.results["forces"] = a.get_forces()


class OracleCalc(Calculator):
    """Exact FCI H2 energy/forces from the PySCF spline (oracle.py); identical to FCICalc to < 1e-6."""
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        import oracle
        v = atoms.positions[1] - atoms.positions[0]; r = np.linalg.norm(v); u = v / r; g = oracle.dEdr(r)
        self.results["energy"] = oracle.energy(r); self.results["forces"] = np.array([g * u, -g * u])


def reference():
    if REF == "pennylane":
        from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator
        return PennyLaneAdaptVQECalculator()
    if REF == "qiskit":
        from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
        return QiskitVQECalculator(backend="statevector", n_jobs=1, vqe_eigenvalue=1e-7)
    return FCICalc()


np.random.seed(SEED)
h2 = Atoms("H2", positions=[[0.7, 0, 0], [0, 0, 0]]); h2.center(vacuum=25.0)
ref = Counting(reference())
h2.calc = ref
BFGSLineSearch(h2, trajectory=f"{OUT}_opt.traj", maxstep=0.1, logfile=None).run(fmax=0.05, steps=20)
MaxwellBoltzmannDistribution(h2, temperature_K=T, rng=np.random.default_rng(SEED)); Stationary(h2)

if MODE == "falcon":
    from falcon_md.otf_calculator import FALCON
    from falcon_md.models.agox_models import GPR
    from falcon_md.utils.langevin import Langevin
    NCALLS[0] = 0
    if MODEL == "distance":
        from ase_quantum_vqe.surrogates import DistanceGPR
        model = DistanceGPR(use_forces=True)
    else:
        model = GPR(h2)
    h2.calc = FALCON(model=model, calc=ref, train_start=TRAIN_START, train_log=TRAIN_LOG, training_data=read(f"{OUT}_opt.traj@0:"),
                     accuracy_e=EPS, accuracy_f=ACC_F, td_filename=f"{OUT}_training.traj")
else:
    if os.environ.get("THERMO") == "falcon":
        from falcon_md.utils.langevin import Langevin
    else:
        from ase.md.langevin import Langevin
    h2.calc = OracleCalc() if REF == "oracle" else FCICalc()

dyn = Langevin(h2, DT * units.fs, temperature_K=T, friction=FRICTION) if (MODE == "falcon" or os.environ.get("THERMO") == "falcon") else \
      Langevin(h2, DT * units.fs, temperature_K=T, friction=FRICTION, rng=np.random.default_rng(SEED))
log = {"t_fs": [], "epot": [], "r": [], "tkin": [], "ref_call": [], "pos": [], "fz": []}
last = [0]


def record():
    log["t_fs"].append(dyn.nsteps * DT)
    log["epot"].append(float(h2.get_potential_energy()))
    log["r"].append(float(h2.get_distance(0, 1)))
    log["tkin"].append(float(h2.get_temperature()))
    log["ref_call"].append(int(NCALLS[0] > last[0])); last[0] = NCALLS[0]
    log["pos"].append(h2.positions.tolist())
    f = h2.get_forces(); u = h2.positions[1] - h2.positions[0]; u /= np.linalg.norm(u)
    log["fz"].append(float(np.dot(f[1] - f[0], u) / 2))   # bond-projected force


dyn.attach(record, interval=1)
t0 = time.time(); dyn.run(STEPS); wall = time.time() - t0

# exact energy along the trajectory (tests surrogate vs truth, e.g. during energy spikes)
import oracle    # exact FCI curve (spline of PySCF energies and gradients, error < 1e-6)
ex = [oracle.energy(r) for r in log["r"]]
fx = [oracle.dEdr(r) for r in log["r"]]     # bond-projected force on atom 1 = -dE/dr along u ... sign below
fx = [-g for g in fx]
log["epot_exact"] = ex; log["fz_exact"] = fx; del log["pos"]
log.update(mode=MODE, ref=REF, seed=SEED, steps=STEPS, wall_s=wall, ref_calls=NCALLS[0],
           settings=dict(T=T, dt_fs=DT, friction=FRICTION, accuracy_e=EPS, accuracy_f=ACC_F, train_start=TRAIN_START, train_log=TRAIN_LOG, model=MODEL))
json.dump(log, open(f"{OUT}.json", "w"))
err = np.array(log["epot"]) - np.array(ex)
ferr = np.abs(np.array(log["fz"]) - np.array(fx))
print(f"{OUT}: max |F_pred - F_exact| (bond) = {ferr.max():.3f} eV/A, mean = {ferr.mean():.4f}")
print(f"{OUT}: {STEPS} steps = {STEPS*DT/1000:.2f} ps, ref calls = {NCALLS[0]}, wall = {wall:.0f} s, "
      f"max |E_pred - E_exact| = {np.abs(err).max():.3f} eV, Epot range = {min(log['epot']):.2f}..{max(log['epot']):.2f} eV, "
      f"r range = {min(log['r']):.3f}..{max(log['r']):.3f} A", flush=True)
