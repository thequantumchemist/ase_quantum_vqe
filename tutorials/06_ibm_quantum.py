"""
Tutorial 6: running on IBM Quantum hardware, tested locally with a fake backend (~20 s).

    pip install "ase_quantum_vqe[aer]"
    python 06_ibm_quantum.py
"""
# --8<-- [start:imports]
from ase import Atoms
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from qiskit.primitives import Estimator
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock

from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator, make_ibm_estimator
# --8<-- [end:imports]

# --8<-- [start:real]
# On real hardware, give the backend name; the calculator connects through Qiskit Runtime:
#
#   calc = QiskitVQECalculator(backend="ibm_brisbane", ibm_channel="ibm_quantum_platform",
#                              ibmq_token="...", ibm_instance="...",
#                              shots=4000, resilience_level=1)
# --8<-- [end:real]

# --8<-- [start:fake]
# Local test of exactly the same code path: a fake backend with the noise model of a real device
estimator = make_ibm_estimator(FakeManilaV2(), shots=4000, resilience_level=0)
# --8<-- [end:fake]

# --8<-- [start:shallow]
# a shallow circuit (Hartree-Fock reference, X gates only): device noise is small
problem = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g").run()
mapper = JordanWignerMapper()
H = mapper.map(problem.hamiltonian.second_q_op())
hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
e_hf_exact = Estimator().run(hf, H).result().values[0]
e_hf_fake = estimator.run(hf, H).result().values[0]
print(f"HF electronic energy: exact {e_hf_exact:.3f} Ha, fake backend {e_hf_fake:.3f} Ha")
# --8<-- [end:shallow]

# --8<-- [start:deep]
# full ADAPT-VQE on the fake backend: deep circuits decohere without error mitigation
h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.735]])
h2.calc = QiskitVQECalculator(estimator_override=estimator, maxiter=20)
e_noisy = h2.get_potential_energy()

h2.calc = QiskitVQECalculator(backend="statevector")
print(f"ADAPT-VQE total energy: fake backend {e_noisy:.3f} eV, exact {h2.get_potential_energy():.3f} eV")
# --8<-- [end:deep]
