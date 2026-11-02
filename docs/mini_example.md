## Minimal example

Geometry and harmonic frequency of H₂ with ADAPT-VQE, compared with FCI in the same basis, taken from
the [quickstart tutorial](tutorials/01-quickstart.md):

```python
--8<-- "tutorials/01_quickstart.py:imports"

--8<-- "tutorials/01_quickstart.py:calculator"

--8<-- "tutorials/01_quickstart.py:optimise"

--8<-- "tutorials/01_quickstart.py:vibrations"
```

To use analytic forces instead of finite differences, replace the calculator:

```python
from ase_quantum_vqe.pennylane_calculator import PennyLaneAdaptVQECalculator
atoms.calc = PennyLaneAdaptVQECalculator(basis="sto-3g")
```

## Important settings

| Parameter | Default | Meaning |
|---|---|---|
| `basis` | `"sto3g"` / `"sto-3g"` | Gaussian basis set (Qiskit / PennyLane spelling) |
| `charge`, `spin` | `0`, `0` | total charge and 2S |
| `coreorb` | `0` | number of frozen doubly occupied orbitals |
| `backend` (Qiskit) | `"statevector"` | `"statevector"`, `"aer_shots"` or an IBM backend name |
| `delta`, `nfree` (Qiskit) | `1e-3`, `2` | finite-difference step (Å) and 2- or 4-point stencil |
| `n_jobs` (Qiskit) | `1` | parallel displaced VQE solves per force call |
| `vqe_eigenvalue` | `1e-7` / `1e-5` | ADAPT threshold (Qiskit: energy change; PennyLane: gradient) |
| `maxiter` | `200` | optimizer iterations per VQE solve |

Finite differences amplify any lack of convergence of the displaced VQE solves; analytic forces avoid
this and need one VQE solve per force call. The [manual](manual.md) lists every option.
