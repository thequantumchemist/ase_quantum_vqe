# 6. Running on IBM Quantum hardware

The Qiskit backend runs on IBM Quantum devices through Qiskit Runtime. This tutorial shows the settings
for a real device and tests the same code path locally with a fake backend that carries the noise model
of a real device.

This tutorial demonstrates:

- Connecting to a real device
- A local test with `make_ibm_estimator` and a fake backend
- The effect of circuit depth under device noise

```bash
python 06_ibm_quantum.py
```

## Step 1: Settings for a real device

```python
--8<-- "tutorials/06_ibm_quantum.py:imports"

--8<-- "tutorials/06_ibm_quantum.py:real"
```

Qiskit Runtime provides V2 estimators, while `qiskit-algorithms` expects V1 estimators. The calculator
wraps the Runtime estimator in an adapter that transpiles circuits to the device and maps the
observables onto the qubit layout.

## Step 2: A fake backend

```python
--8<-- "tutorials/06_ibm_quantum.py:fake"
```

## Step 3: Shallow and deep circuits

```python
--8<-- "tutorials/06_ibm_quantum.py:shallow"

--8<-- "tutorials/06_ibm_quantum.py:deep"
```

```text
HF electronic energy: exact -1.837 Ha, fake backend -1.717 Ha
ADAPT-VQE total energy: fake backend -6.117 eV, exact -30.948 eV
```

The shallow Hartree–Fock circuit is only shifted by readout and gate errors. The ADAPT-VQE circuits,
transpiled to the device's connectivity, are deep enough to decohere almost completely without error
mitigation. Real-device calculations therefore need error mitigation (`resilience_level`), compact
circuits and, for forces, analytic estimators.

!!! note
    The adapter is tested with fake backends, not on real hardware.
