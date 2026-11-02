# 3. Active spaces and classical references

Larger molecules need active spaces. `coreorb` freezes the lowest doubly occupied orbitals in both
quantum backends and in `PySCFCalculator`, so that VQE and its classical references describe exactly
the same problem.

This tutorial demonstrates:

- A frozen-core active space for F₂ (eight frozen orbitals, two active electrons in two orbitals)
- FCI and CCSD references in the same active space
- An energy scan along the bond

```bash
python 03_active_spaces.py
```

## Step 1: Energy scan

```python
--8<-- "tutorials/03_active_spaces.py:imports"

--8<-- "tutorials/03_active_spaces.py:scan"
```

## Step 2: Compare

```python
--8<-- "tutorials/03_active_spaces.py:print"
```

```text
r (A)                 ADAPT-VQE   FCI (same active space)  CCSD (same active space)
 1.25               -5332.74973               -5332.74973               -5332.74973
 1.30               -5332.97440               -5332.97440               -5332.97440
 1.35               -5332.93075               -5332.93075               -5332.93075
 1.40               -5332.68800               -5332.68800               -5332.68800
max |E_VQE - E_FCI| = 1.0e-08 eV
```

With two active electrons, CCSD is exact as well. For more active electrons, use `method="fci"` as the
reference: it runs FCI in the frozen-core active space (a CASCI with all non-core orbitals active).
