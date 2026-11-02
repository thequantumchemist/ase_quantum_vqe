# 7. Workflows: optimization, vibrations and strain analysis

Because the calculators follow ASE's interface, existing workflows run unchanged. This tutorial chains a
geometry optimization, a vibrational analysis and a JEDI strain analysis for H₃⁺ with analytic VQE
forces.

This tutorial demonstrates:

- Geometry optimization and harmonic frequencies of H₃⁺
- The Hessian from ASE's `Vibrations` as input for further analysis
- Strain analysis with JEDI

```bash
python 07_workflows.py
```

## Step 1: Optimize

```python
--8<-- "tutorials/07_workflows.py:imports"

--8<-- "tutorials/07_workflows.py:optimise"
```

## Step 2: Vibrations

```python
--8<-- "tutorials/07_workflows.py:vibrations"
```

The equilateral ion has a doubly degenerate e′ pair near 2118 cm⁻¹ and a symmetric a₁′ stretch near
3447 cm⁻¹ in the STO-3G basis.

## Step 3: Strain analysis with JEDI

```python
--8<-- "tutorials/07_workflows.py:jedi"
```

JEDI distributes the strain energy of the stretched structure over bonds and angles, using the VQE
Hessian. The bonds of H₃⁺ (about 1 Å) are longer than JEDI's default covalent cutoff: strainjedi 1.0
finds them with a larger covalent factor, strainjedi 1.1 needs them explicitly, and the script handles
both. For the global minimum search with minima hopping, see `paper/h3+.py` in the repository.
