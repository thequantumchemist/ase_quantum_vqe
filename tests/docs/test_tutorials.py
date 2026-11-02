"""
The tutorial scripts run without errors.

The documentation includes its code from these scripts, so this keeps the docs in sync with the
package. Tutorials whose optional dependencies are missing are skipped. Deselect with
`pytest -m "not tutorial"`.
"""
import importlib.util
import runpy
from pathlib import Path

import pytest

TUTORIALS = Path(__file__).parents[2] / "tutorials"

pytestmark = pytest.mark.tutorial

NEEDS = {
    "01_quickstart.py": [],
    "02_analytic_forces.py": ["pennylane"],
    "03_active_spaces.py": [],
    "04_shot_noise.py": ["pennylane", "qiskit_aer"],
    "05_active_learning_md.py": ["pennylane", "falcon_md"],
    "06_ibm_quantum.py": ["qiskit_aer"],
    "07_workflows.py": ["pennylane", "strainjedi"],
}


@pytest.mark.parametrize("script", sorted(NEEDS))
def test_tutorial_runs(script, tmp_path, monkeypatch):
    for module in NEEDS[script]:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} not installed")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(TUTORIALS))      # as `python script.py` does
    runpy.run_path(str(TUTORIALS / script), run_name="__main__")


def test_every_tutorial_is_tested():
    assert sorted(p.name for p in TUTORIALS.glob("[0-9]*.py")) == sorted(NEEDS)
