from setuptools import setup, find_packages

setup(
    name="ase_quantum_vqe",
    version="1.1.0",
    description="ASE calculators for variational quantum eigensolver (VQE) energies, forces, and dipole moments "
                "(Qiskit and PennyLane backends)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thequantumchemist/ase_quantum_vqe",
    author="Wilke Dononelli",
    author_email="wido@uni-bremen.de",
    license="GPL-3.0",
    packages=find_packages(exclude=("tests", "paper")),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "ase>=3.26",
        "pyscf>=2.6",
        "joblib>=1.3",
        "qiskit==1.4.4",
        "qiskit-algorithms==0.3.1",
        "qiskit-nature==0.7.2",
        "qiskit-ibm-runtime==0.41.1",
        # recent releases ship wheels; older ones are sdists whose setup.py needs pkg_resources,
        # which current setuptools no longer provides (breaks fresh installs, e.g. with uv)
        "ibm-platform-services>=0.77",
    ],
    extras_require={
        "pennylane": ["pennylane>=0.45,<0.46"],
        "aer": ["qiskit-aer>=0.15,<0.16"],
        # FALCON/AGOX currently require ASE <= 3.26 (agox imports ase.constraints.IndexedConstraint)
        "falcon": ["falcon-md>=1.3", "agox>=3.11", "ase>=3.26,<3.27"],
        # strainjedi 1.1 needs ASE >= 3.28, FALCON/AGOX need ASE <= 3.26; with [all], strainjedi 1.0 is used
        "jedi": ["strainjedi>=1.0"],
        "test": ["pytest"],
        "all": ["pennylane>=0.45,<0.46", "qiskit-aer>=0.15,<0.16", "falcon-md>=1.3", "agox>=3.11",
                "ase>=3.26,<3.27", "strainjedi>=1.0", "matplotlib", "pytest"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
