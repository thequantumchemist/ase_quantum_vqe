from setuptools import setup, find_packages

setup(
    name='ase_quantum_vqe',
    version='1.0.0',
    description='The python distribution for the ASE Qiskit interface',
    url='https://github.com/thequantumchemist/ase_quantum_vqe',
    author='Wilke Dononelli',
    author_email='wido@uni-bremen.de',
    license='GPL-3.0',
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        'qiskit==1.4.4',
        'qiskit-ibm-runtime==0.41.1',
        'qiskit-algorithms==0.3.1',
        'qiskit-nature==0.7.2',
        'qiskit-nature-pyscf>=0.4.0',
        'ase>=3.26',
        'agox>=3.11',
        'falcon-md>=1.0',
        'numpy>=2.0',
        'joblib>=1.5.1',
        'strainjedi>=1.0'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
