from setuptools import setup

setup(
    name='ase_quantum_vqe',
    version='1.0.0',
    description='The python distribution for the ASE Qiskit interface',
    url='https://github.com/thequantumchemist/ase_quantum_vqe',
    author='Wilke Dononelli',
    author_email='wido@uni-bremen.de',
    license='GPL-3.0',
    packages=['.','utils'],
    install_requires=['agox',
                      'numpy',
                      'ase',
                      'pytest'
                      ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',

        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)
