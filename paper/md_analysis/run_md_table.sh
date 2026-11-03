#!/bin/sh
# Paper: Figure 4 and Table 7. Runs H2 Langevin MD (600 K, 0.5 fs, friction 0.002).
# Runtime on one core: ~30 s per 10 ps FALCON run, ~20 s per exact run, ~5 min for the energy-only run.

# Figure 4: energy-only FALCON (default AGOX GPR, as in the original setup) vs force-controlled distance GP
MODE=falcon REF=pennylane MODEL=agox SEED=2 OUT=falcon_pl_2f python h2_md.py
MODE=falcon REF=pennylane MODEL=agox SEED=1 OUT=falcon_pl_1 python h2_md.py
MODE=falcon REF=pennylane MODEL=distance TRAIN_START=2 TRAIN_LOG=0 ACC_E=0.05 ACC_F=0.1 SEED=2 OUT=nolog_e0.05_f0.1_s2 python h2_md.py
python plot_md_final.py 4072

# Table: effect of model and criterion (same initial conditions)
MODE=falcon REF=pennylane MODEL=agox ACC_F=1.0 SEED=2 OUT=falcon_pl_2_af10 python h2_md.py
MODE=falcon REF=pennylane MODEL=distance TRAIN_START=2 TRAIN_LOG=0 ACC_E=0.05 SEED=2 OUT=nolog_e0.05_finf_s2 python h2_md.py

# Table 7: 10 ps, ten seeds each
for s in 1 2 3 4 5 6 7 8 9 10; do
  MODE=direct REF=oracle THERMO=falcon SEED=$s STEPS=20000 OUT=long_exact_s$s python h2_md.py
  MODE=falcon REF=pennylane MODEL=distance TRAIN_START=2 TRAIN_LOG=0 ACC_E=0.05 ACC_F=0.1 SEED=$s STEPS=20000 OUT=long_f0.1_s$s python h2_md.py
  MODE=falcon REF=qiskit    MODEL=distance TRAIN_START=2 TRAIN_LOG=0 ACC_E=0.05 ACC_F=0.1 SEED=$s STEPS=20000 OUT=long_qk_f0.1_s$s python h2_md.py
done
for s in 1 2 3 4; do
  MODE=falcon REF=pennylane MODEL=distance TRAIN_START=2 TRAIN_LOG=0 ACC_E=0.05 ACC_F=0.5 SEED=$s STEPS=20000 OUT=long_f0.5_s$s python h2_md.py
done
python summarize_md.py

# offline test of surrogate settings on exact 2.5 ps trajectories
MODE=direct REF=fci SEED=1 OUT=direct_1 python h2_md.py
MODE=direct REF=fci SEED=2 OUT=direct_2 python h2_md.py
python pseudo_al.py
