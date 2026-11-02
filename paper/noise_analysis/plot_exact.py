"""Paper: Figure 2. Expected RMS force error vs total shots per force evaluation, from exact sampling variances."""
import json, numpy as onp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
src = "exact_variance.json" if os.path.exists("exact_variance.json") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "noise", "exact_variance.json")
d = json.load(open(src))
style = {"qiskit_fd_0.001": ("finite differences, $\\delta_F$ = 0.001 Å", "-"),
         "qiskit_fd_0.01": ("finite differences, $\\delta_F$ = 0.01 Å", "--"),
         "pennylane_analytic": ("analytic", "-")}
colors = {"qiskit_fd_0.001": "C0", "qiskit_fd_0.01": "C1", "pennylane_analytic": "C2"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
s = onp.logspace(0, 14, 400)                   # shots per measurement setting
for ax, (name, res) in zip(axes, d.items()):
    for key, (lab, ls) in style.items():
        r = res[key]
        b2 = onp.mean(onp.array(r["bias"]) ** 2); v = onp.mean(r["var_per_shot"])
        ax.loglog(s * r["settings_per_force"], onp.sqrt(b2 + v / s), ls, color=colors[key], label=lab)
    for thr in (0.05, 0.005):
        ax.axhline(thr, color="gray", lw=0.8, ls=":")
    ax.set_xlim(1e3, 1e14); ax.set_ylim(1e-3, 1e3)
    ax.set_title({"H2": "H$_2$", "H3+": "H$_3^+$"}[name]); ax.set_xlabel("total shots per force evaluation")
axes[0].set_ylabel("RMS force error (eV Å$^{-1}$)"); axes[0].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("noise_forces.pdf"); fig.savefig("noise_forces.png", dpi=300)
