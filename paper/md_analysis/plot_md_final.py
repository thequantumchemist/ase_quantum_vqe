"""Paper: Figure 4. Exact vs FALCON-predicted energy and force along two H2 trajectories.
Usage: python plot_md_final.py 4072   (start step of the zoom window)."""
import json, numpy as np
from runio import load_run
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
ZOOM0 = int(sys.argv[1])

runs = [("falcon_pl_2f", "energy criterion, agox GPR (525 VQE calculations)", "#E69F00"),
        ("nolog_e0.05_f0.1_s2", "force criterion, distance GP (7 VQE calculations)", "#0072B2")]
data = {k: load_run(k) for k, _, _ in runs}
emin = min(min(d["epot_exact"]) for d in data.values())
zoom = (ZOOM0, ZOOM0 + 60)

plt.rcParams.update({"font.size": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex="col", gridspec_kw={"width_ratios": [2, 1]}, constrained_layout=True)
for key, label, c in runs:
    d = data[key]; step = np.arange(1, len(d["epot"]) + 1)
    E_ex, E_pr = np.array(d["epot_exact"]) - emin, np.array(d["epot"]) - emin
    F_ex, F_pr = np.array(d["fz_exact"]), np.array(d["fz"])
    for col, sl in ((0, slice(None)), (1, slice(*zoom))):
        lw_ex, lw_pr = (0.5, 0.5) if col == 0 else (3.0, 1.2)
        ax[0, col].plot(step[sl], E_ex[sl], "-", color=c, lw=lw_ex, alpha=0.35 if col else 1, label=f"{label}, exact")
        ax[1, col].plot(step[sl], F_ex[sl], "-", color=c, lw=lw_ex, alpha=0.35 if col else 1)
        if col == 1:
            ax[0, col].plot(step[sl], E_pr[sl], "--", color=c, lw=lw_pr, marker="o", ms=2, label=f"{label}, predicted")
            ax[1, col].plot(step[sl], F_pr[sl], "--", color=c, lw=lw_pr, marker="o", ms=2)
ax[0, 0].set_ylabel("$E_\\mathrm{pot} - E_\\mathrm{min}$ (eV)"); ax[1, 0].set_ylabel("force along H–H bond (eV Å$^{-1}$)")
ax[1, 0].set_xlabel("MD step (0.5 fs)"); ax[1, 1].set_xlabel("MD step (zoom)")
ax[0, 0].set_xlim(0, 5000); ax[0, 1].set_xlim(*zoom)
for a in ax[:, 0]:
    a.axvspan(*zoom, color="gray", alpha=0.15, lw=0)
h, l = ax[0, 1].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
ax[0, 0].set_title("full trajectory", fontsize=8); ax[0, 1].set_title(f"steps {zoom[0]}–{zoom[1]}", fontsize=8)
fig.savefig("h2_md_final.pdf", bbox_inches="tight"); fig.savefig("h2_md_final.png", dpi=300, bbox_inches="tight")
print("ok")
