"""Paper: Figure 4 (single-column). Exact vs FALCON-predicted energy and force along two H2 trajectories,
with the steps at which VQE was called (markers in the run color with black edge, on top of the lines). Usage: python plot_md_onecol.py [zoom_start=4072]"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from runio import load_run

ZOOM0 = int(sys.argv[1]) if len(sys.argv) > 1 else 4072
zoom = (ZOOM0, ZOOM0 + 60)
runs = [("falcon_pl_2f", r"fingerprint GPR, $\varepsilon_E$", {"E": "#E69F00", "F": "#D55E00"}),
        ("nolog_e0.05_f0.1_s2", r"distance GP, $\varepsilon_E + \varepsilon_F$", {"E": "#0072B2", "F": "#56B4E9"})]
MK = dict(marker="o", ls="", mec="black", mew=0.35, zorder=5)
data = {k: load_run(k) for k, _, _ in runs}
emin = min(min(d["epot_exact"]) for d in data.values())

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(2, 2, figsize=(3.3, 3.5), sharex="col",
                       gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.08, "hspace": 0.22})

for key, label, col in runs:
    d = data[key]; step = np.arange(1, len(d["epot"]) + 1)
    calls = np.array(d["ref_call"], dtype=bool)
    rows = [("E", np.array(d["epot_exact"]) - emin, np.array(d["epot"]) - emin),
            ("F", np.array(d["fz_exact"]), np.array(d["fz"]))]
    for r, (q, exact, pred) in enumerate(rows):
        c = col[q]
        # full trajectory: exact values only
        ax[r, 0].plot(step, exact, "-", color=c, lw=0.35, zorder=2)
        ax[r, 0].plot(step[calls], exact[calls], mfc=c, ms=2.0, **MK)
        # zoom: exact (thick, light) and values used by the MD (dashed with points)
        sl = slice(zoom[0] - 1, zoom[1])
        ax[r, 1].plot(step[sl], exact[sl], "-", color=c, lw=2.4, alpha=0.35, zorder=2)
        ax[r, 1].plot(step[sl], pred[sl], "--", color=c, lw=0.8, zorder=3)
        zc = calls[sl]
        ax[r, 1].plot(step[sl][zc], pred[sl][zc], mfc=c, ms=3.2, **MK)

ax[0, 0].set_ylabel(r"$E_\mathrm{pot}-E_\mathrm{min}$ (eV)")
ax[1, 0].set_ylabel(r"bond force (eV $\mathrm{\AA}^{-1}$)")
ax[1, 0].set_xlabel("MD step (0.5 fs)"); ax[1, 1].set_xlabel("MD step")
ax[0, 0].set_xlim(0, 5000); ax[0, 1].set_xlim(*zoom)
ax[0, 0].set_xticks([0, 2000, 4000]); ax[0, 1].set_xticks([zoom[0] + 8, zoom[0] + 28, zoom[0] + 48])
for r in range(2):
    ax[r, 1].yaxis.tick_right()
    ax[r, 0].axvspan(*zoom, color="0.85", lw=0, zorder=0)
for a, s in zip(ax.flat, "abcd"):
    a.text(0.0, 1.02, f"({s})", transform=a.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=7)

for r, q in enumerate("EF"):
    handles = [Line2D([], [], color=runs[0][2][q], lw=1.2, label=runs[0][1]),
               Line2D([], [], color=runs[1][2][q], lw=1.2, label=runs[1][1]),
               Line2D([], [], marker="o", ls="", ms=3.0, mfc="0.75", mec="black", mew=0.35,
                      label="VQE call")]
    ax[r, 0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
                    handlelength=1.4, borderaxespad=0.3, labelspacing=0.25)
fig.savefig("h2_md_onecol.pdf", bbox_inches="tight"); fig.savefig("h2_md_onecol.png", dpi=600, bbox_inches="tight")
print("ok")
