"""Paper: Figure 2 (single-column). Expected RMS force error from the exact sampling variances (lines) and
sampled RMS errors for H2 with qiskit-aer (markers, 16th-84th percentile bootstrap error bars)."""
import json, os, numpy as onp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
def _find(n):
    for p in (n, os.path.join(here, n), os.path.join(here, "..", "data", "noise", n)):
        if os.path.exists(p): return p
d = json.load(open(_find("exact_variance.json")))
smp_path = _find("sampled_h2.json") or ""
smp = json.load(open(smp_path)) if os.path.exists(smp_path) else None

style = {"qiskit_fd_0.001": (r"finite differences, $\delta_F$ = 0.001 Å", "-", "#0072B2", "o"),
         "qiskit_fd_0.01": (r"finite differences, $\delta_F$ = 0.01 Å", "--", "#E69F00", "s"),
         "pennylane_analytic": ("analytic", "-", "#009E73", "^")}
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "axes.linewidth": 0.6})
fig, axes = plt.subplots(2, 1, figsize=(3.3, 4.4), sharex=True, gridspec_kw={"hspace": 0.12})
s = onp.logspace(0, 14, 400)
for ax, name, title in zip(axes, ["H2", "H3+"], ["H$_2$", "H$_3^+$"]):
    res = d[name]
    for key, (lab, ls, col, mk) in style.items():
        r = res[key]; b2 = onp.mean(onp.array(r["bias"]) ** 2); v = onp.mean(r["var_per_shot"])
        ax.loglog(s * r["settings_per_force"], onp.sqrt(b2 + v / s), ls, color=col, lw=1.1, label=lab)
        if name == "H2" and smp is not None and key in smp:
            x = onp.array(smp["shots_per_setting"]) * r["settings_per_force"]
            y = onp.array(smp[key]["rms"]); lo = onp.array(smp[key]["lo"]); hi = onp.array(smp[key]["hi"])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=mk, ms=3.2, mfc=col, mec="black", mew=0.4,
                        ecolor="black", elinewidth=0.6, capsize=1.5, zorder=5)
    for thr in (0.05, 0.005):
        ax.axhline(thr, color="0.55", lw=0.6, ls=":")
    ax.set_xlim(1e3, 1e14); ax.set_ylim(1e-3, 1e3)
    ax.set_ylabel(r"RMS force error (eV Å$^{-1}$)")
    ax.text(0.04, 0.05, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
axes[-1].set_xlabel("total shots per force evaluation")
h, l = axes[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h.append(Line2D([], [], marker="o", ls="", ms=3.2, mfc="0.8", mec="black", mew=0.4)); l.append("sampled (qiskit-aer)")
axes[0].legend(h, l, loc="upper right", frameon=False, handlelength=2.2, labelspacing=0.3)
for a, t in zip(axes, "ab"):
    a.text(0.0, 1.02, f"({t})", transform=a.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=7)
fig.savefig("noise_forces.pdf", bbox_inches="tight")
fig.savefig("noise_forces.png", dpi=600, bbox_inches="tight")
print("ok")
