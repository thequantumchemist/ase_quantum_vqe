"""Paper: Table 7. Statistics of the 10 ps H2 runs (exact forces vs FALCON + distance GP)."""
import numpy as np
from runio import load_run

ROWS = [("exact (FCI)", "long_exact", range(1, 11), False),
        ("PennyLane, eps_F = 0.1", "long_f0.1", range(1, 11), True),
        ("Qiskit, eps_F = 0.1", "long_qk_f0.1", range(1, 11), True),
        ("PennyLane, eps_F = 0.5", "long_f0.5", range(1, 5), True)]
N_OPT = 3   # VQE calculations of the preceding geometry optimization (initial training data)

print(f"{'forces':24s} {'VQE calc':>12s} {'<r>/A':>7s} {'sd(r)/A':>8s} {'<T>/K':>12s} {'mean|dF|':>9s} {'max|dF|':>8s}")
for label, prefix, seeds, surrogate in ROWS:
    runs = [load_run(f"{prefix}_s{s}") for s in seeds]
    r = np.concatenate([d["r"] for d in runs]); T = np.array([np.mean(d["tkin"]) for d in runs])
    calls = [d["ref_calls"] + N_OPT for d in runs]
    dF = [np.abs(np.array(d["fz"]) - np.array(d["fz_exact"])) for d in runs]
    c = f"{min(calls)}-{max(calls)} ({np.mean(calls):.1f})" if surrogate else "-"
    m = f"{np.mean([x.mean() for x in dF]):.3f}" if surrogate else "-"
    x = f"{max(x.max() for x in dF):.2f}" if surrogate else "-"
    print(f"{label:24s} {c:>12s} {r.mean():7.3f} {r.std():8.3f} {T.mean():6.0f} +- {T.std(ddof=1)/np.sqrt(len(T)):3.0f} {m:>9s} {x:>8s}")
