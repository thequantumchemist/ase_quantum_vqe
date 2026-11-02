"""Gaussian-process model on pairwise interatomic distances, compatible with FALCON.

Features: all pairwise distances d_p (fixed atom order; rotation/translation invariant). For H2 this is
just the bond length. The GP can be trained on energies only, or on energies and forces
(gradient-enhanced GP): the kernel derivatives are mapped to Cartesian coordinates with the
Jacobian dd_p/dR, so predicted forces are exact derivatives of the predicted energy and both
energy and force uncertainties are available.

Hyperparameters: RBF length scale chosen by maximizing the (profiled) log marginal likelihood on a
grid at each training; amplitude profiled analytically; constant prior mean = mean training energy.

Usage with FALCON (force criterion, as in the paper):

    from falcon_md.otf_calculator import FALCON
    from ase_quantum_vqe.surrogates import DistanceGPR
    atoms.calc = FALCON(model=DistanceGPR(use_forces=True), calc=vqe_calc,
                        training_data=initial_frames, train_start=2, train_log=False,
                        accuracy_e=0.05, accuracy_f=0.1)

Suited to small molecules with a fixed atom ordering (features are not permutation invariant and
their number grows quadratically with the number of atoms).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class DistanceDescriptor:
    """FALCON calls get_features(list) and create_features(atoms)."""

    def features(self, atoms):
        R = atoms.get_positions(); n = len(R)
        return np.array([np.linalg.norm(R[i] - R[j]) for i in range(n) for j in range(i + 1, n)])

    def get_features(self, structures):
        return np.array([[self.features(a)] for a in structures])      # (n_struct, 1, n_pairs)

    def create_features(self, atoms):
        return self.features(atoms)[None, :]                            # (1, n_pairs)


def _jacobian(R):
    """J[p, a*3+alpha] = d d_p / d R_{a,alpha}."""
    n = len(R); pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    J = np.zeros((len(pairs), 3 * n)); d = np.zeros(len(pairs))
    for p, (i, j) in enumerate(pairs):
        v = R[i] - R[j]; d[p] = np.linalg.norm(v); u = v / d[p]
        J[p, 3 * i:3 * i + 3] = u; J[p, 3 * j:3 * j + 3] = -u
    return d, J


class DistanceGPR:
    implemented_properties = ["energy", "forces", "uncertainty"]

    def __init__(self, use_forces=True, noise_E=1e-3, noise_F=1e-2,
                 length_scales=np.logspace(np.log10(0.03), np.log10(1.0), 16),
                 s2_min=0.1, n_fixed=3, l_default=0.3, s2_default=1.0):
        self.descriptor = DistanceDescriptor()
        self.use_forces = use_forces
        self.noise_E, self.noise_F = noise_E, noise_F
        self.length_scales = length_scales
        # amplitude floor and fixed hyperparameters for tiny training sets: with one or two points the
        # marginal likelihood cannot determine the amplitude, which would make the model overconfident
        self.s2_min, self.n_fixed, self.l_default, self.s2_default = s2_min, n_fixed, l_default, s2_default
        self.trained = False

    # ---------------- kernel blocks (s^2 = 1) ----------------
    def _blocks(self, xa, Ja, xb, Jb, l, grads_a, grads_b):
        """Covariance between observations at a and b. Returns matrix with rows [E_a, g_a(3N)] x cols [E_b, g_b]."""
        diff = xa - xb; k = np.exp(-diff @ diff / (2 * l * l))
        dk_dxb = k * diff / l ** 2                                     # dk/dx'_m
        H = k * (np.eye(len(xa)) / l ** 2 - np.outer(diff, diff) / l ** 4)   # d2k/dx_l dx'_m
        rows = [np.concatenate([[k], dk_dxb @ Jb])] if grads_b else [np.array([k])]
        if grads_a:
            dk_dxa = -dk_dxb
            ga_E = Ja.T @ dk_dxa                                        # cov(g_a, E_b)
            if grads_b:
                rows.append(np.hstack([ga_E[:, None], Ja.T @ H @ Jb]))
            else:
                rows.append(ga_E[:, None])
        return np.vstack([np.atleast_2d(r) for r in rows])

    def _K(self, l):
        n = len(self.X); blocks = [[self._blocks(self.X[i], self.J[i], self.X[j], self.J[j], l,
                                                  self.use_forces, self.use_forces) for j in range(n)] for i in range(n)]
        return np.block(blocks)

    def train(self, structures):
        self.X, self.J, y = [], [], []
        for a in structures:
            d, J = _jacobian(a.get_positions()); self.X.append(d); self.J.append(J)
            e = a.get_potential_energy(); y.append([e])
            if self.use_forces:
                y[-1] = np.concatenate([[e], -a.get_forces().ravel()])   # gradient = -forces
        self.n_atoms = len(structures[0])
        self.mu = np.mean([v[0] for v in y])
        r = np.concatenate([np.concatenate([[v[0] - self.mu], v[1:]]) for v in y])
        m = 1 + 3 * self.n_atoms if self.use_forces else 1
        is_E = np.tile(np.r_[1.0, np.zeros(m - 1)], len(structures)).astype(bool)
        best = None
        grid = [self.l_default] if len(structures) < self.n_fixed else self.length_scales
        for l in grid:
            K0 = self._K(l)
            if len(structures) < self.n_fixed:
                s2 = self.s2_default
            else:
                s2 = max(r @ np.linalg.lstsq(K0 + 1e-8 * np.eye(len(r)), r, rcond=None)[0] / len(r), self.s2_min)
            K = s2 * K0 + np.diag(np.where(is_E, self.noise_E ** 2, self.noise_F ** 2))
            try:
                c = cho_factor(K)
            except np.linalg.LinAlgError:
                continue
            alpha = cho_solve(c, r)
            lml = -0.5 * r @ alpha - np.log(np.diag(c[0])).sum()
            if best is None or lml > best[0]:
                best = (lml, l, s2, c, alpha)
        _, self.l, self.s2, self.cho, self.alpha = best
        self.trained = True

    # ---------------- prediction ----------------
    def _kstar(self, atoms):
        d, J = _jacobian(atoms.get_positions())
        cols = [self.s2 * self._blocks(d, J, self.X[i], self.J[i], self.l, True, self.use_forces)
                for i in range(len(self.X))]
        return np.hstack(cols), d, J                                    # rows: [E*, g*(3N)]

    def _predict(self, atoms):
        if getattr(self, "_cache_pos", None) is not None and np.array_equal(self._cache_pos, atoms.positions):
            return self._cache
        ks, d, J = self._kstar(atoms)
        mean = ks @ self.alpha; mean[0] += self.mu
        v = cho_solve(self.cho, ks.T)
        prior = np.r_[self.s2, self.s2 / self.l ** 2 * (J ** 2).sum(axis=0)]
        var = np.maximum(prior - np.einsum("ij,ji->i", ks, v), 0.0)
        self._cache_pos = atoms.positions.copy(); self._cache = (mean, var)
        return mean, var

    def predict_energy(self, atoms, **kw):
        return float(self._predict(atoms)[0][0])

    def predict_forces(self, atoms, **kw):
        return -self._predict(atoms)[0][1:].reshape(-1, 3)

    def predict_uncertainty(self, atoms, **kw):
        return float(np.sqrt(self._predict(atoms)[1][0]))

    def predict_uncertainty_forces(self, atoms, **kw):
        return np.sqrt(self._predict(atoms)[1][1:]).reshape(-1, 3)
