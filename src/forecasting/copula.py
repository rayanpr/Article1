from __future__ import annotations

import numpy as np
from scipy.stats import norm


class GaussianCopulaScenarios:
    def fit(self, residuals: np.ndarray) -> "GaussianCopulaScenarios":
        r = np.asarray(residuals, dtype=float)
        self.residuals_ = r
        n, d = r.shape
        ranks = np.argsort(np.argsort(r, axis=0), axis=0).astype(float)
        u = (ranks + 0.5) / n
        z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        self.z_ = z
        cov = np.cov(z.T)
        cov = cov + 1e-8 * np.eye(d)
        self.corr_ = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        self.means_ = r.mean(axis=0)
        self.n_ = n
        return self

    def sample(self, n_scenarios: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        d = self.corr_.shape[0]
        L = np.linalg.cholesky(self.corr_)
        z = rng.standard_normal((n_scenarios, d)) @ L.T
        u = norm.cdf(z)
        out = np.empty_like(u)
        for j in range(d):
            out[:, j] = np.quantile(self.residuals_[:, j], np.clip(u[:, j], 0.0, 0.999999))
        return out
