from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from src.optimization.ems import BatterySpec


def box_argmin(c: np.ndarray, batt: BatterySpec) -> np.ndarray:
    c = np.asarray(c, dtype=float).ravel()
    h = len(c)
    lo = np.full(h, -batt.p_ch)
    ub = np.full(h, batt.p_dis)
    a = np.zeros((h, h))
    for t in range(h):
        a[t, : t + 1] = 1.0
    a_ub = np.vstack([a, -a])
    b_ub = np.concatenate(
        [np.full(h, batt.e_max - batt.e0), np.full(h, batt.e0 - batt.e_min * batt.e_max)]
    )
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(lo[t], ub[t]) for t in range(h)],
        method="highs",
    )
    if not res.success:
        return lo.copy()
    return res.x


def box_argmax(c: np.ndarray, batt: BatterySpec) -> np.ndarray:
    return box_argmin(-np.asarray(c, dtype=float), batt)


def _spo_terms(c_hat: np.ndarray, c: np.ndarray, batt: BatterySpec) -> tuple[float, np.ndarray]:
    """Termes de la perte SPO+ (Definition 3, Elmachtoub & Grigas, 2022).

    Pour le probleme de minimisation ``min_{u in X} c^T u`` :

        l_SPO+(c_hat, c) = max_{u in X} (c - 2 c_hat)^T u + 2 c_hat^T w*(c) - z*(c)
                         = (2 c_hat - c)^T ( w*(c) - w*(2 c_hat - c) )

    avec ``w*(c) = argmin_{u in X} c^T u`` (oracle sur le cout vrai) et
    ``w*(2 c_hat - c) = argmin_{u in X} (2 c_hat - c)^T u`` (probleme modifie).

    Sous-gradient (Proposition 3.3) :
        dl_SPO+ / d c_hat = 2 ( w*(c) - w*(2 c_hat - c) )

    La theorie garantit l_SPO+ >= l_SPO >= 0 (Proposition 3.1) et
    l_SPO+(c, c) = 0.
    """
    w_c = box_argmin(c, batt)  # w*(c) : decision oracle sous le cout vrai
    w_mod = box_argmin(2.0 * c_hat - c, batt)  # w*(2 c_hat - c) : probleme modifie
    loss = float(np.dot(2.0 * c_hat - c, w_c - w_mod))
    grad = 2.0 * (w_c - w_mod)
    if loss < 0.0:  # bruit numerique uniquement (la theorie donne loss >= 0)
        return 0.0, np.zeros_like(c_hat)
    return loss, grad


def spo_plus_loss(price_hat: np.ndarray, price_true: np.ndarray, batt: BatterySpec) -> float:
    """Perte SPO+ (Definition 3, Elmachtoub & Grigas 2022)."""
    c_hat = np.asarray(price_hat, dtype=float).ravel()
    c = np.asarray(price_true, dtype=float).ravel()
    loss, _ = _spo_terms(c_hat, c, batt)
    return loss


def spo_plus_loss_torch(price_hat, price_true, batt: BatterySpec):
    """
    SPO+ loss function - now optimized version.
    
    This function has been optimized for better performance with batch processing.
    For the original implementation, see spo_optimized.py.
    """
    from .spo_optimized import spo_plus_loss_torch_optimized
    return spo_plus_loss_torch_optimized(price_hat, price_true, batt)
