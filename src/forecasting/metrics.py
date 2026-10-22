from __future__ import annotations

import numpy as np

DEFAULT_TAUS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])


def pinball_loss(y: np.ndarray, q: np.ndarray, taus: np.ndarray) -> float:
    e = np.asarray(y, dtype=float)[..., None] - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(taus * e, (taus - 1.0) * e)))


def crps_from_quantiles(
    y: np.ndarray, q: np.ndarray, taus: np.ndarray = DEFAULT_TAUS
) -> float:
    y2 = np.asarray(y, dtype=float)[..., None]
    q = np.asarray(q, dtype=float)
    e = y2 - q
    terms = np.maximum(taus * e, (taus - 1.0) * e)
    return float(2.0 * np.mean(terms.sum(axis=-1)) / len(taus))


def crps_from_scenarios(y: np.ndarray, scenarios: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    s = np.asarray(scenarios, dtype=float)
    n = s.shape[-1]
    term1 = np.mean(np.abs(s - y[..., None]), axis=-1)
    i, j = np.triu_indices(n, k=1)
    term2 = 0.5 * np.mean(np.abs(s[..., i] - s[..., j]), axis=-1)
    return float(np.mean(term1 - term2))


def empirical_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def conformal_interval(
    y_cal: np.ndarray, q_med_cal: np.ndarray, y_test: np.ndarray, q_med_test: np.ndarray, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.abs(y_cal - q_med_cal)
    qhat = np.quantile(scores, 1.0 - alpha)
    return q_med_test - qhat, q_med_test + qhat


def evaluate_forecast(
    y: np.ndarray, q: np.ndarray, taus: np.ndarray = DEFAULT_TAUS
) -> dict:
    lo_i = int(np.argmin(np.abs(taus - 0.1)))
    hi_i = int(np.argmin(np.abs(taus - 0.9)))
    med_i = int(np.argmin(np.abs(taus - 0.5)))
    lo, hi = q[..., lo_i], q[..., hi_i]
    return {
        "pinball_mean": pinball_loss(y, q, taus),
        "crps": crps_from_quantiles(y, q, taus),
        "mae_median": float(np.mean(np.abs(y - q[..., med_i]))),
        "coverage_80": empirical_coverage(y, lo, hi),
        "width_80": mean_width(lo, hi),
    }
