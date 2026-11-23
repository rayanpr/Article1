"""Verification de la conformite de la perte SPO+ implementee a la Definition 3
d'Elmachtoub & Grigas (2022), "Smart Predict, then Optimize" (arXiv:1710.08005).

Reference (Definition 3, papier section 3.2) pour un probleme de minimisation
    P(c) :  z*(c) = min_{w in S} c^T w,      w*(c) = argmin_{w in S} c^T w

    l_SPO+(c_hat, c) = max_{w in S} { c^T w - 2 c_hat^T w } + 2 c_hat^T w*(c) - z*(c)
                     = (2 c_hat - c)^T ( w*(c) - w*(2 c_hat - c) )

    sous-gradient (Proposition 3.3) :
        d l_SPO+ / d c_hat = 2 ( w*(c) - w*(2 c_hat - c) )

et la perte SPO (Definition 2) :
    l_SPO(c_hat, c) = max_{w in W*(c_hat)} c^T w - z*(c) = c^T w*(c_hat) - z*(c) >= 0.

Proprietes verifiees ici :
  P1  l_SPO+(c,c) = 0
  P2  l_SPO+ >= l_SPO >= 0            (Proposition 3.1)
  P3  le gradient implemente est le sous-gradient d'Elmachtoub-Grigas
  P4  l_SPO+ est convexe en c_hat (convexite par difference finie le long de segments)

Usage :
    python scripts/verify_spo_convention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.ems import BatterySpec
from src.optimization.spo import box_argmin, spo_plus_loss, spo_plus_loss_torch

TOL = 1e-8


# --------------------------------------------------------------------------- #
# Reference Elmachtoub-Grigas (Definition 3)
# --------------------------------------------------------------------------- #
def eg_spo_plus(c_hat: np.ndarray, c: np.ndarray, batt: BatterySpec) -> float:
    """l_SPO+ d'Elmachtoub-Grigas pour le modele boite (LP de la batterie)."""
    c_hat = np.asarray(c_hat, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    w_c = box_argmin(c, batt)                 # w*(c)      : oracle vrai cout
    w_mod = box_argmin(2.0 * c_hat - c, batt)  # w*(2c_hat-c): probleme modifie
    return float(np.dot(2.0 * c_hat - c, w_c - w_mod))


def eg_spo_loss(c_hat: np.ndarray, c: np.ndarray, batt: BatterySpec) -> float:
    """Perte SPO (Definition 2) : regret de la decision induite par la prediction."""
    c_hat = np.asarray(c_hat, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    w_hat = box_argmin(c_hat, batt)  # w*(c_hat)
    z_c = float(np.dot(c, box_argmin(c, batt)))  # z*(c)
    return float(np.dot(c, w_hat) - z_c)


def eg_subgradient(c_hat: np.ndarray, c: np.ndarray, batt: BatterySpec) -> np.ndarray:
    """Sous-gradient d'Elmachtoub-Grigas : 2 (w*(c) - w*(2 c_hat - c))."""
    c_hat = np.asarray(c_hat, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    w_c = box_argmin(c, batt)
    w_mod = box_argmin(2.0 * c_hat - c, batt)
    return 2.0 * (w_c - w_mod)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_values_match_reference(rng: np.random.Generator, batt: BatterySpec, n: int = 25) -> bool:
    """La perte implementee doit coincider avec la Definition 3."""
    print("\n[P0] Valeurs : implementation vs Definition 3 (Elmachtoub-Grigas)")
    max_abs, max_rel = 0.0, 0.0
    for _ in range(n):
        c = rng.normal(0.0, 1.0, 24) + 1.0
        c_hat = c + rng.normal(0.0, 0.4, 24)
        got = spo_plus_loss(c_hat, c, batt)
        ref = eg_spo_plus(c_hat, c, batt)
        max_abs = max(max_abs, abs(got - ref))
        max_rel = max(max_rel, abs(got - ref) / max(abs(ref), 1e-9))
    ok = max_abs < 1e-6
    print(f"    ecart absolu max = {max_abs:.3e} | ecart relatif max = {max_rel:.3e}")
    print(f"    -> {'CONFORME' if ok else 'NON CONFORME'}")
    return ok


def test_zero_at_optimum(rng: np.random.Generator, batt: BatterySpec, n: int = 10) -> bool:
    """P1 : l_SPO+(c,c) = 0."""
    print("\n[P1] l_SPO+(c,c) = 0")
    worst = 0.0
    for _ in range(n):
        c = rng.normal(0.0, 1.0, 24) + 1.0
        worst = max(worst, abs(spo_plus_loss(c, c, batt)), abs(eg_spo_plus(c, c, batt)))
    ok = worst < 1e-6
    print(f"    |perte| max sur predicition parfaite = {worst:.3e} -> {'OK' if ok else 'ECHEC'}")
    return ok


def test_dominates_spo(rng: np.random.Generator, batt: BatterySpec, n: int = 50) -> bool:
    """P2 : l_SPO+ >= l_SPO >= 0 (Proposition 3.1)."""
    print("\n[P2] l_SPO+ >= l_SPO >= 0")
    worst_gap, min_spo, min_plus = np.inf, np.inf, np.inf
    for _ in range(n):
        c = rng.normal(0.0, 1.0, 24) + 1.0
        c_hat = c + rng.normal(0.0, 0.8, 24)
        plus = spo_plus_loss(c_hat, c, batt)
        spo = eg_spo_loss(c_hat, c, batt)
        worst_gap = min(worst_gap, plus - spo)
        min_spo = min(min_spo, spo)
        min_plus = min(min_plus, plus)
    ok = worst_gap > -1e-6 and min_spo > -1e-6 and min_plus > -1e-6
    print(f"    min l_SPO = {min_spo:.3e} | min l_SPO+ = {min_plus:.3e} | min ecart (SPO+ - SPO) = {worst_gap:.3e}")
    print(f"    -> {'OK' if ok else 'ECHEC'}")
    return ok


def test_gradient(rng: np.random.Generator, batt: BatterySpec, n: int = 10) -> bool:
    """P3 : gradient implemente == sous-gradient d'Elmachtoub-Grigas (a l'echelle pres).

    L'implementation normalise la perte par ``scale = mean(|c|) * H`` : le gradient
    renvoye est donc ``scale * batch`` fois le sous-gradient theorique.
    """
    print("\n[P3] Gradient : implementation vs sous-gradient 2(w*(c) - w*(2c_hat-c))")
    worst = 0.0
    for _ in range(n):
        c = rng.normal(0.0, 1.0, 24) + 1.0
        c_hat = c + rng.normal(0.0, 0.4, 24)
        ph = torch.tensor(c_hat[None, :], dtype=torch.float64, requires_grad=True)
        loss = spo_plus_loss_torch(ph, torch.tensor(c[None, :], dtype=torch.float64), batt)
        loss.backward()
        scale = float(np.mean(np.abs(c))) * c.size  # normalisation de l'implementation
        got = ph.grad.numpy()[0] * scale
        ref = eg_subgradient(c_hat, c, batt)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    ok = worst < 1e-5
    print(f"    ecart max (gradient a l'echelle) = {worst:.3e} -> {'OK' if ok else 'ECHEC'}")
    return ok


def test_convexity(rng: np.random.Generator, batt: BatterySpec, n: int = 20) -> bool:
    """P4 : convexite de l_SPO+ en c_hat (sur le modele boite, hors ties)."""
    print("\n[P4] Convexite de l_SPO+ en c_hat")
    worst = -np.inf
    for _ in range(n):
        c = rng.normal(0.0, 1.0, 24) + 1.0
        a = c + rng.normal(0.0, 0.5, 24)
        b = c + rng.normal(0.0, 0.5, 24)
        t = rng.uniform()
        la, lb = spo_plus_loss(a, c, batt), spo_plus_loss(b, c, batt)
        lt = spo_plus_loss(t * a + (1 - t) * b, c, batt)
        worst = max(worst, lt - (t * la + (1 - t) * lb))
    ok = worst < 1e-6
    print(f"    violation max de la convexite = {worst:.3e} -> {'OK' if ok else 'ECHEC'}")
    return ok


def test_differentiable_ems(n: int = 4, horizon: int = 24) -> bool | None:
    """P5 : chemin cvxpylayers (LP EMS complet) — SPO+ conforme a la Definition 3."""
    try:
        import cvxpy as cp

        from src.optimization.differentiable import build_ems_layer, spo_exact_loss
    except Exception as exc:  # pragma: no cover - dependance optionnelle
        print(f"\n[P5] EMS differentiel : indisponible ({exc})")
        return None

    print("\n[P5] EMS differentiel (cvxpylayers) : SPO+ conforme a la Definition 3")
    rng = np.random.default_rng(7)
    rho, scale = 0.5, 114.1
    try:
        layer = build_ems_layer(horizon, export_ratio=rho)
    except ImportError as exc:  # pragma: no cover - dependance optionnelle
        print(f"    cvxpylayers indisponible ({exc}) — test ignore (utiliser .venv311)")
        return None

    def ems_solve(net, price):
        g = cp.Variable(horizon, nonneg=True)
        x = cp.Variable(horizon, nonneg=True)
        ch = cp.Variable(horizon, nonneg=True)
        dis = cp.Variable(horizon, nonneg=True)
        e = cp.Variable(horizon + 1)
        cons = [
            e[0] == 0.5 * 100.0,
            e[1:] == e[:-1] + 0.95 * ch - dis / 0.95,
            e >= 0.1 * 100.0,
            e <= 100.0,
            ch <= 30.0,
            dis <= 30.0,
            g - x == net + ch - dis,
            g <= 500.0,
            x <= 500.0,
        ]
        prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(price, g) - cp.multiply(price * rho, x))), cons)
        prob.solve(solver=cp.HIGHS)
        return g.value, x.value

    net = np.abs(rng.normal(10.0, 3.0, (n, horizon)))
    price = np.abs(rng.normal(0.2, 0.05, (n, horizon)))
    price_hat = price + rng.normal(0.0, 0.02, (n, horizon))

    ref_losses, ref_grads = [], []
    for i in range(n):
        g_c, x_c = ems_solve(net[i], price[i])
        g_m, x_m = ems_solve(net[i], 2.0 * price_hat[i] - price[i])
        psi_c, psi_m = g_c - rho * x_c, g_m - rho * x_m
        ref_losses.append(float(np.dot(2.0 * price_hat[i] - price[i], psi_c - psi_m)))
        ref_grads.append(2.0 * (psi_c - psi_m))

    ph = torch.tensor(price_hat, dtype=torch.float64, requires_grad=True)
    pt = torch.tensor(price, dtype=torch.float64)
    loss = spo_exact_loss(layer, torch.tensor(net, dtype=torch.float64), ph,
                          torch.tensor(net, dtype=torch.float64), pt,
                          np.array([0.0, 0.0]), np.array([1.0, 1.0]),
                          export_ratio=rho, beta=0.0, scale=scale)
    loss.backward()

    got_loss = float(loss) * scale
    exp_loss = float(np.mean(ref_losses))
    got_grad = ph.grad.numpy() * scale
    # l'implementation minimise la moyenne sur le batch : le gradient de chaque
    # echantillon vaut (1/batch) fois le sous-gradient d'Elmachtoub-Grigas
    exp_grad = np.asarray(ref_grads) / n

    err_loss = abs(got_loss - exp_loss) / max(abs(exp_loss), 1e-9)
    err_grad = float(np.max(np.abs(got_grad - exp_grad)))
    ok = err_loss < 1e-6 and err_grad < 1e-6
    print(f"    perte : impl={got_loss:.6f} ref={exp_loss:.6f} (err. rel. {err_loss:.2e})")
    print(f"    gradient : err. max = {err_grad:.2e} -> {'OK' if ok else 'ECHEC'}")
    return ok


def main() -> int:
    rng = np.random.default_rng(42)
    batt = BatterySpec()
    print("=" * 78)
    print("Verification SPO+ : conformite a la Definition 3 d'Elmachtoub & Grigas (2022)")
    print("=" * 78)
    results = [
        test_values_match_reference(rng, batt),
        test_zero_at_optimum(rng, batt),
        test_dominates_spo(rng, batt),
        test_gradient(rng, batt),
        test_convexity(rng, batt),
    ]
    ems = test_differentiable_ems()
    if ems is not None:
        results.append(ems)
    ok = all(results)
    print("\n" + "=" * 78)
    print("VERDICT :", "CONFORME a Elmachtoub-Grigas" if ok else "NON CONFORME -> correction necessaire")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
