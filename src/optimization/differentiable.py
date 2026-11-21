from __future__ import annotations

import warnings

import numpy as np
import torch

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer

    HAS_CVXPYLAYERS = True
except Exception:
    HAS_CVXPYLAYERS = False

SOLVE_METHOD = "ECOS"

_LAYER_CACHE: dict = {}


def build_ems_layer(horizon: int = 24, e_max: float = 100.0, e_min: float = 0.1,
                    p_ch: float = 30.0, p_dis: float = 30.0,
                    eta_ch: float = 0.95, eta_dis: float = 0.95, soc0: float = 0.5,
                    export_ratio: float = 0.5, g_max: float = 500.0,
                    x_max: float = 500.0):
    if not HAS_CVXPYLAYERS:
        raise ImportError("cvxpylayers indisponible (installer diffcp — voir .venv311)")
    key = (horizon, e_max, e_min, p_ch, p_dis, eta_ch, eta_dis, soc0, export_ratio, g_max, x_max)
    if key in _LAYER_CACHE:
        return _LAYER_CACHE[key]
    net = cp.Parameter(horizon, name="net")
    price = cp.Parameter(horizon, name="price")
    g = cp.Variable(horizon, nonneg=True)
    x = cp.Variable(horizon, nonneg=True)
    ch = cp.Variable(horizon, nonneg=True)
    dis = cp.Variable(horizon, nonneg=True)
    e = cp.Variable(horizon + 1)
    cons = [
        e[0] == soc0 * e_max,
        e[1:] == e[:-1] + eta_ch * ch - dis / eta_dis,
        e >= e_min * e_max,
        e <= e_max,
        ch <= p_ch,
        dis <= p_dis,
        g - x == net + ch - dis,
        g <= g_max,
        x <= x_max,
    ]
    cost = cp.sum(cp.multiply(price, g) - cp.multiply(price * export_ratio, x))
    prob = cp.Problem(cp.Minimize(cost), cons)
    layer = CvxpyLayer(prob, parameters=[net, price], variables=[g, x, ch, dis, e])
    _LAYER_CACHE[key] = layer
    return layer


def solve_with_layer(layer, net, price):
    g, x, ch, dis, e = layer(net, price, solver_args={"solve_method": SOLVE_METHOD})
    return {"import": g, "export": x, "charge": ch, "discharge": dis, "soc": e}


def _spo_exact_impl(layer, net_hat, price_hat, net_true, price_true,
                    scaler_mean, scaler_std, export_ratio: float = 0.5,
                    beta: float = 1.0, scale: float = 1.0) -> torch.Tensor:
    mean_t = torch.as_tensor(scaler_mean, dtype=net_hat.dtype, device=net_hat.device)
    std_t = torch.as_tensor(scaler_std, dtype=net_hat.dtype, device=net_hat.device)

    net_hat_r = net_hat * std_t[0] + mean_t[0]
    price_hat_r = price_hat * std_t[1] + mean_t[1]
    net_true_r = net_true * std_t[0] + mean_t[0]
    price_true_r = price_true * std_t[1] + mean_t[1]

    g1, x1, ch1, dis1, _ = layer(net_hat_r, price_hat_r, solver_args={"solve_method": SOLVE_METHOD})
    g2, x2, _, _, _ = layer(net_hat_r, 2.0 * price_hat_r - price_true_r,
                             solver_args={"solve_method": SOLVE_METHOD})

    spo_terms = (
        (price_true_r - 2.0 * price_hat_r) * (g2 - export_ratio * x2)
        + (2.0 * price_hat_r - price_true_r) * (g1 - export_ratio * x1)
    )
    spo = spo_terms.sum(dim=-1).clamp_min(0.0).mean()

    b = net_true_r + ch1 - dis1
    recourse = (price_true_r * (torch.relu(b) - export_ratio * torch.relu(-b))).sum(dim=-1).mean()

    return (spo + beta * recourse) / scale


def spo_exact_loss(layer, net_hat, price_hat, net_true, price_true,
                   scaler_mean, scaler_std, export_ratio: float = 0.5,
                   beta: float = 1.0, scale: float = 1.0) -> torch.Tensor:
    try:
        return _spo_exact_impl(layer, net_hat, price_hat, net_true, price_true,
                               scaler_mean, scaler_std, export_ratio, beta, scale)
    except Exception as exc:
        warnings.warn(f"couche EMS (cvxpylayers) en echec : {exc} — lot ignore")
        return net_hat.sum() * 0.0
